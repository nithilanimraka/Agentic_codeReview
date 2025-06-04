import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from pathlib import Path
from torch_geometric.data import Data
from tree_sitter import Language, Parser
from transformers import AutoTokenizer, AutoModel
import numpy as np
import traceback
from torch_geometric.nn import FAConv
from torch_geometric.nn import global_max_pool
from torch.nn import GRU     
from torch_geometric.data import Data, Batch
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now, retrieve the token from the environment
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)

GLOBAL_NODE_TYPES = sorted(list(set([
    # Common types from before
    'identifier', 'block', 'expression_statement', 'string', 'integer',
    'call', 'attribute', 'comment', 'ERROR',

    # Python specific 
    'module', 'function_definition', 'parameters', 'pass_statement',
    'if_statement', 'for_statement', 'import_statement', 'import_from_statement',
    'comparison_operator', 'decorated_definition', 'decorator',
    'list', 'tuple', 'dictionary', 'subscript', 'slice', 
    'while_statement', 'try_statement', 'class_definition', 'lambda',

    # Java specific 
    'program', 'class_declaration', 'method_declaration', 'modifiers',
    'type_identifier', 'void_type', 'formal_parameters', 'formal_parameter',
    'local_variable_declaration', 'method_invocation', 'field_access',
    'integer_literal', 'string_literal',

    # C/C++ specific (add many more if you use C/C++)
    'translation_unit', 'declaration', 'parameter_list',
    'compound_statement', 'field_expression', 'pointer_declarator',

    'assignment',
    '=', 
    'binary_operator', 
    'binary_expression', 
    '+',
    'return', 
    'return_statement', 
    '-',
    '%',
    '==', 
    'if', 
    'boolean_operator', 
    'or',
    ':', 
    '*', 
    'argument_list',
    '(', 
    ')', 

])))

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal Performance Shaders (MPS) device (GPU).")
elif torch.cuda.is_available(): # Included for completeness, unlikely on a Mac
    device = torch.device("cuda")
    print("Using NVIDIA CUDA device (GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# Create a mapping from type name to index
GLOBAL_TYPE_MAP = {t: i for i, t in enumerate(GLOBAL_NODE_TYPES)}
GLOBAL_TYPE_COUNT = len(GLOBAL_NODE_TYPES)

#print(f"Initialized Global Node Type Vocabulary with {GLOBAL_TYPE_COUNT} types.")


def _load_parsers(languages):
    parsers = {}
    extension = '.dll' if os.name == 'nt' else '.so'
    
    for lang in languages:
        try:
            lang_path = os.path.abspath(f'src/duplicate_check/build/{lang}{extension}')
            print(f"Attempting to load {lang} from {lang_path}")
            
            if not os.path.exists(lang_path):
                print(f"Error: {lang_path} does not exist")
                continue
                
            lang_object = Language(lang_path, lang)
            lang_parser = Parser()
            lang_parser.set_language(lang_object)
            parsers[lang] = lang_parser
            print(f"Successfully loaded {lang} parser")
            
        except Exception as e:
            print(f"Error loading {lang} parser: {e}")
    
    return parsers

def _extract_units(tree, code, filepath, lang):
    units = []
    root_node = tree.root_node
    
    if lang == 'python':
        query_pattern = """
        (function_definition
            name: (identifier) @function_name
            body: (block) @function_body) @function
        """
    elif lang == 'java':
        query_pattern = """
        (method_declaration
            name: (identifier) @method_name
            body: (block) @method_body) @method
        """
    else:
        return units
    
    extension = '.dll' if os.name == 'nt' else '.so'
    language = Language(f'src/duplicate_check/build/{lang}{extension}', lang) 
    
    query = language.query(query_pattern)
    captures = query.captures(root_node)
    
    # Group captures by their type
    grouped = {}
    for node, tag in captures:
        if tag not in grouped:
            grouped[tag] = []
        grouped[tag].append(node)
    
    # Pair names with bodies
    if lang == 'python':
        functions = grouped.get('function', [])
        names = grouped.get('function_name', [])
        bodies = grouped.get('function_body', [])
    else:  # java
        functions = grouped.get('method', [])
        names = grouped.get('method_name', [])
        bodies = grouped.get('method_body', [])
    
    for i in range(len(functions)):
        # Get the function/method name
        name_node = names[i]
        unit_name = code[name_node.start_byte:name_node.end_byte].decode('utf8')
        
        # Get the function/method body
        body_node = bodies[i]
        unit_code = code[body_node.start_byte:body_node.end_byte].decode('utf8')

        # Extract AST 
        unit_parser = Parser()
        unit_parser.set_language(language)
        unit_tree = unit_parser.parse(unit_code.encode())
        unit_ast = unit_tree.root_node  # Root of the AST 
        
        # Get the full function/method node for start/end lines
        func_node = functions[i]
        start_line = func_node.start_point[0] + 1  
        end_line = func_node.end_point[0] + 1
        
        units.append({
            'unit_code': unit_code,
            'start_line': start_line,
            'end_line': end_line,
            'unit_name': unit_name,
            'filepath': filepath,
            'ast': unit_ast,  # Store the AST for graph construction
            'lang': lang     
        })
    
    return units

def parse_directory(dir_path, parsers):
    code_units = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            path = Path(root) / file
            if path.suffix in ['.py', '.java', '.cpp', '.h']:
                lang = 'python' if path.suffix == '.py' else \
                       'java' if path.suffix == '.java' else 'cpp'
                if lang not in parsers:
                    continue
                try:
                    with open(path, 'rb') as f:
                        code = f.read()
                    tree = parsers[lang].parse(code)
                    code_units.extend(_extract_units(tree, code, str(path), lang))
                except Exception as e:
                    print(f"Error parsing {path}: {e}")
    return code_units

def refine_ast(node, code):
    """Refine AST by merging shadow nodes and reconstructing parent nodes"""
    
    node_content_str = ""
    try:
        node_content_str = code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
    except Exception as e:
         print(f"Warning: Exception during node content decoding (node type: {node.type}): {e}")
         node_content_str = ""

    new_node = {
        'type': node.type,
        'children': [],
        'content': node_content_str
    }

    if not node.children:
        return new_node
    # Process children
    for child in node.children:
        refined_child = refine_ast(child, code)
        
        # Skip shadow nodes 
        if (refined_child['type'] == node.type and 
            refined_child['content'] == new_node['content']):
            continue
            
        new_node['children'].append(refined_child)

    if node.type in ['function_definition', 'method_declaration']:
        parts = []
        for child in node.children:
            if child.type != 'block':
                parts.append(code[child.start_byte:child.end_byte].decode('utf8'))
        new_node['content'] = ' '.join(parts)
    
    return new_node

# Load model directly
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-110m-embedding')
model = AutoModel.from_pretrained('Salesforce/codet5p-110m-embedding', trust_remote_code=True)
content_model = model
print(f"DEBUG: Forcing model to use device: {device}")
content_model.to(device)


def ast_to_graph(refined_ast, lang, tokenizer, content_model, type_map, type_count): 
    """Convert refined AST dictionary to graph with semantic embeddings and type embeddings."""
    print(f"\n--- Processing AST for language: {lang} ---")
    G = nx.DiGraph()
    nodes_added_count = 0
    nodes_with_embedding_count = 0

    if content_model is None or not hasattr(content_model, 'config') or not hasattr(content_model.config, 'hidden_size'):
        print("Error: content_model or its config is not properly initialized.")
        return G 

    TRANSFORMER_HIDDEN_SIZE = content_model.config.hidden_size
    if TRANSFORMER_HIDDEN_SIZE is None and hasattr(content_model.config, 'd_model'): 
        TRANSFORMER_HIDDEN_SIZE = content_model.config.d_model
    if TRANSFORMER_HIDDEN_SIZE is None:
        print("FATAL: Could not determine transformer hidden size from model config.")
        TRANSFORMER_HIDDEN_SIZE = 256
        print(f"Warning: Assuming transformer hidden size is {TRANSFORMER_HIDDEN_SIZE}")


    EXPECTED_EMBEDDING_LEN = TRANSFORMER_HIDDEN_SIZE + type_count


    def traverse(node, parent_id=None):
        nonlocal nodes_added_count, nodes_with_embedding_count
        node_id = len(list(G.nodes())) 
        node_type = node.get('type', 'UnknownType')
        node_content = node.get('content', '')

        # --- Content Embedding ---
        content_embedding = np.zeros(TRANSFORMER_HIDDEN_SIZE, dtype=np.float32) 
        if node_content.strip():
            try:
                inputs = tokenizer(node_content, return_tensors='pt', truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = content_model(**inputs)  
                if isinstance(outputs, torch.Tensor):
                    embedding_tensor = outputs.mean(dim=1).squeeze()
                elif hasattr(outputs, 'last_hidden_state'):
                    embedding_tensor = outputs.last_hidden_state.mean(dim=1).squeeze()
                else:
                    print(f"Warning: Unknown output type from content_model: {type(outputs)}. Cannot extract embedding. Using zeros.")
                    embedding_tensor = torch.zeros(TRANSFORMER_HIDDEN_SIZE, device=inputs['input_ids'].device if 'input_ids' in inputs else 'cpu')


                if embedding_tensor.ndim == 0:
                    content_embedding = np.zeros(TRANSFORMER_HIDDEN_SIZE, dtype=np.float32)
                else:
                    content_embedding_np = embedding_tensor.cpu().numpy().astype(np.float32)
                    if content_embedding_np.shape == (TRANSFORMER_HIDDEN_SIZE,):
                        content_embedding = content_embedding_np
                    elif content_embedding_np.ndim == 2 and content_embedding_np.shape[0] == 1 and content_embedding_np.shape[1] == TRANSFORMER_HIDDEN_SIZE:
                        content_embedding = content_embedding_np.squeeze(0)
                    elif content_embedding_np.shape == (): 
                         print(f"Warning: Embedding tensor became scalar for node {node_id}. Content: '{node_content[:30]}...'. Using zeros.")
                         content_embedding = np.zeros(TRANSFORMER_HIDDEN_SIZE, dtype=np.float32)
                    else:
                        print(f"Warning: Unexpected content_embedding shape {content_embedding_np.shape} for node {node_id}. Expected ({TRANSFORMER_HIDDEN_SIZE},). Content: '{node_content[:30]}...'. Using zeros.")
                        content_embedding = np.zeros(TRANSFORMER_HIDDEN_SIZE, dtype=np.float32)
            except Exception as e:
                print(f" ERROR generating content embedding for node_id: {node_id}")
                print(f"Node Type: {node_type}, Content: '{node_content[:50]}...'")
                print(f"Exception: {type(e).__name__} - {e}")
                traceback.print_exc() 
                content_embedding = np.zeros(TRANSFORMER_HIDDEN_SIZE, dtype=np.float32)

        #Type Embedding
        # Use the passed type_map and type_count arguments
        type_embedding = np.zeros(type_count, dtype=np.float32)
        type_idx = type_map.get(node_type, -1)
        if type_idx != -1:
            type_embedding[type_idx] = 1.0 

        #Combine Embeddings 
        node_embedding_to_store = None
        try:
            node_embedding = np.concatenate([content_embedding, type_embedding])
            if node_embedding.shape == (EXPECTED_EMBEDDING_LEN,):
                node_embedding_to_store = node_embedding
                nodes_with_embedding_count += 1
            else:
                 print(f"ERROR: Final embedding shape mismatch for node {node_id}! Got {node_embedding.shape}, Expected ({EXPECTED_EMBEDDING_LEN},). Storing None.")
        except ValueError as e_concat:
             print(f"ERROR concatenating embeddings for node {node_id}: {e_concat}. Storing None.")


        #Add Node to Graph
        G.add_node(
            node_id,
            type=node_type,
            content=node_content,
            embedding=node_embedding_to_store # Store the potentially None embedding
        )
        nodes_added_count += 1

        #Add edge from child to parent (reversed edge)
        if parent_id is not None:
            G.add_edge(node_id, parent_id) # Edge direction: child -> parent

        #Process children
        children = node.get('children', [])
        for child in children:
            if isinstance(child, dict):
                traverse(child, node_id)

    #Initial call to traverse 
    if refined_ast and isinstance(refined_ast, dict):
        print(f"Starting AST traversal for root type: {refined_ast.get('type', 'Unknown')}")
        traverse(refined_ast, parent_id=None) # Start traversal from the root dictionary node
    else:
        print("Warning: refined_ast input was invalid. Graph will be empty.")


    # Final Summary Print 
    print(f"Finished processing AST. Graph created with {nodes_added_count} nodes. ---")
    print(f"Nodes with successfully generated embedding: {nodes_with_embedding_count} / {nodes_added_count}")

    # Add check for graph consistency before returning
    if nodes_with_embedding_count == 0 and nodes_added_count > 0:
        print("Warning: Graph has nodes but none have valid embeddings.")
    return G

#Re-implemented ASTGPool Class with Manual Edge Filtering 
class ASTGPool(nn.Module):
    """
    Custom pooling layer for ASTs based on child count (or other custom score).
    Selects top-k nodes based on the score and filters the graph.
    Manually implements top-k selection AND edge filtering to avoid PyG import issues.
    """
    def __init__(self, in_channels, ratio=0.5, min_score=None, multiplier=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = float(ratio) if ratio is not None else 1.0
        self.min_score = min_score
        self.multiplier = multiplier

    def compute_score(self, x, edge_index, batch=None):
        """Calculates the importance score for each node."""
        device = x.device
        if x.size(0) == 0:
            return torch.empty(0, device=device)
        # Score based on in-degree
        target_nodes = edge_index[0]
        degree = torch.zeros(x.size(0), dtype=torch.float32, device=device)
        ones = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
        degree.scatter_add_(0, target_nodes, ones)
        score = degree
        if self.min_score is not None:
             score = score.where(score >= self.min_score, torch.tensor(-1e10, device=device))
        return score

    def filter_edges_manual(self, edge_index, perm, num_nodes):
        """
        Manually filters edges to keep only those where both endpoints are in perm.
        Also remaps the node indices in the edge_index.
        Args:
            edge_index: Original edge index (2, E).
            perm: Tensor of node indices to keep (N_keep,).
            num_nodes: Original number of nodes.
        Returns:
            Filtered and remapped edge_index (2, E_keep).
        """
        if perm.numel() == 0:
             return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)

        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=perm.device)
        node_mask[perm] = True
        source_nodes_kept = node_mask[edge_index[1]]
        target_nodes_kept = node_mask[edge_index[0]]
        edge_mask = source_nodes_kept & target_nodes_kept

        # Filter the edge_index
        filtered_edge_index = edge_index[:, edge_mask]

        node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=perm.device)
        node_map[perm] = torch.arange(perm.size(0), device=perm.device)

        remapped_edge_index = node_map[filtered_edge_index]

        return remapped_edge_index


    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Applies pooling based on computed scores using torch.topk and manual edge filtering.
        """
        num_nodes = x.size(0)
        if num_nodes == 0: 
            empty_perm = torch.empty(0, dtype=torch.long, device=x.device)
            empty_batch = torch.empty(0, dtype=torch.long, device=x.device) if batch is not None else None
            empty_score = torch.empty(0, device=x.device)
            return x, edge_index, edge_attr, empty_batch, empty_perm, empty_score

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        score = self.compute_score(x, edge_index, batch)

        unique_batch_ids, nodes_per_graph = torch.unique(batch, return_counts=True)
        k_per_graph = (nodes_per_graph.float() * self.ratio).ceil().long()

        perm_list = []
        start_index = 0
        for i in range(unique_batch_ids.size(0)):
            num_nodes_in_graph = nodes_per_graph[i]
            if num_nodes_in_graph == 0: continue 
            k = k_per_graph[i]
            k = max(1, min(k, num_nodes_in_graph))

            graph_scores = score[start_index : start_index + num_nodes_in_graph]
            _, topk_indices_in_graph = torch.topk(graph_scores, k, dim=0)
            perm_list.append(topk_indices_in_graph + start_index)
            start_index += num_nodes_in_graph

        if not perm_list:
             perm_tensor = torch.empty(0, dtype=torch.long, device=x.device)
        else:
             perm_tensor = torch.cat(perm_list, dim=0)

        x_filtered = x[perm_tensor] * self.multiplier
        batch_filtered = batch[perm_tensor]
        score_filtered = score[perm_tensor]

        edge_index_filtered = self.filter_edges_manual(edge_index, perm_tensor, num_nodes)
        edge_attr_filtered = None
        if edge_attr is not None:
            node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=perm_tensor.device)
            node_mask[perm_tensor] = True
            source_nodes_kept = node_mask[edge_index[1]]
            target_nodes_kept = node_mask[edge_index[0]]
            edge_mask = source_nodes_kept & target_nodes_kept
            edge_attr_filtered = edge_attr[edge_mask]

        return x_filtered, edge_index_filtered, edge_attr_filtered, batch_filtered, perm_tensor, score_filtered

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, pooling_ratio=0.5, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim # Dimension after GRU
        self.output_dim = output_dim # Final embedding dimension
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        current_dim = input_dim
        for i in range(num_layers):
            self.convs.append(FAConv(channels=current_dim, eps=0.1, dropout=dropout))
            self.pools.append(ASTGPool(in_channels=current_dim, ratio=pooling_ratio))

        self.gru = GRU(input_size=self.input_dim * num_layers, 
                       hidden_size=self.hidden_dim,         
                       batch_first=True)

        self.final_lin = nn.Linear(self.hidden_dim, self.output_dim) 

        # Projection layer for the root embedding
        self.root_proj = nn.Linear(self.input_dim, self.output_dim) 

        self.residual_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, data, root_node_initial_embedding):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if x is None or x.size(0) == 0:
             print("Error: GNNEncoder received data with no node features (x is None or empty).")
             num_graphs = data.num_graphs if hasattr(data, 'num_graphs') else 1
             return torch.zeros((num_graphs, self.output_dim), device=data.edge_index.device if data.edge_index is not None else self.residual_alpha.device)

        layer_global_outputs = []
        original_x = x

        current_x = x
        current_edge_index = edge_index
        current_batch = batch
        current_original_x = original_x

        for i in range(self.num_layers):
            if current_x.size(0) == 0:
                print(f"Warning: Skipping GNN layer {i} due to empty input after pooling.")
                break

            conv_x = self.convs[i](current_x, current_original_x, current_edge_index)
            conv_x = F.relu(conv_x)
            conv_x = F.dropout(conv_x, p=self.dropout, training=self.training)

            global_representation = global_max_pool(conv_x, current_batch)
            layer_global_outputs.append(global_representation)

            pool_result = self.pools[i](x=conv_x, edge_index=current_edge_index, batch=current_batch)
            pooled_x = pool_result[0]
            pooled_edge_index = pool_result[1]
            pooled_batch = pool_result[3]
            perm = pool_result[4]

            current_x = pooled_x
            current_edge_index = pooled_edge_index
            current_batch = pooled_batch

            if current_x.size(0) > 0:
                 if current_original_x.shape[0] == data.num_nodes: 
                     current_original_x = current_original_x[perm]
                 else:
                      if perm.max() < current_original_x.shape[0]:
                           current_original_x = current_original_x[perm]
                      else:
                           print(f"Warning: Permutation index out of bounds for current_original_x in layer {i}. Shape: {current_original_x.shape}, Max perm index: {perm.max()}")

            else:
                print(f"Info: Pooling removed all nodes after layer {i}.")
                break

        if not layer_global_outputs:
            print("Error: No global outputs collected from GNN layers.")
            num_graphs = data.num_graphs if hasattr(data, 'num_graphs') else 1
            return torch.zeros((num_graphs, self.output_dim), device=x.device)

        try:
             batch_sizes = [o.shape[0] for o in layer_global_outputs]
             if len(set(batch_sizes)) > 1:
                 print(f"Warning: Inconsistent batch sizes found in layer_global_outputs: {batch_sizes}. Using outputs matching the first layer.")
                 ref_batch_size = batch_sizes[0]
                 layer_global_outputs = [o for o in layer_global_outputs if o.shape[0] == ref_batch_size]
                 if not layer_global_outputs:
                      raise RuntimeError("No consistent batch size outputs found.")

             fused_input = torch.cat(layer_global_outputs, dim=1)
        except RuntimeError as e_cat:
             print(f"Error during concatenation of layer outputs: {e_cat}")
             print(f"Shapes were: {[o.shape for o in layer_global_outputs]}")
             num_graphs = data.num_graphs if hasattr(data, 'num_graphs') else 1
             return torch.zeros((num_graphs, self.output_dim), device=x.device)

        gru_input = fused_input.unsqueeze(1)
        gru_output, _ = self.gru(gru_input)
        gru_output = gru_output.squeeze(1)

        final_gnn_embedding = self.final_lin(gru_output)

        final_embedding = final_gnn_embedding 
        if root_node_initial_embedding is not None:
            try:
                if not isinstance(root_node_initial_embedding, torch.Tensor):
                    root_tensor = torch.tensor(root_node_initial_embedding, dtype=torch.float32)
                else:
                    root_tensor = root_node_initial_embedding.to(dtype=torch.float32)

                root_tensor = root_tensor.to(final_gnn_embedding.device)

                if root_tensor.shape[-1] != self.input_dim:
                     print(f"ERROR: Root node embedding has wrong feature dimension. Expected {self.input_dim}, Got {root_tensor.shape[-1]}. Skipping residual.")
                else:
                    if root_tensor.dim() == 1 and final_embedding.dim() == 2:
                        root_tensor = root_tensor.unsqueeze(0)
                    elif root_tensor.dim() == 2 and final_embedding.dim() == 2:
                         if root_tensor.shape[0] == 1 and final_embedding.shape[0] > 1:
                              root_tensor = root_tensor.expand(final_embedding.shape[0], -1)
                         elif root_tensor.shape[0] != final_embedding.shape[0]:
                              print(f"Warning: Batch size mismatch between root ({root_tensor.shape[0]}) and final ({final_embedding.shape[0]}) after potential unsqueeze. Skipping residual.")
                              root_tensor = None
                    elif root_tensor.dim() != final_embedding.dim():
                         print(f"Warning: Dimensionality mismatch between root ({root_tensor.dim()}D) and final ({final_embedding.dim()}D). Skipping residual.")
                         root_tensor = None 

                    if root_tensor is not None:
                        projected_root = self.root_proj(root_tensor) 

                        if projected_root.shape == final_embedding.shape:
                            alpha = torch.sigmoid(self.residual_alpha)
                            final_embedding = alpha * final_gnn_embedding + (1 - alpha) * projected_root
                        else:
                            print(f"Warning: Projected root shape {projected_root.shape} != Final GNN shape {final_embedding.shape}. Skipping residual.")

            except Exception as e_resid:
                print(f"Error during residual connection processing: {type(e_resid).__name__} - {e_resid}. Skipping residual.")
                traceback.print_exc()
        else:
            print("Warning: No root node initial embedding provided for residual connection.")

        return final_embedding


def nx_to_pyg(G, gnn_encoder_model): # Pass the initialized GNN model
    """Convert NetworkX graph to PyG Data object and apply the FULL GNN Encoder."""
    print(f"\n>>> nx_to_pyg started for graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    final_embedding_np = None
    node_features = []
    node_mapping = {}
    root_node_initial_embedding = None
    embedding_len = -1 

    try:
        if not isinstance(G, nx.DiGraph) or G.number_of_nodes() == 0:
            print("nx_to_pyg: Input G is not a valid non-empty NetworkX DiGraph. Returning None.")
            return None

        sorted_nodes = sorted(G.nodes()) 
        root_node_id = 0 
        nodes_with_valid_embedding = 0

        for i, node_id in enumerate(sorted_nodes):
            node_mapping[node_id] = i
            try:
                node_data = G.nodes[node_id]
                embedding_retrieved = node_data.get('embedding') 

                if embedding_retrieved is not None and isinstance(embedding_retrieved, np.ndarray) and embedding_retrieved.ndim == 1:
                    current_len = embedding_retrieved.shape[0]
                    if embedding_len == -1: 
                        embedding_len = current_len
                        print(f"nx_to_pyg: Determined embedding length = {embedding_len} from node {node_id}")
                    elif current_len != embedding_len:
                        print(f" FATAL ERROR: Node {node_id}: Inconsistent embedding lengths! Expected {embedding_len}, got {current_len}. Skipping node.")
                        node_features.append(np.zeros(embedding_len, dtype=np.float32)) 
                        continue 

                    node_features.append(embedding_retrieved.astype(np.float32))
                    nodes_with_valid_embedding += 1
                    if node_id == root_node_id: 
                       root_node_initial_embedding = embedding_retrieved.astype(np.float32)
                       print(f"nx_to_pyg: Found initial embedding for root node {root_node_id}.")

                else:
                    print(f" Warning: Node {node_id} has missing or invalid embedding (type: {type(embedding_retrieved)}). Appending zeros.")
                    if embedding_len != -1:
                       node_features.append(np.zeros(embedding_len, dtype=np.float32))
                    elif i == len(sorted_nodes) - 1 and nodes_with_valid_embedding == 0:
                         print("FATAL ERROR: No valid node embeddings found in the graph.")
                         return None

            except Exception as e_node:
                print(f"  Error processing node {node_id}: {e_node}. Appending zeros if possible.")
                if embedding_len != -1:
                   node_features.append(np.zeros(embedding_len, dtype=np.float32))

        print(f"nx_to_pyg: Node feature extraction loop finished. Nodes processed: {len(sorted_nodes)}. Valid embeddings found: {nodes_with_valid_embedding}.")

        if nodes_with_valid_embedding == 0: 
            print("nx_to_pyg: ERROR - No nodes with valid embeddings found. Cannot proceed.")
            return None
        try:
            x = torch.tensor(np.array(node_features), dtype=torch.float)
            print(f"nx_to_pyg: Tensor 'x' created with shape {x.shape}.")
            if gnn_encoder_model and x.shape[1] != gnn_encoder_model.input_dim:
                 print(f"FATAL ERROR: Tensor 'x' feature dimension ({x.shape[1]}) != GNN input dimension ({gnn_encoder_model.convs[0].in_channels})")
                 return None
        except Exception as e_tensor:
            print(f"nx_to_pyg: Error creating tensor 'x': {e_tensor}")
            traceback.print_exc()
            return None
        edge_list = []
        for src, dst in G.edges():
             if src in node_mapping and dst in node_mapping:
                 edge_list.append([node_mapping[src], node_mapping[dst]])
             else:
                 print(f"Warning: Skipping edge ({src}, {dst}) due to missing node in mapping.")

        if not edge_list:
            print("nx_to_pyg: Warning - Graph has nodes but no edges.")
            edge_index = torch.empty((2, 0), dtype=torch.long) 
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        print(f"nx_to_pyg: Edge index created with shape {edge_index.shape}.")

        data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
        print(f"nx_to_pyg: Data object created. Nodes={data.num_nodes}, Edges={data.num_edges}.")

        if gnn_encoder_model is None:
             print("Error: gnn_encoder_model not provided to nx_to_pyg. Cannot generate embedding.")
             return None

        gnn_encoder_model.eval()
        try:
            with torch.no_grad(): 
                # Pass data and root embedding to the GNN model
                final_embedding_tensor = gnn_encoder_model(data, root_node_initial_embedding)

            if final_embedding_tensor is not None:
                final_embedding_np = final_embedding_tensor.squeeze().cpu().numpy() # Squeeze batch dim if present
                print(f"nx_to_pyg: GNN processing successful. Output embedding shape: {final_embedding_np.shape}")
            else:
                print("nx_to_pyg: GNN model returned None.")

        except Exception as e_gnn:
            print(f"nx_to_pyg: Error during GNN forward pass: {type(e_gnn).__name__} - {e_gnn}")
            traceback.print_exc()
            return None 
        print(f"<<< nx_to_pyg finished.")
        return final_embedding_np

    except Exception as e_outer:
        print(f"Error converting graph (Outer Try/Except in nx_to_pyg): {e_outer}")
        traceback.print_exc()
        return None

#Revised detect_duplicates to handle potential issues
def detect_duplicates_revised(functions_with_embeddings, threshold=0.85):
    """Identify duplicate functions based on embedding similarity (handles potential None/shape issues)"""
    embeddings = [f.get('embedding') for f in functions_with_embeddings]

    valid_embeddings = []
    original_indices = []
    for idx, emb in enumerate(embeddings):
        if isinstance(emb, np.ndarray):
            valid_embeddings.append(emb)
            original_indices.append(idx)
        else:
            print(f"Warning: Skipping function at index {idx} due to invalid embedding.")

    if len(valid_embeddings) < 2:
        print("Need at least two valid embeddings to compare.")
        return set()

    try:
        embeddings_array = np.array(valid_embeddings)
        if embeddings_array.ndim == 1:
             print("Warning: Only one valid embedding left after filtering Nones.")
             return set()
        if len(set(emb.shape[0] for emb in embeddings_array)) > 1:
             print("Error: Embeddings have inconsistent feature dimensions.")
             return set()
    except Exception as e_stack:
        print(f"Error preparing embeddings for similarity calculation: {e_stack}")
        return set()



    similarities = cosine_similarity(embeddings_array)

    duplicates = set()
    num_valid = len(valid_embeddings)
    for i in range(num_valid):
        for j in range(i + 1, num_valid):
            if similarities[i, j] >= threshold:
                original_idx_i = original_indices[i]
                original_idx_j = original_indices[j]
                duplicates.add((original_idx_i, original_idx_j, similarities[i, j]))

    return duplicates

def revised_pipeline(codebase_path, languages=['python', 'java'], threshold=0.85):
    """Revised pipeline integrating corrections"""

    try:
        global device 

        local_tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-110m-embedding')
        local_model = AutoModel.from_pretrained('Salesforce/codet5p-110m-embedding', trust_remote_code=True) 
        local_content_model = local_model
        local_content_model.to(device) 

        config = local_content_model.config
        TRANSFORMER_HIDDEN_SIZE = config.hidden_size 
        print(f"Using Transformer Hidden Size: {TRANSFORMER_HIDDEN_SIZE}")

        GLOBAL_TYPE_COUNT = len(GLOBAL_NODE_TYPES)
        EXPECTED_EMBEDDING_LEN = TRANSFORMER_HIDDEN_SIZE + GLOBAL_TYPE_COUNT

        GNN_HIDDEN_DIM = TRANSFORMER_HIDDEN_SIZE 
        GNN_OUTPUT_DIM = TRANSFORMER_HIDDEN_SIZE 

        gnn_encoder = GNNEncoder(input_dim=EXPECTED_EMBEDDING_LEN, 
                                 hidden_dim=GNN_HIDDEN_DIM,      
                                 output_dim=GNN_OUTPUT_DIM,      
                                 num_layers=3, pooling_ratio=0.7)

    except Exception as e_init:
        print(f"FATAL: Failed to initialize models: {e_init}")
        traceback.print_exc()
        return []

    # Code ingestion & Initial AST
    parsers = _load_parsers(languages)
    print(f"Loaded {len(parsers)} parsers.")
    functions = parse_directory(codebase_path, parsers)
    print(f"Extracted {len(functions)} functions.")

    processed_functions = []
    for i, func_unit in enumerate(functions):
        print(f"\n--- Processing function {i+1}/{len(functions)}: {func_unit['unit_name']} in {func_unit['filepath']} ---")
        try:
            raw_code_bytes = func_unit.get('raw_code', func_unit['unit_code'].encode()) 
            refined_ast_dict = refine_ast(func_unit['ast'], raw_code_bytes) 
            func_unit['refined_ast'] = refined_ast_dict 


            graph = ast_to_graph(refined_ast_dict, func_unit['lang'], local_tokenizer, local_content_model, GLOBAL_TYPE_MAP, GLOBAL_TYPE_COUNT)
            func_unit['graph'] = graph 

            if graph is None or graph.number_of_nodes() == 0:
                 print("Skipping GNN embedding due to empty or invalid graph.")
                 continue


            final_embedding = nx_to_pyg(graph, gnn_encoder)
            func_unit['embedding'] = final_embedding 

            if final_embedding is not None:
                print(f"Successfully generated embedding with shape {final_embedding.shape}")
                processed_functions.append(func_unit) 
            else:
                print("Failed to generate final embedding for this function.")

        except Exception as e_func:
            print(f"ERROR processing function {func_unit['unit_name']}: {e_func}")
            traceback.print_exc()
            continue 


    print(f"\n--- Finished processing all functions. Got {len(processed_functions)} valid embeddings. ---")

    #Duplicate detection
    if not processed_functions:
        print("No functions processed successfully. Cannot detect duplicates.")
        return []

    print(f"Detecting duplicates among {len(processed_functions)} functions...")
    valid_funcs_for_detection = [f for f in processed_functions if f.get('embedding') is not None]

    # Check if embeddings are numpy arrays before passing
    embeddings_list = [f['embedding'] for f in valid_funcs_for_detection]
    if not all(isinstance(emb, np.ndarray) for emb in embeddings_list):
         print("Error: Not all embeddings are numpy arrays. Cannot compute similarity.")
         valid_embeddings = [emb for emb in embeddings_list if isinstance(emb, np.ndarray)]
         if not valid_embeddings: return []

    duplicates = detect_duplicates_revised(valid_funcs_for_detection, threshold)

    results = []
    for i, j, sim_score in duplicates: 
        results.append({
            'function1': valid_funcs_for_detection[i]['unit_name'],
            'file1': valid_funcs_for_detection[i]['filepath'],
            'function2': valid_funcs_for_detection[j]['unit_name'],
            'file2': valid_funcs_for_detection[j]['filepath'],
            'similarity': sim_score
        })

    print(f"Found {len(results)} duplicate pairs.")
    return results


