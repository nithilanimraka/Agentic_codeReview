import os
import re
import json
import hmac
import hashlib
import requests
import numpy as np
import logging
from threading import Lock
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import tiktoken
import google.generativeai as genai
from tree_sitter import Language, Parser
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
GITHUB_REPO = os.getenv("GITHUB_REPO")
REPO_OWNER = os.getenv("REPO_OWNER")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_TOKENS_PER_CHUNK = 960000

# Tree-sitter configuration
TREE_SITTER_LIB = str(Path(__file__).parent / 'build' / 'my-languages.so')


# Add verification
if not os.path.exists(TREE_SITTER_LIB):
    logger.error(f"Tree-sitter library not found at {TREE_SITTER_LIB}")
    USE_TREE_SITTER = False
else:
    USE_TREE_SITTER = True
    logger.info(f"Found Tree-sitter library at {TREE_SITTER_LIB}")

# Language mapping
LANGUAGE_MAP = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".cs": "c_sharp",
    ".php": "php",
    ".html": "html",
    ".css": "css",
    ".ts": "typescript",
    ".cpp": "cpp",
    ".json": "json",
}

# Initialize parsers
PARSERS = {}

# Initialize FastAPI app
app = FastAPI()

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# --- Helper Functions ---

def count_tokens(text: str, model_name: str = "gemini-1.5-flash") -> int:
    """
    Estimate token count for Gemini and GPT models.
    Uses tiktoken for GPT, and a 3.5 chars/token approximation for Gemini.
    """
    if model_name.startswith("gpt-"):
        import tiktoken
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    elif model_name.startswith("gemini"):
        # Gemini models (1.5 Flash, 1.5 Pro, 2.0 etc.) use approx. 3.5 chars/token
        return int(len(text) / 3.5)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_language_by_extension(filename):
    for ext, lang in LANGUAGE_MAP.items():
        if filename.endswith(ext):
            return lang
    return None

def get_parser_for_language(lang_name):
    if lang_name not in PARSERS:
        parser = Parser()
        language = Language(TREE_SITTER_LIB, lang_name)
        parser.set_language(language)
        PARSERS[lang_name] = parser
    return PARSERS[lang_name]

def extract_blocks(code_text, lang_name):
    parser = get_parser_for_language(lang_name)
    tree = parser.parse(bytes(code_text, "utf8"))
    root = tree.root_node
    blocks = []

    def collect_blocks(node):
        if lang_name == "python" and node.type in ("function_definition", "class_definition"):
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        elif lang_name == "java" and node.type in ("method_declaration", "class_declaration", "interface_declaration"):
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        elif lang_name == "javascript" and node.type in ("function_declaration", "class_declaration"):
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        elif lang_name == "c_sharp" and node.type in ("method_declaration", "class_declaration"):
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        elif lang_name == "php" and node.type in ("function_declaration", "class_declaration"):
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        elif lang_name == "typescript" and node.type in ("function_declaration", "class_declaration"):
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        elif lang_name == "cpp" and node.type in ("function_definition", "class_definition"):
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        elif lang_name == "html" and node.type == "element":
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        elif lang_name == "css" and node.type == "rule_set":
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        elif lang_name == "json" and node.type == "object":
            blocks.append(code_text[node.start_byte:node.end_byte].strip())
        
        for child in node.children:
            collect_blocks(child)

    collect_blocks(root)
    return blocks

def chunk_diff_by_blocks(diff_content):
    chunks = []
    file_diffs = re.split(r'(diff --git a/.* b/.*)', diff_content)
    paired_diffs = []

    for i in range(1, len(file_diffs), 2):
        header = file_diffs[i]
        content = file_diffs[i + 1] if i + 1 < len(file_diffs) else ''
        paired_diffs.append((header.strip(), content.strip()))

    for header, content in paired_diffs:
        match = re.search(r'diff --git a/(.*) b/', header)
        filename = match.group(1) if match else "unknown"
        lang = get_language_by_extension(filename)

        full_diff = header + '\n' + content
        if not lang or count_tokens(full_diff) < MAX_TOKENS_PER_CHUNK:
            if chunks and count_tokens(chunks[-1]) + count_tokens(full_diff) <= MAX_TOKENS_PER_CHUNK:
                chunks[-1] += "\n" + full_diff
            else:
                chunks.append(full_diff)
            continue

        added_lines = []
        for line in content.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])
            elif not line.startswith('-') and not line.startswith('---'):
                added_lines.append(line)

        added_code = "\n".join(added_lines)
        blocks = extract_blocks(added_code, lang)

        current_chunk = [header]
        token_count = count_tokens(header)

        for block in blocks:
            block_tokens = count_tokens(block)
            if token_count + block_tokens > MAX_TOKENS_PER_CHUNK:
                chunks.append("\n".join(current_chunk))
                current_chunk = [header]
                token_count = count_tokens(header)
            current_chunk.append(block)
            token_count += block_tokens

        if current_chunk:
            current_chunk_str = "\n".join(current_chunk)
            if chunks and count_tokens(chunks[-1]) + count_tokens(current_chunk_str) <= MAX_TOKENS_PER_CHUNK:
                chunks[-1] += "\n" + current_chunk_str
            else:
                chunks.append(current_chunk_str)

    return chunks

def summarize_chunk(chunk):
    model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config={"temperature": 0.3})
    prompt = (
        "Analyze the following git diff and summarize it clearly. For each change, explain:\n"
        "1. What was modified or added (the code difference).\n"
        "2. What problem or error it addresses (if any).\n"
        "3. What improvement or benefit the change provides.\n\n"
        "Git diff:\n"
        f"{chunk}"
    )
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error summarizing chunk: {str(e)}")
        return "Summary error"

def generate_final_summary(pr_title, chunk_summaries, pr_diff):
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.3)
    
    prompt_template = """
**🔍 Comprehensive PR Analysis Report**

**PR Title:** {title}

**📝 Code Changes Summary:**
{summaries}

**🔎 Detailed Analysis:**
1. **Impact Assessment**: Analyze the overall impact of these changes
2. **Potential Issues**: Highlight any potential problems or risks
3. **Improvement Suggestions**: Provide actionable suggestions for improvement
4. **Best Practices Check**: Verify if changes follow coding standards

**Diff Overview (First 100 lines):**

**📌 Final Recommendations:**
- Provide clear, actionable next steps
- Highlight critical changes that need attention
- Suggest any follow-up actions needed
"""
    
    # Get first 100 lines of diff for context
    diff_preview = "\n".join(pr_diff.split("\n")[:100])
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    formatted_input = prompt.format(
        title=pr_title,
        summaries="\n\n".join(chunk_summaries),
        diff_preview=diff_preview
    )
    
    response = model.invoke(formatted_input)
    return response.content

# --- Core PR Analysis Functions ---

# async def verify_signature(request: Request):
#     """Verify the GitHub webhook signature."""
#     signature = request.headers.get("X-Hub-Signature-256")
#     if not WEBHOOK_SECRET:
#         raise HTTPException(status_code=500, detail="WEBHOOK_SECRET is not set.")
#     if not signature:
#         raise HTTPException(status_code=403, detail="No GitHub signature provided.")
    
#     payload_body = await request.body()
#     computed_signature = "sha256=" + hmac.new(
#         WEBHOOK_SECRET.encode(), payload_body, hashlib.sha256
#     ).hexdigest()
    
#     if not hmac.compare_digest(computed_signature, signature):
#         raise HTTPException(status_code=403, detail="Invalid signature")

# def fetch_pr_diff(diff_url):
#     """Fetch the diff for a PR from GitHub."""
#     headers = {"Authorization": f"token {GITHUB_TOKEN}"}
#     try:
#         diff_response = requests.get(diff_url, headers=headers)
#         if diff_response.status_code == 403 and "rate limit exceeded" in diff_response.text.lower():
#             raise HTTPException(status_code=429, detail="GitHub API rate limit exceeded.")
#         diff_response.raise_for_status()
#         return diff_response.text
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error fetching PR diff: {e}")
#         return ""

def analyze_pr(pr_title, pr_diff):
    """Main function to analyze PR with chunking and summarization."""
    # Step 1: Chunk the diff
    chunks = chunk_diff_by_blocks(pr_diff)
    logger.info(f"Created {len(chunks)} chunks from PR diff")
    
    # Step 2: Summarize each chunk (but don't store individual summaries)
    chunk_contents = []
    for chunk in chunks:
        chunk_contents.append(chunk)
    
    # Combine all chunks for final analysis
    combined_chunks = "\n\n".join(chunk_contents)
    
    # Step 3: Generate final comprehensive summary only
    final_summary = generate_final_summary(pr_title, [combined_chunks], pr_diff)
    
    # Simplified comment with only final review
    full_comment = f"""
## 🚀 PR Analysis Report: {pr_title}

### 📌 Comprehensive Review
{final_summary}

### 🔍 Diff Summary
The changes have been analyzed in their entirety, focusing on:
- Overall impact assessment
- Potential risks and issues
- Improvement recommendations
- Code quality evaluation
"""
    return full_comment

# @app.post("/analyze_chunks")
# async def analyze_pr_request(request: Request):
#     """Endpoint to analyze a GitHub PR."""
#     await verify_signature(request)
#     data = await request.json()
    
#     if "pull_request" not in data:
#         return {"message": "Not a PR event"}
    
#     pr = data.get("pull_request", {})
#     pr_title = pr.get("title", "No title provided.")
#     diff_url = pr.get("diff_url")
    
#     if not diff_url:
#         raise HTTPException(status_code=400, detail="Diff URL not found in PR data.")
    
#     pr_diff = fetch_pr_diff(diff_url)
#     analysis_comment = analyze_pr(pr_title, pr_diff)
    
#     comments_url = pr.get("comments_url")
#     if not comments_url:
#         raise HTTPException(status_code=400, detail="Comments URL not found in PR data.")
    
#     comment_payload = {"body": analysis_comment}
#     comment_response = requests.post(
#         comments_url,
#         headers={"Authorization": f"token {GITHUB_TOKEN}"},
#         json=comment_payload
#     )
    
#     if comment_response.status_code == 201:
#         return {"message": "Feedback posted successfully"}
#     else:
#         raise HTTPException(status_code=500, detail="Failed to post feedback")