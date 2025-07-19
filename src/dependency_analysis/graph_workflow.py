import os
import json
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from .ai_agents import dependency_analyzer_agent_executor
from .tools import store_dependencies

load_dotenv()
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    pr_data: dict
    structured_diff_text: str
    dependency_analysis: dict
    natural_language: str
    stored_result: str



def run_dependency_analysis(state: AgentState) -> dict:
    """Node to run Dependency Analysis agent."""
    logger.info("--- Dependency Analysis ---")
    try:
        result = dependency_analyzer_agent_executor.invoke({
            "structured_diff_text": state["structured_diff_text"],
            "repository_id": state["pr_data"]["repo_url"],
            "pr_commit_sha": state["pr_data"]["commit_sha"],
            "neo4j_uri": os.getenv("NEO4J_URI"),
            "neo4j_user": os.getenv("NEO4J_USERNAME"),
            "neo4j_pass": os.getenv("NEO4J_PASSWORD")
        })

        # print("\n=== RAW LLM OUTPUT ===")  # Debug line
        # print(result["output"])     
        
        # Parse the combined output
        output = result["output"]
        try:
            json_start = output.find('```json') + 7
            json_end = output.find('```', json_start)
            json_str = output[json_start:json_end].strip()
            
            nl_start = output.find('```', json_end + 3) + 3
            nl_end = output.find('```', nl_start)
            natural_language = output[nl_start:nl_end].strip() if nl_start > 3 else output
            
            return {
                "dependency_analysis": json.loads(json_str),
                "natural_language": natural_language
            }
        except Exception as e:
            logger.error(f"Output parsing error: {str(e)}")
            return {
                "dependency_analysis": {"dependencies": []},
                "natural_language": output
            }
    except Exception as e:
        error_msg = f"Dependency analysis error: {str(e)}"
        logger.error(error_msg)
        return {
            "dependency_analysis": {"dependencies": []},
            "natural_language": error_msg
        }

def store_dependency_graph(state: AgentState) -> dict:
    """Node to store dependencies in Neo4j."""
    logger.info("--- Storing Dependencies ---")
    try:
        # Properly format the input for the tool
        tool_input = {
            "repository_id": state["pr_data"]["repo_url"],
            "dependencies": state["dependency_analysis"]["dependencies"],
            "neo4j_uri": os.getenv("NEO4J_URI"),
            "neo4j_user": os.getenv("NEO4J_USERNAME"),
            "neo4j_pass": os.getenv("NEO4J_PASSWORD")
        }
        
        # Use invoke() with proper input structure
        result = store_dependencies.invoke(tool_input)
        return {"stored_result": result}
    except Exception as e:
        error_msg = f"Storage failed: {str(e)}"
        logger.error(error_msg)
        return {"stored_result": error_msg}

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("dependency_analyzer", run_dependency_analysis)
workflow.add_node("store_graph", store_dependency_graph)

workflow.set_entry_point("dependency_analyzer")
workflow.add_edge("dependency_analyzer", "store_graph")
workflow.add_edge("store_graph", END)

app_graph = workflow.compile()

def execute_analysis(pr_data: dict, structured_diff_text: str) -> dict:
    """Execute full analysis workflow."""
    logger.info(f"Starting analysis for PR #{pr_data.get('pull_request_number')}")
    try:
        final_state = app_graph.invoke({
            "pr_data": pr_data,
            "structured_diff_text": structured_diff_text
        })
        
        final_result = {
            "dependency_analysis_str": final_state.get("dependency_analysis"),
            "natural_language": final_state.get("natural_language"),
    
        }

        # Split "natural_language" at "Dependency Analysis:"
        natural_language = final_result.get("natural_language", "")
        if "Dependency Analysis:" in natural_language:
            before, after = natural_language.split("Dependency Analysis:", 1)
            final_result["natural_language"] = before.strip()
            final_result["dependency_analysis"] = after.strip()
        else:
            final_result["dependency_analysis"] = None  # or ""

        return {
            "dependency_analysis": final_result.get("dependency_analysis")
        }


    except Exception as e:
        error_msg = f"Workflow execution failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    