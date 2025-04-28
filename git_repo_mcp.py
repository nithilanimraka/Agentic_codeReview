import os
from fastapi import FastAPI
from pydantic import BaseModel

from agents import Agent, Runner, trace
from agents.mcp import MCPServer, MCPServerStdio
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv

# --- Configuration & Setup ---

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")
    raise ValueError("OPENAI_API_KEY is required.")

# --- Pydantic Model for Request Body ---

class QueryRequest(BaseModel):
    """Defines the expected structure of the incoming request."""
    repo_path: str
    query: str

# General instructions for the agent, applicable to all queries
SYSTEM_INSTRUCTIONS = """
You are an expert Git repository assistant, specifically designed to analyze and answer questions about the local Git repository located at {directory_path}. Your primary function is to provide insights based *only* on the data within this repository using the available tools.

**Core Directives:**

1.  **Strict Focus:** Only answer questions directly related to the Git repository at {directory_path}. This includes its history, commits, branches, contributors, file contents, code structure, potential quality signals, and related metrics derived from the repository's data.
2.  **Tool Utilization:** You may use the provided tools to gather information from the Git repository if it is required only. Always use the 'repo_path' tool parameter, setting it explicitly to {directory_path} for every Git operation.
3.  **Off-Topic Refusal:** If the user asks a question unrelated to the specified Git repository (e.g., general knowledge, unrelated programming questions, personal opinions), politely decline to answer. State clearly that your function is limited to analyzing the current Git repository. A suitable response would be: "I can only answer questions directly related to the Git repository at {directory_path}."
4.  **Complete Answers:** Provide comprehensive answers based on the information retrieved from the tools. Synthesize the data effectively.
5.  **No Follow-up Questions:** Conclude your responses definitively. **Do not end your answers with questions** asking the user if they want more details or further action. Present the information clearly and await the user's next distinct query.

**Example Interaction:**

* **User:** "Who is the main contributor?"
* **You:** (Use tools to find the top contributor) "The main contributor to the repository at {directory_path} is [Contributor Name] with [X] commits." (Do NOT ask: "Would you like to see their recent commits?")

* **User:** "What's the capital of France?"
* **You:** "I can only answer questions directly related to the Git repository at {directory_path}."

If you think you can refine your answer by enhancing your thought process (eg: running commands to get more accurate answers), do it without hesitation or prompting the user for it.
Adhere strictly to these instructions. Your goal is to be an accurate, focused, and tool-driven Git repository analysis assistant for the path {directory_path}.

"""

# Mapping of preset questions (exactly as sent from VS Code) to specific LLM prompts
# The {directory_path} placeholder will be filled in dynamically.
PRESET_PROMPTS = {
    "Who's the most frequent contributor?":
        "Identify the user who has made the most commits in the repository located at {directory_path}. Provide their name or username and the commit count.",
    "Summarize the last change in the repository.":
        "Describe the most recent commit made to the repository at {directory_path}. Include the author, commit message, and date if possible.",
    "What are the Contributor insights?":
        "Provide insights about the contributors to the repository at {directory_path}. This could include contribution frequency, percentage of contributions (eg:40%), types of changes, or collaboration patterns. Focus on the top 3-5 contributors if possible.",
    "Provide the Commit-Message Quality based on days of the week":
        "Analyze the commit messages in the repository at {directory_path}. Group commits by the day of the week they were made (Monday, Tuesday, etc.) and assess the general quality (e.g., descriptiveness, length, clarity) of messages for each day. Provide a summary.",
    "Give me the Recent activity / cadence":
        "Describe the recent commit activity and cadence for the repository at {directory_path}. How frequent are commits? Are there specific days or times with more activity? Look at the last month or a reasonable recent period.",
    "Provide me the Risk & quality signals of the repository":
        "Analyze the repository at {directory_path} for potential risk and quality signals based on its Git history. Consider factors like commit frequency, commit message quality, contributor distribution (bus factor), recent bug fix frequency, or large complex commits. Provide a summary of potential signals.",
    "Give the Ownership & bus-factor of files":
        "Analyze the file ownership and estimate the bus factor for key files or directories in the repository at {directory_path}. Identify files primarily maintained by a single contributor."
}

# --- Core Agent Logic  ---

async def stream_git_repo_query(directory_path: str, user_query: str):
    """
    Connects to the MCP server, runs the agent query, and yields token deltas.
    """
    try:

        # Determine the actual prompt to send to the LLM
        llm_input_prompt = PRESET_PROMPTS.get(user_query) # Check if it's a preset question
        if llm_input_prompt:
            llm_input_prompt = llm_input_prompt.format(directory_path=directory_path)
            print(f"Using preset prompt for query: '{user_query}'")
        else:
            # It's a custom query, use it directly
            llm_input_prompt = user_query
            print(f"Using custom query: '{user_query}'")

        # Format the system instructions with the specific directory path
        formatted_system_instructions = SYSTEM_INSTRUCTIONS.format(directory_path=directory_path)

        # Use async context manager for MCPServerStdio
        async with MCPServerStdio(
            cache_tools_list=True,
            params={
                "command": "python",
                "args": [
                    "-m",
                    "mcp_server_git",  # Ensure mcp_server_git is installed and accessible
                    "--repository",
                    directory_path
                ]
            },
        ) as server:
            # Define the agent within the context of the server
            agent = Agent(
                name="Assistant",
                model="gpt-4.1-mini", 
                instructions=formatted_system_instructions, # Use the formatted general instructions,
                mcp_servers=[server],
            )
            
            result = Runner.run_streamed(starting_agent=agent, input=llm_input_prompt, max_turns=20)

            # Stream the raw token deltas as soon as they arrive
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    # Yield each chunk of text as it comes in
                    yield event.data.delta

            print("history list:", result.to_input_list()) 

            if result.trace:                               # will be None if tracing disabled
                trace_url = f"https://platform.openai.com/traces/trace?trace_id={result.trace.trace_id}"
                print(f"🔍  Full trace: {trace_url}")
            else:
                print("⚠️  Tracing disabled or export failed.")
                

    except FileNotFoundError:
        # Specific error if the mcp_server_git command isn't found
        error_msg = f"Error: 'mcp_server_git' command not found. Make sure the agents-mcp-git package is installed and accessible in your Python environment's PATH.\n"
        print(error_msg)
        yield error_msg # Yield error message to the stream
    except Exception as e:
        # Catch other potential errors during MCP server startup or agent run
        error_msg = f"Error processing git query: {e}\n"
        print(error_msg) # Log the error server-side
        yield error_msg # Yield error message to the stream


# --- FastAPI Application ---

# Define lifespan manager if you need setup/teardown (optional here)
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Code to run on startup
#     print("Analyzer server starting up...")
#     yield
#     # Code to run on shutdown
#     print("Analyzer server shutting down...")
# app = FastAPI(lifespan=lifespan)





