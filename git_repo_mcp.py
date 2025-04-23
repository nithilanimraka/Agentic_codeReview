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

# --- Core Agent Logic (Refactored for Streaming) ---

async def stream_git_repo_query(directory_path: str, query: str):
    """
    Connects to the MCP server, runs the agent query, and yields token deltas.
    """
    try:
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
                instructions=f"Answer questions about the local git repository at {directory_path}. Use the provided 'repo_path' tool parameter for Git operations.",
                mcp_servers=[server],
            )

            # Start the run in streaming mode
            # Use trace for observability if needed
            with trace(workflow_name="MCP Git Query Stream"):
                result = Runner.run_streamed(starting_agent=agent, input=query)

                # Stream the raw token deltas as soon as they arrive
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        # Yield each chunk of text as it comes in
                        yield event.data.delta
                    # You could potentially yield other event types if needed by the frontend
                    # elif event.type == "tool_start":
                    #     yield f"[DEBUG: Tool Start - {event.data.tool_name}]\n"
                    # elif event.type == "tool_end":
                    #     yield f"[DEBUG: Tool End - {event.data.tool_name}]\n"

                # Ensure the final output (if any) is processed if needed,
                # though streaming deltas usually covers it.
                # final_output = await result.final_output # Await if needed

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





