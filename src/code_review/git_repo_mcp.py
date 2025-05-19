import os
from collections import deque
from typing import Dict, List, Deque, Tuple
import contextvars
from pydantic import BaseModel

from agents import Agent, Runner, trace, function_tool
from agents.mcp import MCPServer, MCPServerStdio
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv


# Global context variable that will hold the id per request
current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_session_id")

# Load environment variables 
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")
    raise ValueError("OPENAI_API_KEY is required.")

# Store history as deque of tuples: (role, content)
# Using deque for efficient appends and limiting size
MAX_HISTORY_PAIRS = 4 # Store last 4 user/assistant pairs (8 messages total)
session_histories: Dict[str, Deque[Tuple[str, str]]] = {}

class QueryRequest(BaseModel):
    """Defines the expected structure of the incoming request."""
    repo_path: str
    query: str
    session_id: str

class EndSessionRequest(BaseModel):
    """Defines the structure for ending a session."""
    session_id: str

SYSTEM_INSTRUCTIONS = """
You are an expert Git repository assistant for the repository at {directory_path}. Your primary function is to provide insights based *only* on the data within this repository using the available tools.

**Core Directives**

1.  **Strict Focus**: Answer ONLY questions directly related to the Git repository at **{directory_path}**. Refuse off-topic requests politely: *“I can only answer questions directly related to the Git repository at {directory_path}.”*
2.  **Tool Utilization**: Use the provided Git-analysis tools whenever information must be fetched from the repository. Always pass `repo_path="{directory_path}"` for Git tools.
3.  **COMPLETE ANSWERS**: Synthesize all retrieved data into clear, comprehensive responses. Do **not** end replies with follow-up questions.

**MANDATORY Conversation Memory Handling**

* A function tool named **`get_recent_turns`** provides the last 4 user/assistant message pairs for context. It takes **no arguments**.
* **Invoke `get_recent_turns()` FIRST** (before calling any other tools or generating a final answer) if **ANY** of the following conditions are met regarding the user's latest message:
    * The message contains pronouns referring to previous turns (e.g., "he", "she", "it", "they", "that", "those", "them").
    * The message uses relative references (e.g., "the previous commit", "the file mentioned above", "what about the first point?", "continue with that").
    * The message explicitly asks to use memory (e.g., "use history", "see above", "remember what I said").
    * The request *clearly depends* on information exchanged in prior turns to be understood or answered correctly.
    * You have **ANY doubt** about the context or what the user is referring to.
* **How to Use**: Emit a single function-call: `name="get_recent_turns", arguments=""`.
* **After Getting History**: Use the provided history (`tool_result`) to understand the context and formulate your response. If the history *still* doesn't clarify the user's request, ask the user for clarification.
* **FAILURE TO USE `get_recent_turns` WHEN NEEDED WILL RESULT IN INACCURATE OR NONSENSICAL ANSWERS. PRIORITIZE USING IT WHEN IN DOUBT.**

**Thought-Process Freedom**
* If additional Git tool calls or reasoning steps (after potentially using `get_recent_turns`) will improve accuracy, perform them.

**Example Flow (Implicit Reference):**

User: "Who made the last commit?"
Assistant (calls Git log tool): "The last commit (abc1234) was made by Alice."
User: "What files did *it* modify?"
Assistant (Recognizes "it" refers to the previous commit):
    1.  **Calls `get_recent_turns()`** -> Receives history including the previous exchange.
    2.  Calls Git diff tool for commit abc1234.
    3.  Responds: "Commit abc1234 modified the following files: [list of files]."

**Example Flow (Explicit Reference):**

User: "Tell me about the main branch."
Assistant (Calls Git tools): "The main branch has [details]..."
User: "**See above**, what's the latest tag on that branch?"
Assistant (Recognizes "See above" and "that branch"):
    1.  **Calls `get_recent_turns()`** -> Receives history.
    2.  Calls Git tag tool for the main branch.
    3.  Responds: "The latest tag on the main branch is v1.2.3."

Adhere strictly to these instructions for the repository at {directory_path}.
"""

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

async def stream_git_repo_query(session_id: str, directory_path: str, user_query: str):
    """
    Connects to the MCP server, runs the agent query with history(optional),
    updates history, and yields token deltas.
    """
    # Get history for this session, or initialize if new
    # deque for efficient append/popleft
    history = session_histories.setdefault(session_id, deque(maxlen=MAX_HISTORY_PAIRS * 2))
    print(f"Session {session_id}: History length = {len(history)}")

    # Convert deque history to the list format expected by Runner (if needed)
    # Assuming Runner expects [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    # history_list = [{"role": role, "content": content} for role, content in history]

    assistant_response_content = "" # Accumulate the assistant's full response

    try:
        # Determine the actual prompt to send to the LLM
        llm_input_prompt = PRESET_PROMPTS.get(user_query)
        if llm_input_prompt:
            llm_input_prompt = llm_input_prompt.format(directory_path=directory_path)
            print(f"Session {session_id}: Using preset prompt for query: '{user_query}'")
        else:
            llm_input_prompt = user_query # Use custom query directly
            print(f"Session {session_id}: Using custom query: '{user_query}'")

        # Format the system instructions with the specific directory path
        formatted_system_instructions = SYSTEM_INSTRUCTIONS.format(directory_path=directory_path)

        # Async context manager for MCPServerStdio
        async with MCPServerStdio(
            cache_tools_list=True,
            params={
                "command": "python",
                "args": ["-m", "mcp_server_git", "--repository", directory_path]
            },
        ) as server:
            agent = Agent(
                name="Assistant",
                model="gpt-4.1-mini",
                instructions=formatted_system_instructions,
                mcp_servers=[server],
                tools=[get_recent_turns],
            )

            result = Runner.run_streamed(starting_agent=agent, input=llm_input_prompt, max_turns=20)

            # Stream the raw token deltas and accumulate the response
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta
                    assistant_response_content += delta # Accumulate
                    yield delta # Yield chunk immediately

            # Update History AFTER successful run 
            # Add the user query and the full assistant response to the history deque
            history.append(("user", user_query))
            history.append(("assistant", assistant_response_content))
            session_histories[session_id] = history # Store updated deque back (though modify in place)
            print(f"Session {session_id}: Updated history. New length: {len(history)}")
            print(session_histories[session_id])
            # Deque automatically handles the maxlen limit

            if result.trace:
                trace_url = f"https://platform.openai.com/traces/trace?trace_id={result.trace.trace_id}"
                print(f"🔍 Full trace for session {session_id}: {trace_url}")
            else:
                print(f"⚠️ Tracing disabled or export failed for session {session_id}.")


    except FileNotFoundError:
        error_msg = f"Error: 'mcp_server_git' command not found. Make sure the agents-mcp-git package is installed and accessible in your Python environment's PATH.\n"
        print(error_msg)
        yield error_msg
    except Exception as e:
        error_msg = f"Error processing git query for session {session_id}: {e}\n"
        print(error_msg)
        yield error_msg


def _memory_fn(session_id: str):
    k=MAX_HISTORY_PAIRS
    history_deque: Deque[Tuple[str, str]] = session_histories.get(session_id, deque())
    # Convert the right-most 2k entries (user/assistant alternating) to dicts
    return [{"role": role, "content": content} for role, content in list(history_deque)[-2*k:]]

@function_tool
def get_recent_turns() -> List[Dict[str, str]]:
    """
    Return up to the last 4 user/assistant pairs for this session
    as chat-completion-style dicts.
    """
    session_id = current_session_id.get(None)  # default None
    if session_id is None:
        raise RuntimeError("current_session_id not set – did you forget to call .set() in the route?")
    
    print("🔧 get_recent_turns CALLED for session_id=%s", session_id)
    return _memory_fn(session_id)
    
