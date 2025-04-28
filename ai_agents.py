# agents.py
import os
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import retrieve_dependency_graphs, store_dependencies,get_file_content

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_retries=3,
    request_timeout=30
)


# Dependency Analysis Agent
def create_dependency_analyzer():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze code dependencies and return:
1. JSON for database storage 
2. Dependency analysis in natural language:

- ONLY report file-to-file dependencies between files inside the GitHub repository (for example: files like `A.py`, `B.js`, etc.).
- Do NOT report dependencies on external libraries, packages, or modules (like `fastapi`, `dotenv`, `requests`, etc.).
- STRICTLY follow this format if the target file exists in the diff:
  "Dependency Analysis:
  The [source] depends on [target]."
- STRICTLY follow this format if the target file does not exist in the diff, follow this format:
  "Dependency Analysis:
  The [source] depends on [target]. Content of [target]:
  [content of the target file]"
- Use `get_file_content` if needed, but only for internal repository files (not external libraries).
- NEVER report or describe external libraries or packages.
- Use the EXACT `commit_sha` from the PR data when fetching files.

IMPORTANT: Only show dependencies between source code files in the repository. Ignore anything else. Follow the format strictly without adding any other information.

Note that jason format is an example, you should not return it in the output.
JSON Format:
```json
{{
  "dependencies": [
    {{"source": "fileA.js", "target": "fileB.js", "type": "imports", "source_type": "JS", "target_type": "JS"}}
  ]
}}
```"""),
        ("system", """Use THIS exact commit_sha from the PR data: {pr_commit_sha}"""),
        ("human", "Repository: {repository_id}\nCode Changes: {structured_diff_text}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(
        llm=llm,
        tools=[retrieve_dependency_graphs, store_dependencies,get_file_content],
        prompt=prompt
    )
    return AgentExecutor(
        agent=agent,
        tools=[retrieve_dependency_graphs, store_dependencies,get_file_content],
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=7
    )
# Create the executor instances
dependency_analyzer_agent_executor = create_dependency_analyzer()

