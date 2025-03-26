import os
from typing import List, Dict, Optional, TypedDict,Annotated
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, START,END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


load_dotenv()

groq_api_key = os.environ.get('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set")

openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

llm_groq = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="deepseek-r1-distill-llama-70b"
)

llm_openai = ChatOpenAI(model="gpt-4o-2024-08-06",
                        api_key=openai_api_key,
                         temperature=0)


# Schema for structured output to use in planning
class FinalReview(BaseModel):
        fileName: str = Field(description="The name of the file that has an issue")
        start_line_with_prefix: str = Field(description="The starting line number in the file (REQUIRED). \
                                            If the start_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        end_line_with_prefix: str = Field(description="The ending line number in the file (REQUIRED). \
                                          If the end_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        codeSegmentToFix: str = Field(description="The code segment that needs to be fixed from code diff")
        language: str = Field(description="The language of the code segment")
        issue: str = Field(description="The issue on the code segment")
        suggestion: str = Field(description="The suggestion to fix the code segment")
        suggestedCode: Optional[str] = Field(None, description="The updated code segment for the fix")
        severity: str = Field(description="The severity of the issue. Can be 'error', 'warning', or 'info'")

class FinalReviews(BaseModel):
    finalReviews: List[FinalReview] = Field(description="Final Reviews of Data of the Code.",)

structured_openai_llm = llm_openai.with_structured_output(FinalReviews)


# Schema for structured output to use in planning
class ReviewData(BaseModel):
        fileName: str = Field(description="The name of the file that has an issue", required=True)
        start_line_with_prefix: str = Field(description="The starting line number in the file (REQUIRED). \
                                            If the start_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix", required=True)
        end_line_with_prefix: str = Field(description="The ending line number in the file (REQUIRED). \
                                          If the end_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix", required=True)
        codeSegmentToFix: str = Field(description="The code segment that needs to be fixed from code diff", required=True)
        issue: str = Field(description="The issue on the code segment", required=True)

class ReviewDatas(BaseModel):
    reviewDatas: List[ReviewData] = Field(description="Reviews of Data of the Code.",)

structured_llm = llm_groq.with_structured_output(ReviewDatas)

def line_numbers_handle(start_line_with_prefix, end_line_with_prefix):
    value1 = start_line_with_prefix
    print("before remove prefix start line:", value1)
    if(start_line_with_prefix[0]=='-'): 
        start_line = int(value1.replace("-", "").strip())  # Remove '+' and strip spaces
    else:
        start_line = int(value1.replace("+", "").strip())
    print("after removing prefix start line:", start_line)

    value2 = end_line_with_prefix
    print("before remove prefix end line:", value2)
    if(end_line_with_prefix[0]=='-'):
        end_line = int(value2.replace("-", "").strip())
    else:
        end_line = int(value2.replace("+", "").strip()) 
    print("after removing prefix end line:", end_line)
    print()

    return start_line, end_line
     

# class State(TypedDict):
#     PR_data: str
#     #PR_title: str
#     error_issues: Annotated[List[Dict], "List of error handling issues"]
#     security_issues: Annotated[List[Dict], "List of security issues"]
#     performance_issues: Annotated[List[Dict], "List of performance issues"]
#     quality_issues: Annotated[List[Dict], "List of quality issues"]
#     other_issues: Annotated[List[Dict], "List of other issues"]

class State(TypedDict):
    PR_data: str
    #PR_title: str
    error_issues: list[ReviewData]
    security_issues: list[ReviewData]
    performance_issues: list[ReviewData]
    quality_issues: list[ReviewData]
    other_issues: list[ReviewData]
    all_issues: str

error_prompt = ChatPromptTemplate.from_messages([
     ("system","""
        You are an expert code reviewer **strictly focused on error handling and logging**.  
        **Focus ONLY on**:  
        - Missing error handling (e.g., uncaught exceptions, no try/catch blocks).  
        - Logs lacking severity levels, timestamps, or context (e.g., user ID, request ID).  
        - Generic error messages (e.g., "Error occurred" instead of "Failed to connect to DB: timeout after 5s").  
        - Silent failures (e.g., swallowing exceptions without logging).  

        **Ignore and DO NOT mention**:  
        - Security risks (e.g., SQL injection, XSS) → handled by the security agent.  
        - Slow code or inefficient retries → handled by the performance agent.  
        - Code formatting/naming → handled by the quality agent.  

        **Examples of issues to report**:  
        - `except: pass` (no logging).  
        - `logger.error("Error")` (no stack trace or details).  
      
        Examples for start_line_with_prefix when the start_line is from new file: "+5, +2, +51, +61" 
        Examples for start_line_with_prefix when the start_line is from old file: "-8, -1, -56, -20" 

        Examples for end_line_with_prefix when the start_line is from new file: "+10, +2, +77, +65" 
        Examples for end_line_with_prefix when the start_line is from old file: "-1, -5, -22, -44" 

        VERY IMPORTANT:  
        - Always give ONLY the codeSegment that needs to be fixed for codeSegmentToFix. Don't provide a lot of trailing and preceding code.
        - Don’t create issues on your own for the sake of providing outputs. If there are none, don’t produce outputs.
                    
    """), 
    ("human", "Analyze the following code diff and provide issues if exist:\n{PR_data}"),
])

security_prompt = ChatPromptTemplate.from_messages([
     ("system", """
        You are an expert code reviewer **strictly focused on security vulnerabilities**.  
        **Focus ONLY on**:  
        - OWASP Top 10 risks (e.g., SQLi, XSS, broken auth).  
        - Hardcoded secrets (e.g., API keys, passwords in code).  
        - Missing encryption (e.g., plaintext passwords).  
        - Insecure dependencies (e.g., outdated libraries with CVEs).  

        **Ignore and DO NOT mention**:  
        - General error handling (e.g., missing try/catch) → handled by the error agent.  
        - Slow code → handled by the performance agent.  
        - Code style → handled by the quality agent.  

        **Examples of issues to report**:  
        - `query = f"SELECT * FROM users WHERE id = [user_input]"` (SQLi risk).  
        - `password = "secret123"` (hardcoded credential).  
      
        Examples for start_line_with_prefix when the start_line is from new file: "+5, +2, +51, +61" 
        Examples for start_line_with_prefix when the start_line is from old file: "-8, -1, -56, -20" 

        Examples for end_line_with_prefix when the start_line is from new file: "+10, +2, +77, +65" 
        Examples for end_line_with_prefix when the start_line is from old file: "-1, -5, -22, -44" 

        VERY IMPORTANT:  
        - Always give ONLY the codeSegment that needs to be fixed for codeSegmentToFix. Don't provide a lot of trailing and preceding code.
        - Don’t create issues on your own for the sake of providing outputs. If there are none, don’t produce outputs.
      """),
    ("human", "Analyze the following code diff and provide issues if exist:\n{PR_data}"),
])

performance_prompt = ChatPromptTemplate.from_messages([
     ("system", """
        You are an expert code reviewer **strictly focused on performance**.  
        **Focus ONLY on**:  
        - Inefficient algorithms (e.g., O(n²) loops on large datasets).  
        - Memory leaks (e.g., unclosed resources, caching without TTL).  
        - N+1 database queries or missing indexes.  
        - Redundant computations (e.g., recalculating the same value in a loop).  

        **Ignore and DO NOT mention**:  
        - Missing error handling → handled by the error agent.  
        - Security flaws → handled by the security agent.  
        - Variable naming → handled by the quality agent.  

        **Examples of issues to report**:  
        - `for user in users: db.query(UserDetails).filter(id=user.id)` (N+1 query).  
        - `data = [x**2 for x in range(10_000)]` (precompute if reused).  
      
        Examples for start_line_with_prefix when the start_line is from new file: "+5, +2, +51, +61" 
        Examples for start_line_with_prefix when the start_line is from old file: "-8, -1, -56, -20" 

        Examples for end_line_with_prefix when the start_line is from new file: "+10, +2, +77, +65" 
        Examples for end_line_with_prefix when the start_line is from old file: "-1, -5, -22, -44" 

        VERY IMPORTANT:  
        - Always give ONLY the codeSegment that needs to be fixed for codeSegmentToFix. Don't provide a lot of trailing and preceding code.
        - Don’t create issues on your own for the sake of providing outputs. If there are none, don’t produce outputs.
    """),
    ("human", "Analyze the following code diff and provide issues if exist:\n{PR_data}"),
])

quality_prompt = ChatPromptTemplate.from_messages([
     ("system", """
        You are an expert code reviewer **strictly focused on code quality**.  
        **Focus ONLY on**:  
        - Unclear naming (e.g., `def process()` vs. `def calculate_invoice()`).  
        - Configuration issues (e.g., incorrect env variables).
        - Licensing, deployment scripts, or undocumented business logic.
        - Violations of PEP8/SOLID (e.g., 1000-line functions, no interfaces).  

        **Ignore and DO NOT mention**:  
        - Error handling → handled by the error agent.  
        - Security/performance → handled by their agents.  

        **Examples of issues to report**:  
        - `a = 10` (non-descriptive variable name).  
      
        Examples for start_line_with_prefix when the start_line is from new file: "+5, +2, +51, +61" 
        Examples for start_line_with_prefix when the start_line is from old file: "-8, -1, -56, -20" 

        Examples for end_line_with_prefix when the start_line is from new file: "+10, +2, +77, +65" 
        Examples for end_line_with_prefix when the start_line is from old file: "-1, -5, -22, -44" 

        VERY IMPORTANT:  
        - Always give ONLY the codeSegment that needs to be fixed for codeSegmentToFix. Don't provide a lot of trailing and preceding code.
        - Don’t create issues on your own for the sake of providing outputs. If there are none, don’t produce outputs. 
    """),
    ("human", "Analyze the following code diff and provide issues if exist:\n{PR_data}"),
])

other_prompt = ChatPromptTemplate.from_messages([
     ("system", """
            You are an expert code reviewer specializing generally. 
            - **Ignore**:
                - Errors, security, performance, code quality (handled by other agents).
            - **Focus ONLY on**:
                - Configuration issues (e.g., incorrect env variables).
                - Licensing, deployment scripts, or undocumented business logic.
            KEY POINTS:
            **The following issues are already handled. If there is any issue related to the following, IGNORE them otherwise you will be repeating it. So its
            very important that you IGNORE issues related to the below mentioned points.**
            - Error, exception handling issues, logging issues.
            - Security vulnerabilities related issues.
            - Performance related issues
            - code quality, readability, maintainability, code standards & consistency related issues.
      
            Examples for start_line_with_prefix when the start_line is from new file: "+5, +2, +51, +61" 
            Examples for start_line_with_prefix when the start_line is from old file: "-8, -1, -56, -20" 

            Examples for end_line_with_prefix when the start_line is from new file: "+10, +2, +77, +65" 
            Examples for end_line_with_prefix when the start_line is from old file: "-1, -5, -22, -44" 

            VERY IMPORTANT:
            - Always give ONLY the codeSegment that needs to be fixed for codeSegmentToFix. Don't provide a lot of trailing and preceding code.
            - Don’t create issues on your own for the sake of providing outputs. If there are none, don’t produce outputs. """),
("human", "Analyze the following code diff and provide issues if exist:\n{PR_data}"),
])

#nodes
def error_handle(state: State):
     messages= error_prompt.format_messages(PR_data=state["PR_data"])
     response = structured_llm.invoke(messages)
     
     return {"error_issues": response.reviewDatas}

def security_handle(state: State):
     response = structured_llm.invoke(
          security_prompt.format_messages(PR_data=state["PR_data"])
     )
     return {"security_issues": response.reviewDatas}

def performance_handle(state: State):
     response = structured_llm.invoke(
          performance_prompt.format_messages(PR_data=state["PR_data"])
     )
     return {"performance_issues": response.reviewDatas}

def quality_handle(state: State):
     response = structured_llm.invoke(
          quality_prompt.format_messages(PR_data=state["PR_data"])
     )
     return {"quality_issues": response.reviewDatas}

def other_handle(state: State):
     response = structured_llm.invoke(
          other_prompt.format_messages(PR_data=state["PR_data"])
     )
     return {"other_issues": response.reviewDatas}

# Proper aggregator implementation
def aggregator(state: State):
    error_lines=["Error and Exception handling issues:\n"]
    security_lines=["Security vulnerability issues:\n"]
    performance_lines=["Performance issues:\n"]
    quality_lines=["Quality issues:\n"]
    other_lines=["Other isses:\n"]

    error_issues = state["error_issues"]
    security_issues = state["security_issues"]
    performance_issues = state["performance_issues"]
    quality_issues = state["quality_issues"]
    other_issues = state["other_issues"]

    for review in error_issues:

        start_line, end_line = line_numbers_handle(review.start_line_with_prefix, review.end_line_with_prefix)

        error_lines.append(f"File: {review.fileName}")
        error_lines.append(f"Lines: {start_line} to {end_line}")
        error_lines.append("Code Segment:")
        error_lines.append(review.codeSegmentToFix)
        error_lines.append(f"Issue: {review.issue}")
        error_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    error_str = '\n'.join(error_lines)

    for review in security_issues:

        start_line, end_line = line_numbers_handle(review.start_line_with_prefix, review.end_line_with_prefix)

        security_lines.append(f"File: {review.fileName}")
        security_lines.append(f"Lines: {start_line} to {end_line}")
        security_lines.append("Code Segment:")
        security_lines.append(review.codeSegmentToFix)
        security_lines.append(f"Issue: {review.issue}")
        security_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    security_str = '\n'.join(security_lines)

    for review in performance_issues:

        start_line, end_line = line_numbers_handle(review.start_line_with_prefix, review.end_line_with_prefix)

        performance_lines.append(f"File: {review.fileName}")
        performance_lines.append(f"Lines: {start_line} to {end_line}")
        performance_lines.append("Code Segment:")
        performance_lines.append(review.codeSegmentToFix)
        performance_lines.append(f"Issue: {review.issue}")
        performance_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    performance_str = '\n'.join(performance_lines)

    for review in quality_issues:

        start_line, end_line = line_numbers_handle(review.start_line_with_prefix, review.end_line_with_prefix)

        quality_lines.append(f"File: {review.fileName}")
        quality_lines.append(f"Lines: {start_line} to {end_line}")
        quality_lines.append("Code Segment:")
        quality_lines.append(review.codeSegmentToFix)
        quality_lines.append(f"Issue: {review.issue}")
        quality_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    quality_str = '\n'.join(quality_lines)

    for review in other_issues:

        start_line, end_line = line_numbers_handle(review.start_line_with_prefix, review.end_line_with_prefix)

        other_lines.append(f"File: {review.fileName}")
        other_lines.append(f"Lines: {start_line} to {end_line}")
        other_lines.append("Code Segment:")
        other_lines.append(review.codeSegmentToFix)
        other_lines.append(f"Issue: {review.issue}")
        other_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    other_str = '\n'.join(other_lines)
    final_str= error_str+"\n\n"+security_str+"\n\n"+performance_str+"\n\n"+quality_str+"\n\n"+other_str

    return {"all_issues": final_str}
    




# pr_input="""
# diff --git a/src/main/java/com/example/utils/NumberUtils.java b/src/main/java/com/example/utils/NumberUtils.java
# index abc1234..def5678 100644
# --- a/src/main/java/com/example/utils/NumberUtils.java
# +++ b/src/main/java/com/example/utils/NumberUtils.java
# @@ -1,8 +1,8 @@
#  public class NumberUtils {
 
#      public static int divide(int a, int b) {
# +        return a / b; 
#      }
 
#      public static int[] createArray(int size) {
#          int[] arr = new int[size];
# +        for (int i = 0; i <= size; i++) { 
#              arr[i] = i * 2;
#          }
#          return arr;
#      }
 
#      public static String getStringValue(Map<String, String> map, String key) {
# +        return map.get(key).toUpperCase(); 
#      }
#  }

# """

def invoke(structured_diff_text: str):

    # Build workflow
    parallel_builder = StateGraph(State)

    # Add nodes
    parallel_builder.add_node("error_handle", error_handle)
    parallel_builder.add_node("security_handle", security_handle)
    parallel_builder.add_node("performance_handle", performance_handle)
    parallel_builder.add_node("quality_handle", quality_handle)
    parallel_builder.add_node("other_handle", other_handle)
    parallel_builder.add_node("aggregator", aggregator)

    # Add edges to connect nodes
    parallel_builder.add_edge(START, "error_handle")
    parallel_builder.add_edge(START, "security_handle")
    parallel_builder.add_edge(START, "performance_handle")
    parallel_builder.add_edge(START, "quality_handle")
    parallel_builder.add_edge(START, "other_handle")
    parallel_builder.add_edge("error_handle", "aggregator")
    parallel_builder.add_edge("security_handle", "aggregator")
    parallel_builder.add_edge("performance_handle", "aggregator")
    parallel_builder.add_edge("quality_handle", "aggregator")
    parallel_builder.add_edge("other_handle", "aggregator")
    parallel_builder.add_edge("aggregator", END)
    parallel_workflow = parallel_builder.compile()

    # Show workflow
    display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

    state = parallel_workflow.invoke({"PR_data": structured_diff_text})
    final_issues=state["all_issues"]

    return final_issues


#print(final_issues)


final_prompt= ChatPromptTemplate.from_messages([
     ("system", """
        You are an expert code reviewer. Based on the given Pull Request(PR) Data and the issues related to it, Analyze them and generate detailed review comments.
      
        Important:
         - Examples for start_line_with_prefix when the start_line is from new file: "+5, +2, +51, +61" 
         - Examples for start_line_with_prefix when the start_line is from old file: "-8, -1, -56, -20" 

         - Examples for end_line_with_prefix when the start_line is from new file: "+10, +2, +77, +65" 
         - Examples for end_line_with_prefix when the start_line is from old file: "-1, -5, -22, -44" 
    """),
    ("human", "PR data:\n{PR_data} \n\n Issues related: {Issues}"),
])


def final_review(pr_data:str) -> List[Dict]:
     
    final_issues=invoke(pr_data)
    print(final_issues)

    final_response= structured_openai_llm.invoke(final_prompt.format_messages(PR_data=pr_data, Issues=final_issues))

    return [review.model_dump() for review in final_response.finalReviews]

