import os
from typing import List, Dict, Optional, TypedDict,Annotated
from pydantic import BaseModel, Field, field_validator, ConfigDict

import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, START,END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import re

import prompt_templates


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
        fileName: str = Field(..., description="The name of the file that has an issue")
        start_line_with_prefix: str = Field(..., description="The starting line number in the file (REQUIRED). \
                                            If the start_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        end_line_with_prefix: str = Field(..., description="The ending line number in the file (REQUIRED). \
                                          If the end_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        codeSegmentToFix: str = Field(..., description="The code segment that needs to be fixed from code diff")
        language: str = Field(..., description="The language of the code segment")
        issue: str = Field(..., description="The issue on the code segment")
        suggestion: str = Field(..., description="The suggestion to fix the code segment")
        suggestedCode: Optional[str] = Field(None, description="The updated code segment for the fix")
        severity: str = Field(..., description="The severity of the issue. Can be 'error', 'warning', or 'info'")

class FinalReviews(BaseModel):
    finalReviews: List[FinalReview] = Field(..., description="Final Reviews of Data of the Code.",)

structured_openai_llm = llm_openai.with_structured_output(FinalReviews)


# Schema for structured output to use in planning
class ReviewData(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)  # Add strict validation

        fileName: str = Field(..., description="The name of the file that has an issue")
        start_line_with_prefix: str = Field(..., description="The starting line number in the file (REQUIRED). \
                                            If the start_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        end_line_with_prefix: str = Field(..., description="The ending line number in the file (REQUIRED). \
                                          If the end_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        codeSegmentToFix: str = Field(..., description="The code segment that needs to be fixed from code diff")
        issue: str = Field(..., description="The issue on the code segment")

        @field_validator('start_line_with_prefix', 'end_line_with_prefix')
        def validate_line_prefix(cls, value):
            if not re.match(r'^[+-]\d+$', value):
                raise ValueError("Must start with '+' or '-' followed by digits")
            return value

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



#nodes
def error_handle(state: State):
     messages= prompt_templates.error_prompt.format_messages(PR_data=state["PR_data"])
     response = structured_llm.invoke(messages)
     
     return {"error_issues": response.reviewDatas}

def security_handle(state: State):
     response = structured_llm.invoke(
          prompt_templates.security_prompt.format_messages(PR_data=state["PR_data"])
     )
     return {"security_issues": response.reviewDatas}

def performance_handle(state: State):
     response = structured_llm.invoke(
          prompt_templates.performance_prompt.format_messages(PR_data=state["PR_data"])
     )
     return {"performance_issues": response.reviewDatas}

def quality_handle(state: State):
     response = structured_llm.invoke(
          prompt_templates.quality_prompt.format_messages(PR_data=state["PR_data"])
     )
     return {"quality_issues": response.reviewDatas}

def other_handle(state: State):
     response = structured_llm.invoke(
          prompt_templates.other_prompt.format_messages(PR_data=state["PR_data"])
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

        error_lines.append(f"File: {review.fileName}")
        error_lines.append(f"Lines: {review.start_line_with_prefix} to {review.end_line_with_prefix}")
        error_lines.append("Code Segment:")
        error_lines.append(review.codeSegmentToFix)
        error_lines.append(f"Issue: {review.issue}")
        error_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    error_str = '\n'.join(error_lines)

    for review in security_issues:

        security_lines.append(f"File: {review.fileName}")
        security_lines.append(f"Lines: {review.start_line_with_prefix} to {review.end_line_with_prefix}")
        security_lines.append("Code Segment:")
        security_lines.append(review.codeSegmentToFix)
        security_lines.append(f"Issue: {review.issue}")
        security_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    security_str = '\n'.join(security_lines)

    for review in performance_issues:

        performance_lines.append(f"File: {review.fileName}")
        performance_lines.append(f"Lines: {review.start_line_with_prefix} to {review.end_line_with_prefix}")
        performance_lines.append("Code Segment:")
        performance_lines.append(review.codeSegmentToFix)
        performance_lines.append(f"Issue: {review.issue}")
        performance_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    performance_str = '\n'.join(performance_lines)

    for review in quality_issues:

        quality_lines.append(f"File: {review.fileName}")
        quality_lines.append(f"Lines: {review.start_line_with_prefix} to {review.end_line_with_prefix}")
        quality_lines.append("Code Segment:")
        quality_lines.append(review.codeSegmentToFix)
        quality_lines.append(f"Issue: {review.issue}")
        quality_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    quality_str = '\n'.join(quality_lines)

    for review in other_issues:

        other_lines.append(f"File: {review.fileName}")
        other_lines.append(f"Lines: {review.start_line_with_prefix} to {review.end_line_with_prefix}")
        other_lines.append("Code Segment:")
        other_lines.append(review.codeSegmentToFix)
        other_lines.append(f"Issue: {review.issue}")
        other_lines.append("---")  # Separator

    # Join all lines into a single string with newlines
    other_str = '\n'.join(other_lines)
    final_str= error_str+"\n\n"+security_str+"\n\n"+performance_str+"\n\n"+quality_str+"\n\n"+other_str

    return {"all_issues": final_str}

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

