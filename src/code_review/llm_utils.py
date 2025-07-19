import os
from typing import List, Dict, Optional, TypedDict,Annotated
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
import logging 
import json 

import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, START,END
from langchain_core.prompts import ChatPromptTemplate
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from src.code_review import prompt_templates


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# groq_api_key = os.environ.get('GROQ_API_KEY')
# if not groq_api_key:
#     raise ValueError("GROQ_API_KEY is not set")

google1_api_key = os.environ.get('GEMINI_API_REVIEW_KEY')
google1_api_key = os.environ.get('GEMINI_API_REVIEW_KEY')
if not google1_api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

gemini_api_key = os.environ.get('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set")


gemini_api_key2 = os.environ.get('GOOGLE_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set")

nithila_api_key = os.environ.get('NITHILA_GOOGLE_API_KEY')
if not nithila_api_key:
    raise ValueError("NITHILA_API_KEY is not set")

randinu_api_key = os.environ.get('RANDINU_GOOGLE_API_KEY')
if not randinu_api_key:
    raise ValueError("RANDINU_API_KEY is not set")


gemini_api_key2 = os.environ.get('GOOGLE_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set")

nithila_api_key = os.environ.get('NITHILA_GOOGLE_API_KEY')
if not nithila_api_key:
    raise ValueError("NITHILA_API_KEY is not set")

randinu_api_key = os.environ.get('RANDINU_GOOGLE_API_KEY')
if not randinu_api_key:
    raise ValueError("RANDINU_API_KEY is not set")

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
llm_gemini2 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_api_key2,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_nithila = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=nithila_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_randinu = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=randinu_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_fin_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_api_key2,
    temperature=0,
    max_tokens=None,
    timeout=60,
    max_retries=2,
    )


# Schema for structured output to use in planning
class FinalReview(BaseModel):
        fileName: str = Field(..., description="The name of the file that has an issue")
        codeSegmentToFix: str = Field(..., description="The code segment that needs to be fixed from code diff")
        start_line_with_prefix: str = Field(..., description="The starting line number in the file (REQUIRED). \
                                            If the start_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        end_line_with_prefix: str = Field(..., description="The ending line number in the file (REQUIRED). \
                                          If the end_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        language: str = Field(..., description="The language of the code segment")
        issue: str = Field(..., description="The issue on the code segment")
        suggestion: str = Field(..., description="The suggestion to fix the code segment")
        suggestedCode: Optional[str] = Field(None, description="The updated code segment for the fix")
        severity: str = Field(..., description="The severity of the issue. Can be 'error', 'warning', or 'info'")

class FinalReviews(BaseModel):
    finalReviews: List[FinalReview] = Field(..., description="Final Reviews of Data of the Code.",)
    
structured_final_llm = llm_fin_gemini.with_structured_output(FinalReviews)

# Create an explicit Pydantic parser instance
final_reviews_parser = PydanticOutputParser(pydantic_object=FinalReviews)

# Create an OutputFixingParser for self-correction using the explicit parser
robust_parser = OutputFixingParser.from_llm(
    parser=final_reviews_parser,
    llm=llm_fin_gemini,
    max_retries=3,
)


# Schema for structured output to use in planning
class ReviewData(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)  # Add strict validation

        fileName: str = Field(..., description="The name of the file that has an issue")
        codeSegmentToFix: str = Field(..., description="The code segment that needs to be fixed from code diff")
        start_line_with_prefix: str = Field(..., description="The starting line number in the file (REQUIRED). \
                                            If the start_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        end_line_with_prefix: str = Field(..., description="The ending line number in the file (REQUIRED). \
                                          If the end_line is from the new file, indicate it with a '+' prefix, or if it is from the old file, indicate it with a '-' prefix")
        issue: str = Field(..., description="The issue on the code segment")

        @field_validator('start_line_with_prefix', 'end_line_with_prefix')
        def validate_line_prefix(cls, value):
            if not re.match(r'^[+-]\d+$', value):
                raise ValueError("Must start with '+' or '-' followed by digits")
            return value

class ReviewDatas(BaseModel):
    reviewDatas: List[ReviewData] = Field(description="Reviews of Data of the Code.",)

structured_llm = llm_gemini.with_structured_output(ReviewDatas)

structured_llm2 = llm_gemini2.with_structured_output(ReviewDatas)

structured_llm_nithila = llm_nithila.with_structured_output(ReviewDatas)
structured_llm_randinu = llm_randinu.with_structured_output(ReviewDatas)

structured_llm2 = llm_gemini2.with_structured_output(ReviewDatas)

structured_llm_nithila = llm_nithila.with_structured_output(ReviewDatas)
structured_llm_randinu = llm_randinu.with_structured_output(ReviewDatas)

def line_numbers_handle(start_line_with_prefix, end_line_with_prefix):
    value1 = start_line_with_prefix
    if(start_line_with_prefix[0]=='-'): 
        start_line = int(value1.replace("-", "").strip())  # Remove '+' and strip spaces
    else:
        start_line = int(value1.replace("+", "").strip())

    value2 = end_line_with_prefix
    if(end_line_with_prefix[0]=='-'):
        end_line = int(value2.replace("-", "").strip())
    else:
        end_line = int(value2.replace("+", "").strip()) 

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
    dependency_analysis: str
    error_issues: list[ReviewData]
    security_issues: list[ReviewData]
    performance_issues: list[ReviewData]
    quality_issues: list[ReviewData]
    other_issues: list[ReviewData]
    all_issues: str


#nodes
def error_handle(state: State):
     messages = prompt_templates.error_prompt.format_messages(
        PR_data=state["PR_data"],
        dependency_analysis=state["dependency_analysis"] # Add this line
    )
     try:
         logging.info("Invoking LLM for error handling...")
         response = structured_llm.invoke(messages)
         if(response == None):
             logging.info("LLM call for error handling successful. Found 0 issues.")
             return {"error_issues": []}
         else:
             logging.info(f"LLM call for error handling successful. Found {len(response.reviewDatas)} issues.")
         return {"error_issues": response.reviewDatas}
     except ValidationError as e:
         logging.error(f"Pydantic Validation Error in error_handle: {e}")
         # Log the specific failing input if possible (might require deeper langchain integration or modifying invoke)
         return {"error_issues": []} # Return empty list on validation failure
     except Exception as e:
         logging.error(f"Unexpected Error in error_handle: {e}", exc_info=True) # Log full traceback
         return {"error_issues": []} # Return empty list on other errors

def security_handle(state: State):
     messages = prompt_templates.security_prompt.format_messages(
        PR_data=state["PR_data"],
        dependency_analysis=json.dumps(state.get("dependency_analysis", {}), indent=2)  
    )
     try:
         logging.info("Invoking LLM for security handling...")
         response = structured_llm_nithila.invoke(messages)
         if(response == None):
             logging.info("LLM call for security handling successful. Found 0 issues.")
             return {"security_issues": []}
         else:
             logging.info(f"LLM call for security handling successful. Found {len(response.reviewDatas)} issues.")
         return {"security_issues": response.reviewDatas}
     except ValidationError as e:
         logging.error(f"Pydantic Validation Error in security_handle: {e}")
         return {"security_issues": []}
     except Exception as e:
         logging.error(f"Unexpected Error in security_handle: {e}", exc_info=True)
         return {"security_issues": []}

def performance_handle(state: State):
     messages = prompt_templates.performance_prompt.format_messages(
        PR_data=state["PR_data"],
        dependency_analysis=json.dumps(state.get("dependency_analysis", {}), indent=2)
    )
     try:
         logging.info("Invoking LLM for performance handling...")
         response = structured_llm_randinu.invoke(messages)
         if(response == None):
             logging.info("LLM call for performance handling successful. Found 0 issues.")
             return {"performance_issues": []}
         else:
             logging.info(f"LLM call for performance handling successful. Found {len(response.reviewDatas)} issues.")
         return {"performance_issues": response.reviewDatas}
     except ValidationError as e:
         logging.error(f"Pydantic Validation Error in performance_handle: {e}")
         return {"performance_issues": []}
     except Exception as e:
         logging.error(f"Unexpected Error in performance_handle: {e}", exc_info=True)
         return {"performance_issues": []}

def quality_handle(state: State):
     messages = prompt_templates.quality_prompt.format_messages(
        PR_data=state["PR_data"],
        dependency_analysis=json.dumps(state.get("dependency_analysis", {}), indent=2)  # Add this line
    )
     try:
         logging.info("Invoking LLM for quality handling...")
         response = structured_llm_randinu.invoke(messages)
         if(response == None):
             logging.info("LLM call for quality handling successful. Found 0 issues.")
             return {"quality_issues": []}
         else:
             logging.info(f"LLM call for quality handling successful. Found {len(response.reviewDatas)} issues.")
         return {"quality_issues": response.reviewDatas}
     except ValidationError as e:
         logging.error(f"Pydantic Validation Error in quality_handle: {e}")
         return {"quality_issues": []}
     except Exception as e:
         logging.error(f"Unexpected Error in quality_handle: {e}", exc_info=True)
         return {"quality_issues": []}

def other_handle(state: State):
     messages = prompt_templates.other_prompt.format_messages(
        PR_data=state["PR_data"],
        dependency_analysis=json.dumps(state.get("dependency_analysis", {}), indent=2)  # Add this line
    )
     try:
         logging.info("Invoking LLM for other handling...")
         response = structured_llm.invoke(messages)
         if(response == None):
             logging.info("LLM call for other handling successful. Found 0 issues.")
             return {"other_issues": []}
         else:
             logging.info(f"LLM call for other handling successful. Found {len(response.reviewDatas)} issues.")
         return {"other_issues": response.reviewDatas}
     except ValidationError as e:
         logging.error(f"Pydantic Validation Error in other_handle: {e}")
         return {"other_issues": []}
     except Exception as e:
         logging.error(f"Unexpected Error in other_handle: {e}", exc_info=True)
         return {"other_issues": []}

# Proper aggregator implementation
def aggregator(state: State):

    error_issues = state.get("error_issues", []) # Use .get for safety
    security_issues = state.get("security_issues", [])
    performance_issues = state.get("performance_issues", [])
    quality_issues = state.get("quality_issues", [])
    other_issues = state.get("other_issues", [])

    # --- Combine loops for slightly better readability ---
    def format_reviews(title: str, reviews: List[ReviewData], lines_list: List[str]):
        lines_list.append(title)
        if not reviews: # Handle case where a category might have no issues
             lines_list.append("No issues found in this category.")
             lines_list.append("---")
             return '\n'.join(lines_list)

        for review in reviews:
            lines_list.append(f"File: {review.fileName}")
            lines_list.append(f"Lines: {review.start_line_with_prefix} to {review.end_line_with_prefix}")
            lines_list.append("Code Segment:")
            lines_list.append(review.codeSegmentToFix)
            lines_list.append(f"Issue: {review.issue}")
            lines_list.append("---")
        return '\n'.join(lines_list)

    error_str = format_reviews("Error and Exception handling issues:", error_issues, [])
    security_str = format_reviews("Security vulnerability issues:", security_issues, [])
    performance_str = format_reviews("Performance issues:", performance_issues, [])
    quality_str = format_reviews("Quality issues:", quality_issues, [])
    other_str = format_reviews("Other issues:", other_issues, []) # Fixed typo "isses"

    final_str= error_str+"\n\n"+security_str+"\n\n"+performance_str+"\n\n"+quality_str+"\n\n"+other_str

    return {"all_issues": final_str}

def invoke(structured_diff_text: str, dependency_analysis: dict = None) -> dict:

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
    # display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

    state = parallel_workflow.invoke({
        "PR_data": structured_diff_text,
        "dependency_analysis": dependency_analysis
    })
    final_issues=state["all_issues"]

    return final_issues


#print(final_issues)

def _sanitize_json(text: str) -> str:
    """Escape orphan backslashes and raw newlines so json.loads never dies."""
    text = re.sub(r"(?<!\\)\\(?![nrt\"\\/bfu])", r"\\\\", text)
    return text.replace("\r\n", "\\n").replace("\n", "\\n")

final_prompt= ChatPromptTemplate.from_messages([
     ("system", """
        You are an expert code reviewer AI acting as a final synthesizer. Your goal is to transform raw identified issues into polished, actionable review comments suitable for a developer.

        You will receive two pieces of information:
        1.  `PR_data`: The complete code diff. Use this ONLY for context to understand the code surrounding an issue if necessary.
        2.  `Issues`: A pre-processed list or summary of specific problems identified by specialist agents. Each problem in `Issues` includes:
            * `File`: The filename.
            * `Lines`: The start and end line numbers with prefixes (e.g., `+10 to +12`, `-5 to -5`).
            * `Code Segment`: The exact code snippet related to the issue.
            * `Issue`: A description of the problem found by the specialist agent.

        Your Task:
        Iterate through EACH problem described in the `Issues` input. For every single problem:
        1.  **PRESERVE LOCATION (CRITICAL!):** You MUST extract and use the *exact* `fileName`, `start_line_with_prefix`, `end_line_with_prefix`, and `codeSegmentToFix` as provided for that problem within the `Issues` input. DO NOT modify, recalculate, or estimate these location details. The `start_line_with_prefix` and `end_line_with_prefix` MUST be copied verbatim (e.g., "+10", "-5").
        2.  **ANALYZE & ENRICH:** Based on the provided `issue` description and `codeSegmentToFix` (using `PR_data` for context if needed):
            * Identify the programming `language` of the `codeSegmentToFix`.
            * Generate a clear, actionable `suggestion` explaining *how* the developer can fix the issue.
            * Optionally, if practical, provide a corrected code snippet in `suggestedCode`. If not providing code, ensure this field is null.
      
        Important:
         - Examples for start_line_with_prefix when the start line is from new file: "+5, +2, +51, +61" 
         - Examples for start_line_with_prefix when the start line is from old file: "-8, -1, -56, -20" 

         - Examples for end_line_with_prefix when the end line is from new file: "+10, +2, +77, +65" 
         - Examples for end_line_with_prefix when the end line is from old file: "-1, -5, -22, -44" 
      
        {format_instructions}
        
        All newline characters inside `codeSegmentToFix` MUST be written as \\n.
    """),
    ("human", "PR data:\n{PR_data} \n\n Issues related: {Issues}"),
])

# Manually construct the runnable chain
structured_gemini_llm = final_prompt | llm_fin_gemini

structured_final_chain = (
    final_prompt
    | structured_final_llm
)


def final_review(pr_data: str, dependency_analysis: str) -> List[Dict]:
    print("Entered final review function")
     
    issues_summary = invoke(pr_data, dependency_analysis)  # existing helper

    payload_structured = {
        "PR_data": pr_data,
        "Issues": issues_summary,
    }

    #1 First attempt: rely on model‑native structured output
    try:
        logging.info("Calling structured output mode …")
        response = structured_final_chain.invoke({
            "PR_data": pr_data,
            "Issues":  issues_summary,
            "format_instructions": ""        
        })
        return [r.model_dump() for r in response.finalReviews]
    except Exception as primary_err:
        logging.warning("Structured route failed: %s", primary_err)

    #2 Fallback: text + robust parser
    payload_text = {
        **payload_structured,
        "format_instructions": final_reviews_parser.get_format_instructions(),
    }
    raw = structured_gemini_llm.invoke(payload_text).content

    # 2a. Try normal parser
    try:
        return [r.model_dump() for r in final_reviews_parser.parse(raw).finalReviews]
    except Exception as parse_err:
        logging.info("Initial parse failed (%s) – trying OutputFixingParser", parse_err)

    # 2b. Try OutputFixingParser
    try:
        fixed = robust_parser.parse(raw)
        return [r.model_dump() for r in fixed.finalReviews]
    except Exception as fix_err:
        logging.info("Fixer parser failed (%s) – trying sanitation", fix_err)

    # 2c. Last‑ditch sanitation + manual Pydantic
    try:
        safe_raw = _sanitize_json(raw)
        obj = json.loads(safe_raw)
        validated = FinalReviews(**obj)
        return [r.model_dump() for r in validated.finalReviews]
    except Exception as final_err:
        logging.info("dirtyjson also failed – attempting per‑item salvage")
        try:
            import dirtyjson  
            safe_obj = dirtyjson.loads(raw)
            validated = FinalReviews(**safe_obj)
            return [r.model_dump() for r in validated.finalReviews]
        except Exception:
            # Per‑item salvage if top‑level still broken
            salvaged: list[dict] = []
            for m in re.finditer(r"\{[^{}]+\}", raw):
                try:
                    frag = dirtyjson.loads(m.group(0))
                    salvaged.append(FinalReview(**frag).model_dump())
                except Exception:
                    continue
            if salvaged:
                logging.warning("Partial salvage succeeded: %d items", len(salvaged))
                return salvaged
        logging.error("Final review ultimately failed: %s", final_err, exc_info=True)
        return []