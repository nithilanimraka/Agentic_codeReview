import os
import json
from fastapi import FastAPI, Request, Header,HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
import requests
import asyncio
import sys
from contextlib import asynccontextmanager

from src.code_review.llm_utils import final_review, line_numbers_handle
from src.github_con.github_utils import create_check_run, update_check_run, parse_diff_file_line_numbers, build_review_prompt_with_file_line_numbers
from src.github_con.authenticate_github import verify_signature, connect_repo
from src.code_review.prTitle_analysis import analyze_pr_with_diff, update_faiss_store
import logging
from src.code_review.git_repo_mcp import stream_git_repo_query,current_session_id, session_histories, QueryRequest, EndSessionRequest
from src.summarize_large_diff.summarization import count_tokens,analyze_pr


from src.dependency_analysis import ai_agents
from src.dependency_analysis import graph_workflow
from src.dependency_analysis import tools

load_dotenv()

tools.get_github_owner_repo
tools.close_neo4j_driver
tools.get_changed_files_from_pr
tools.initialize_neo4j_schema
graph_workflow.execute_analysis


# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Neo4j schema on startup
    tools.initialize_neo4j_schema()
    yield
    # Cleanup on shutdown remains the same

app = FastAPI()

class CodeReviewRequest(BaseModel):
    """Pydantic model for the /code-review endpoint request body."""
    diff_content: str

check_run = None  # Initialize globally

@app.post("/webhook")
async def webhook(request: Request, x_hub_signature: str = Header(None), background_tasks: BackgroundTasks = BackgroundTasks):
    payload = await request.body()
    verify_signature(payload, x_hub_signature)
    payload_dict = json.loads(payload)
    
    if "repository" in payload_dict:
        owner = payload_dict["repository"]["owner"]["login"]
        repo_name = payload_dict["repository"]["name"]
        repo = connect_repo(owner, repo_name)
        pr = payload_dict.get("pull_request", {})
        rp = payload_dict.get("repository", {})


        # Check if it's a pull_request event with action 'opened'
        if payload_dict.get("pull_request") and payload_dict.get("action") == "opened":
            pr_number = payload_dict["pull_request"]["number"]
            head_sha = payload_dict['pull_request']['head']['sha']
            action = payload_dict.get("action")
            base_branch = pr["base"]["ref"]
            repo_url = rp["html_url"]
        

            try:
                # Create initial check run
                check_run = create_check_run(repo, head_sha)
                
                #newly added to get pull request diff
                pull_request = repo.get_pull(pr_number)

                pr_title = pull_request.title
                if not pr_title:  # Checks if title is None or an empty string ""
                    print(f"Warning: Pull Request #{pr_number} has an empty title. Using a default.")
                    pr_title = f"[No Title Provided for PR #{pr_number}]" # Assign a default title
              
                #Get diff
                diff_url = pull_request.diff_url
                response = requests.get(diff_url)
                diff_text = response.text

                token_count = count_tokens(diff_text)
                MAX_TOKENS_PER_CHUNK=970000  # Set a threshold for large diffs

                if token_count > MAX_TOKENS_PER_CHUNK:
                    # Handle large diff by calling analyze_chunks endpoint
                    logger.info(f"Diff too large ({token_count} tokens), using chunked analysis")
                    
                    # Post initial comment
                    issue = repo.get_issue(number=pr_number)
                    issue.create_comment(
                        "Hi, I'm analyzing this large PR. This might take a few moments..."
                    )
                    
                    # Call analyze_chunks endpoint
                    analysis_comment = analyze_pr(pr_title,diff_text)
                    
                    # Post the final analysis
                    issue.create_comment(analysis_comment)
                    
                    try:
                      check_run.edit(
                        status='completed',
                        conclusion='success',
                        output={
                          'title': 'Large PR Detected',
                         'summary': 'Analysis performed via comments only'
                      }
                         )
                    except Exception as e:
                        logger.warning(f"Check run completion failed: {str(e)}")
                    
                else:

                    # Parse the diff to extract actual file line numbers.
                    parsed_files = parse_diff_file_line_numbers(response.text)
        
                    # Build a structured diff text for the prompt.
                    structured_diff_text = build_review_prompt_with_file_line_numbers(parsed_files)
                    print(structured_diff_text)

                    print("Before llm call...")

                    issue = repo.get_issue(number=pr_number)
                    issue.create_comment(
                        "Hi, I am a code reviewer bot. I will analyze the PR and provide detailed review comments."
                    )

                    feedback_pr= analyze_pr_summary(pr_title,response.text)
                    
                    if feedback_pr: # Only post if feedback was successfully generated
                        try:
                            posted_comment = issue.create_comment(body=feedback_pr)
                            logger.info(f"Successfully posted feedback comment: {posted_comment.html_url}")
                        except Exception as e:
                            logger.error(f"An unexpected error occurred posting feedback: {str(e)}", exc_info=True)
                    else:
                        logger.warning("No feedback was generated or analysis failed, skipping comment posting.")



                    # Get owner/repo
                    owner, repo_name = tools.get_github_owner_repo(repo_url)
                    if not owner or not repo_name:
                        raise HTTPException(400, "Could not parse repository info")

                    logger.info(
                        f"Processing PR #{pr_number} in {owner}/{repo_name} "
                        f"(Action: {action}, Commit: {head_sha[:7]})"
                    )

                        # Filter relevant actions
                    if action not in ["opened", "reopened", "synchronize"]:
                        return {"status": "ignored", "reason": f"Action '{action}' not supported"}

                        # Fetch changed files
                    try:
                        changed_files = tools.get_changed_files_from_pr(repo, pr_number)
                        logger.info(f"Found {len(changed_files)} changed files")
                    except Exception as e:
                        logger.error(f"Failed to fetch changed files: {str(e)}")
                        raise HTTPException(502, "Could not retrieve PR files")

                    # Analyze code changes (your existing function)
                    pr_data = {
                    "repo_url": repo_url,
                    "commit_sha": head_sha,
                    "base_branch": base_branch,
                    "pull_request_number": pr_number,
                    "changed_files": changed_files,
                    "owner": owner,
                    "repo": repo_name
                    }

                    analysis_result = execute_analysis_and_handle_result(pr_data, structured_diff_text)
                    dependency_data = analysis_result.get("dependency_analysis", {})
                    review_list = final_review(
                                    structured_diff_text, 
                                    dependency_analysis=dependency_data
                                )

                    print("After llm call ...")
                    
                    # Update check run with results
                    update_check_run(
                        check_run=check_run,
                        results=review_list
                    )

                    # Post each review item as a comment on the PR
                    for review in review_list:
                        print("\n\n========================")
                        print(review)

                        # Get the line numbers (int) for the review
                        start_line, end_line = line_numbers_handle(review['start_line_with_prefix'], review['end_line_with_prefix'])


                        prog_lang = review.get('language', '')  # Default to an empty string if 'language' is missing
                        comment_body = (
                            f"**Issue:** {review['issue']}\n\n"
                            f"**Severity:** {review['severity']}\n\n"
                            f"**Suggestion:** {review['suggestion']}\n"
                        )
                        
                        # If suggestedCode exists, add it to the comment
                        if review.get("suggestedCode"):
                            comment_body += f"```{prog_lang}\n{review['suggestedCode']}\n```"

                        #Check whether the start_line and end_line are from new file or old file
                        start_line_side = "LEFT" if review['start_line_with_prefix'][0] == '-' else "RIGHT"
                        end_line_side = "LEFT" if review['end_line_with_prefix'][0] == '-' else "RIGHT"

                        if(start_line != end_line):
                            try:
                                pull_request.create_review_comment(
                                body=comment_body,
                                commit=repo.get_commit(head_sha),
                                path=review['fileName'],
                                start_line=start_line, #line number of the starting line of the code block
                                line=end_line, #line number of the ending line of the code block
                                start_side=start_line_side,  #side of the starting line of the code block
                                side=end_line_side,  # side of the ending line of the code block
                                )
                            except Exception as e:
                                print(f"Failed to post comments: {str(e)}")
                                if hasattr(e, 'data'):
                                    print("Error details:", json.dumps(e.data, indent=2))
                                else:
                                    print("No valid comments to post")

                        else:
                            try:
                                pull_request.create_review_comment(
                                body=comment_body,
                                commit=repo.get_commit(head_sha),
                                path=review['fileName'],
                                line=end_line,
                                side=end_line_side, 
                                )
                            except Exception as e:
                                print(f"Failed to post comments: {str(e)}")
                                if hasattr(e, 'data'):
                                    print("Error details:", json.dumps(e.data, indent=2))
                                else:
                                    print("No valid comments to post")

                    
            except Exception as e:
                # Only update check run if it was successfully created
                if check_run is not None:
                    check_run.edit(
                        status="completed",
                        conclusion="failure",
                        output={
                            "title": "Analysis Failed",
                            "summary": f"Error: {str(e)}"
                        }
                    )
                else:
                    # Fallback error handling
                    print(f"Critical failure before check run creation: {str(e)}")
                    
                raise


            # Trigger analysis
            # background_tasks.add_task(execute_analysis_and_handle_result, pr_data, structured_diff_text)
            return {
                "status": "analysis_started",
                "pr_number": pr_number,
                "repo": f"{owner}/{repo_name}"
            }


            
    return {}

def analyze_pr_summary(pr_title,code_diff):
    feedback = None # Initialize feedback to None
    try:
        logger.info("Starting PR analysis with diff...")
        feedback = analyze_pr_with_diff(pr_title, code_diff)
        logger.info("Successfully received feedback from analyze_pr_with_diff.")

        update_faiss_store(pr_title, code_diff, feedback)

    except HTTPException as http_exc:
        logger.error(f"HTTPException during PR analysis: {http_exc.status_code} - {http_exc.detail}")
        if check_run:
            check_run.edit(status="completed", conclusion="failure", output={"title": "Analysis Configuration Error", "summary": f"Error: {http_exc.detail}"})
        raise # Re-raise to let FastAPI handle it (will result in 5xx response)
    except Exception as analysis_err:
        logger.error(f"Unexpected error during PR analysis or FAISS update: {analysis_err}", exc_info=True) # Log traceback
        if check_run:
            check_run.edit(status="completed", conclusion="failure", output={"title": "Analysis Error", "summary": f"Unexpected error during analysis: {str(analysis_err)}"})

    return feedback


def execute_analysis_and_handle_result(pr_data: dict, structured_diff_text: str) -> dict:
    """Synchronous execution of analysis workflow"""
    try:
        logger.info(f"Starting analysis for PR #{pr_data['pull_request_number']}")
        results = graph_workflow.execute_analysis(pr_data, structured_diff_text)
        return {
            "dependency_analysis": results.get("dependency_analysis"),
            "error": results.get("error")
        }
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return {"error": str(e)}


@app.post("/code-review")
async def code_review_endpoint(review_request: CodeReviewRequest):
    """
    Receives diff content, runs the analysis pipeline, and returns the review list as JSON.
    """
    logger.info("Received request for /code-review")
    try:
        diff_content = review_request.diff_content
        if not diff_content:
            raise HTTPException(status_code=400, detail="diff_content cannot be empty.")

        logger.info("Parsing diff content...")
        parsed_files = parse_diff_file_line_numbers(diff_content)

        logger.info("Building review prompt...")
        structured_prompt = build_review_prompt_with_file_line_numbers(parsed_files)

        logger.info("Generating final review...")
        # Run the potentially long-running LLM call in a thread pool
        review_list = await asyncio.to_thread(final_review, structured_prompt)
        logger.info(f"Code review generated successfully with {len(review_list)} items.")

        return JSONResponse(content=review_list)

    except HTTPException as http_exc:
        logger.error(f"HTTP error during code review: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Error during code review processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during code review: {str(e)}")




async def generator_with_session(session_id, repo_path, query):
    token = current_session_id.set(session_id)
    try:
        async for chunk in stream_git_repo_query(session_id, repo_path, query):
            yield chunk
    finally:
        current_session_id.reset(token)

@app.post("/analyze")
async def analyze_repository(request: QueryRequest):
    """
    Endpoint to receive repository path, query, and session_id,
    and stream back the analysis using conversation history.
    """
    print(f"Received request: session_id='{request.session_id}', repo_path='{request.repo_path}', query='{request.query}'")

    # Basic validation
    if not os.path.isdir(request.repo_path):
         raise HTTPException(status_code=400, detail=f"Invalid repository path: {request.repo_path}")
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.session_id:
         raise HTTPException(status_code=400, detail="Session ID is required.")

    return StreamingResponse(
        generator_with_session(request.session_id, request.repo_path, request.query),
        media_type="text/plain"
    )

@app.post("/end_session")
async def end_session(request: EndSessionRequest):
    """
    Endpoint to remove a session's history from memory.
    """
    session_id = request.session_id
    if session_id in session_histories:
        del session_histories[session_id]
        print(f"Session {session_id}: History deleted.")
        return {"status": "success", "message": f"Session {session_id} ended."}
    else:
        print(f"Session {session_id}: Attempted to delete non-existent session.")
        return {"status": "not_found", "message": f"Session {session_id} not found."}


@app.get("/") 
async def read_root():
    return {"message": "Git Analyzer FastAPI server is running. POST to /analyze"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



