import json
from fastapi import FastAPI, Request, Header
from dotenv import load_dotenv
import requests

from llm_utils import final_review, line_numbers_handle
from github_utils import create_check_run, update_check_run, parse_diff_file_line_numbers, build_review_prompt_with_file_line_numbers
from authenticate_github import verify_signature, connect_repo

app = FastAPI()
load_dotenv()

@app.post("/webhook")
async def webhook(request: Request, x_hub_signature: str = Header(None)):
    payload = await request.body()
    verify_signature(payload, x_hub_signature)
    payload_dict = json.loads(payload)
    
    if "repository" in payload_dict:
        owner = payload_dict["repository"]["owner"]["login"]
        repo_name = payload_dict["repository"]["name"]
        repo = connect_repo(owner, repo_name)
        
        # Check if it's a pull_request event with action 'opened'
        if payload_dict.get("pull_request") and payload_dict.get("action") == "opened":
            pr_number = payload_dict["pull_request"]["number"]
            head_sha = payload_dict['pull_request']['head']['sha']
            
            check_run = None  # Initialize outside try block

            try:
                # Create initial check run
                check_run = create_check_run(repo, head_sha)
                
                #newly added to get pull request diff
                pull_request = repo.get_pull(pr_number)
                diff_url = pull_request.diff_url
                response = requests.get(diff_url)

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

                # Analyze code changes (your existing function)
                review_list = final_review(structured_diff_text)

                print("After llm call ...")
                
                # Update check run with results
                update_check_run(
                    check_run=check_run,
                    results=review_list
                )

                # Post each review item as a comment on the PR
                for review in review_list:
                    print("\n\n")
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
                    if(review['start_line_with_prefix'][0]=='-'): 
                        var_startSide = "LEFT"
                    else:
                        var_startSide = "RIGHT"
    
                    if(review['end_line_with_prefix'][0]=='-'):
                        var_side = "LEFT"
                    else:
                        var_side = "RIGHT"

                    if(start_line != end_line):
                        try:
                            pull_request.create_review_comment(
                            body=comment_body,
                            commit=repo.get_commit(head_sha),
                            path=review['fileName'],
                            start_line=start_line, #line number of the starting line of the code block
                            line=end_line, #line number of the ending line of the code block
                            start_side=var_startSide,  #side of the starting line of the code block
                            side=var_side,  # side of the ending line of the code block
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
                            side=var_side, 
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

            
    return {}
