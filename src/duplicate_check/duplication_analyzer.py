import sys
import os
import faulthandler

faulthandler.enable()
current_script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(os.path.dirname(current_script_dir))


if project_root not in sys.path: 
    sys.path.insert(0, project_root)


print("Current working directory:", os.getcwd())
print("sys.path at start:", sys.path)


import tempfile
import zipfile
import io
import shutil
import logging
import requests

from src.duplicate_check.duplication_test_5 import revised_pipeline

logger = logging.getLogger(__name__)

def perform_duplication_analysis(owner: str, repo_name: str, head_sha: str, issue) -> tuple[str, str, str, list]:
    """
    Downloads the repository, extracts it, runs duplication analysis,
    and returns results and status. Handles temporary directory cleanup internally.
    The output messages are tailored based on the similarity score.

    Args:
        owner (str): The owner of the repository.
        repo_name (str): The name of the repository.
        head_sha (str): The commit SHA of the pull request head.
        issue: The GitHub issue object to post comments to.

    Returns:
        tuple: A tuple containing (duplication_summary_message: str,
                                   duplication_conclusion: str,
                                   duplication_output_details: str,
                                   duplicate_results: list).
    """
    duplication_summary_message = "Duplication analysis not run."
    duplication_conclusion = "neutral"
    duplication_output_details = "No detailed duplication output."
    duplicate_results = []

    analysis_threshold = 0.998 

    archive_url = f"https://api.github.com/repos/{owner}/{repo_name}/zipball/{head_sha}"
    api_headers = {"Accept": "application/vnd.github.v3+json"}
    target_languages_for_duplication = ['python', 'java']

    try:
        logger.info(f"Downloading repository archive from: {archive_url}")
        archive_response = requests.get(archive_url, headers=api_headers, stream=True, timeout=180)
        archive_response.raise_for_status()

        with tempfile.TemporaryDirectory() as temp_dir_path:
            logger.info(f"Extracting repository to temporary directory: {temp_dir_path}")
            with zipfile.ZipFile(io.BytesIO(archive_response.content)) as zf:
                zf.extractall(temp_dir_path)

            extracted_items = os.listdir(temp_dir_path)
            if not extracted_items:
                raise Exception("Temporary directory is empty after archive extraction.")

            code_root_within_temp = os.path.join(temp_dir_path, extracted_items[0])
            if not os.path.isdir(code_root_within_temp):
                raise Exception(f"Extracted item '{extracted_items[0]}' is not a directory.")

            logger.info(f"Code to analyze is at: {code_root_within_temp}")

            logger.info("Starting duplication detection pipeline...")
            duplicate_results = revised_pipeline(
                codebase_path=code_root_within_temp,
                languages=target_languages_for_duplication,
                threshold=analysis_threshold 
            )
            logger.info("Duplication detection pipeline completed.")

            if duplicate_results:
                logger.info(f"Found {len(duplicate_results)} potential duplicate pairs.")
                duplication_output_details = "### Code Duplication Analysis Results:\n"
                
                exact_duplicates_count = 0
                near_duplicates_count = 0

                for pair_idx, pair in enumerate(duplicate_results):
                    similarity = pair['similarity']

                    if similarity == 1.0:
                        message = "These two functions exhibit **exact code duplication**."
                        exact_duplicates_count += 1
                    elif similarity >= analysis_threshold and similarity < 1.0: 
                        message = "We recommend reviewing for refactoring opportunities."
                        near_duplicates_count += 1
                    else:
                        message = "Please review these functions for potential duplication."

                    duplication_output_details += (
                        f"**Duplicate Pair {pair_idx+1}:**\n"
                        f"- **Function 1:** `{pair['function1']}` in `{pair['file1']}`\n"
                        f"- **Function 2:** `{pair['function2']}` in `{pair['file2']}`\n"
                        #f"- **Similarity Score:** {similarity:.4f}\n"
                        f"- **Analysis:** {message}\n---\n"
                    )
                
                # Refine summary message based on counts
                if exact_duplicates_count > 0 and near_duplicates_count > 0:
                    duplication_summary_message = (
                        f"Found {exact_duplicates_count} exact duplicate code pair(s) and "
                        f"{near_duplicates_count} near duplicate code pair(s). "
                        "Please see PR comments for detailed analysis."
                    )
                    duplication_conclusion = "action_required"
                elif exact_duplicates_count > 0:
                    duplication_summary_message = (
                        f"Found {exact_duplicates_count} exact duplicate code pair(s). "
                        "Immediate refactoring is highly recommended. See PR comments for details."
                    )
                    duplication_conclusion = "action_required"
                elif near_duplicates_count > 0:
                    duplication_summary_message = (
                        f"Found {near_duplicates_count} near duplicate code pair(s). "
                        "Review for refactoring opportunities. See PR comments for details."
                    )
                    duplication_conclusion = "action_required"
                else: # Fallback, should not happen if duplicate_results is not empty
                    duplication_summary_message = "Potential duplicate code pairs found. See PR comments for details."
                    duplication_conclusion = "action_required"

                try:
                    issue.create_comment(duplication_output_details)
                except Exception as e_comment_dup:
                    logger.error(f"Failed to post duplication results comment: {e_comment_dup}")

            else:
                logger.info("No significant duplicate code pairs found by the analysis.")
                duplication_summary_message = "No significant duplicate code pairs were identified by the analysis."
                duplication_conclusion = "success"

    except requests.exceptions.Timeout:
        logger.error(f"Timeout while downloading repository archive from {archive_url}.")
        duplication_summary_message = "Error: Repository archive download timed out. Unable to perform duplication analysis."
        duplication_conclusion = "failure"
    except requests.exceptions.RequestException as e_req:
        logger.error(f"Failed to download repository archive: {e_req}. Status: {e_req.response.status_code if e_req.response else 'N/A'}")
        duplication_summary_message = f"Error: Failed to download repository archive. Details: {e_req}."
        duplication_conclusion = "failure"
    except (zipfile.BadZipFile, FileNotFoundError, Exception) as e_extract:
        logger.error(f"Failed to extract or process repository archive: {e_extract}", exc_info=True)
        duplication_summary_message = f"Error: Failed to process repository archive during extraction or analysis. Details: {e_extract}."
        duplication_conclusion = "failure"
    except Exception as e_pipeline:
        logger.error(f"Error during the duplication detection pipeline: {e_pipeline}", exc_info=True)
        duplication_summary_message = f"An unexpected error occurred during the duplication detection pipeline. Details: {e_pipeline}."
        duplication_conclusion = "failure"

    finally:

                    if temp_dir_path and os.path.exists(temp_dir_path):
                        logger.info(f"Cleaning up temporary directory: {temp_dir_path}")
                        try:
                            shutil.rmtree(temp_dir_path)
                            logger.info("Temporary directory cleaned up successfully.")
                        except OSError as e:
                            logger.error(f"Error removing temporary directory {temp_dir_path}: {e}")    

    return duplication_summary_message, duplication_conclusion, duplication_output_details, duplicate_results