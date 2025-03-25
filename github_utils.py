from unidiff import PatchSet
from typing import List, Dict

from llm_utils import line_numbers_handle

def create_check_run(repo, sha):
    """Create a check run using the modern PyGithub API"""
    return repo.create_check_run(
        name="AI Code Review",
        head_sha=sha,
        status="queued",  # Initial status should be 'queued'
        output={
            "title": "Analyzing Changes",
            "summary": "🔍 Scanning code changes with AI...",
            "text": "This may take 20-30 seconds"
        }
    )

def update_check_run(check_run, results):
    """Update check run with proper status transitions"""
    # First update to in_progress
    check_run.edit(
        status="in_progress",
        output={
            "title": "Processing...",
            "summary": "Analyzing code patterns"
        }
    )
    
    # Then update with final results
    annotations = []
    for result in results:

        start_line, end_line = line_numbers_handle(result['start_line_with_prefix'], result['end_line_with_prefix'])
        # Extract line numbers from your analysis results
        annotation = {
            "path": result['fileName'],
            "start_line": start_line,  # REQUIRED
            "end_line": end_line,      # REQUIRED
            "annotation_level": map_severity(result['severity']),
            "message": result['issue'],
            "raw_details": f"Suggestion: {result['suggestion']}\n\n{result.get('suggestedCode', '')}"
        }
            
        annotations.append(annotation)
    
    check_run.edit(
        status="completed",
        # conclusion="success" if len(annotations) == 0 else "action_required",
        conclusion="success",

        output={
            "title": f"Found {len(annotations)} items",
            "summary": "AI Code Review Results",
            "annotations": annotations[:50]  # GitHub limits to 50 annotations per update
        }
    )

def map_severity(level: str) -> str:
    """Map custom severity levels to GitHub annotation levels"""
    return {
        "error": "failure",
        "warning": "warning",
        "info": "notice"
    }.get(level.lower(), "notice")


def parse_diff_file_line_numbers(diff_content: str) -> List[Dict]:
    """
    Parse a unified diff string and return a structured list of changes using
    actual file line numbers.
    
    Returns a list of dicts, each representing a file change:
    {
      "file_name": str,
      "changes": [
          {
              "type": "added" | "removed" | "context",
              "line_number": int,  # For added or context lines, this is target_line_no.
                                   # For removed lines, use source_line_no.
              "content": str
          },
          ...
      ]
    }
    """
    patch = PatchSet(diff_content)
    parsed_files = []

    for patched_file in patch:
        file_info = {
            "file_name": patched_file.path,
            "changes": []
        }
        for hunk in patched_file:
            for line in hunk:
                # Decide which line number to use based on change type.
                if line.is_added or not line.is_removed:
                    line_num = line.target_line_no
                else:
                    line_num = line.source_line_no

                if line_num is None:
                    continue  # Skip lines without a valid number

                # Append each changed line along with its file-based line number.
                file_info["changes"].append({
                    "type": "added" if line.is_added else "removed" if line.is_removed else "context",
                    "line_number": line_num,
                    "content": line.value.rstrip("\n")
                })
        parsed_files.append(file_info)

    return parsed_files


def build_review_prompt_with_file_line_numbers(parsed_files: List[Dict]) -> str:
    """
    Create a prompt that includes the diff using actual file line numbers.
    """
    prompt_lines = []

    for file_data in parsed_files:
        prompt_lines.append("---------------------------------------")
        file_name = file_data["file_name"]
        prompt_lines.append(f"File name of the below changed lines: {file_name}\n")
        prompt_lines.append("Changed lines:")

        for change in file_data["changes"]:
            # Mark added lines with +, removed with -, context with a space
            sign = (
                "+" if change["type"] == "added" else
                "-" if change["type"] == "removed" else
                " "
            )
            prompt_lines.append(
                f"[Line {change['line_number']}] {sign} {change['content']}"
            )
        prompt_lines.append("\n")

    return "\n".join(prompt_lines)



