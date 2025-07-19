from unidiff import PatchSet
from typing import List, Dict

def parse_diff_file_line_numbers(diff_content: str) -> List[Dict]:
    """
    Parse a unified diff string (already decoded as str) and return a structured
    list of changes using actual file line numbers.
    Correctly extracts content without removing the first character.
    """
    # Initialize PatchSet with the string directly, without encoding
    patch = PatchSet(diff_content)
    parsed_files = []

    for patched_file in patch:
        # Determine the correct file path (handling added/removed files)
        file_name = patched_file.path
        if patched_file.target_file and patched_file.target_file.startswith('+++ b/'):
             file_name = patched_file.target_file[len('+++ b/'):]
        elif patched_file.source_file and patched_file.source_file.startswith('--- a/'):
             file_name = patched_file.source_file[len('--- a/'):]
        elif file_name.startswith('a/') or file_name.startswith('b/'):
             file_name = file_name[2:] # Strip a/ or b/ prefix if present

        if file_name == '/dev/null':
            continue

        file_info = {
            "file_name": file_name,
            "changes": []
        }
        current_hunk_lines = []
        for hunk in patched_file:
            for line in hunk:
                # FIX: Get content directly from line.value, only strip newline
                content = line.value.rstrip('\r\n')

                if line.is_added:
                    line_num = line.target_line_no
                    line_type = "added"
                    # NO SLICING NEEDED: content = line_content_raw[1:]
                elif line.is_removed:
                    line_num = line.source_line_no
                    line_type = "removed"
                    # NO SLICING NEEDED: content = line_content_raw[1:]
                else: # Context line
                    line_num = line.target_line_no # Use target for context
                    line_type = "context"
                    # NO SLICING NEEDED: content = line_content_raw[1:]

                if line_num is not None: # Ensure line number is valid
                    current_hunk_lines.append({
                        "type": line_type,
                        "line_number": line_num,
                        "content": content # Store the correct, full content
                    })

        if current_hunk_lines:
             file_info["changes"] = current_hunk_lines
             parsed_files.append(file_info)

    return parsed_files

def build_review_prompt_with_file_line_numbers(parsed_files: List[Dict]) -> str:
    prompt_lines = ["Code Diff to Analyze:", ""] # Add intro

    for file_data in parsed_files:
        file_name = file_data["file_name"]
        prompt_lines.append("=" * 40)
        prompt_lines.append(f"File: {file_name}")
        prompt_lines.append("-" * 40)

        if not file_data["changes"]:
            prompt_lines.append("(No changes in this file section provided)")
            prompt_lines.append("")
            continue

        for change in file_data["changes"]:
            sign = (
                "+" if change["type"] == "added" else
                "-" if change["type"] == "removed" else
                " " # Use space for context lines
            )

            prompt_lines.append(
                f"[Line {change['line_number']:<4}] {sign} {change['content']}"
            )
        prompt_lines.append("")

    return "\n".join(prompt_lines)
