from unidiff import PatchSet
from typing import List, Dict

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
