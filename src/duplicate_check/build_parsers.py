# src/duplicate_check/build_parsers.py
import os
from tree_sitter import Language

# This script is in /app/src/duplicate_check/
# We need to find the project root, which is /app/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VENDOR_DIR = os.path.join(PROJECT_ROOT, 'vendor')
BUILD_DIR = os.path.join(PROJECT_ROOT, 'src', 'duplicate_check', 'build')

# Create the build directory if it doesn't exist
os.makedirs(BUILD_DIR, exist_ok=True)

# A dictionary mapping the language name to its source directory
# IMPORTANT: Verify 'tree-sitter-python' and 'tree-sitter-java' are the correct folder names inside your 'vendor' directory.
languages = {
    "python": "tree-sitter-python",
    "java": "tree-sitter-java"
}

print(f"Starting build of language parsers in: {BUILD_DIR}")

# Loop through each language and build it as a separate .so file
for lang_name, lang_dir in languages.items():
    output_path = os.path.join(BUILD_DIR, f"{lang_name}.so")
    source_path = os.path.join(VENDOR_DIR, lang_dir)

    # Check if the source directory exists before trying to build
    if not os.path.isdir(source_path):
        print(f"ERROR: Source directory not found for {lang_name}: {source_path}")
        continue

    print(f"Building {lang_name} parser...")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_path}")

    Language.build_library(
      output_path,
      [source_path]
    )
    print(f"Successfully built {lang_name}.so")

print("All parsers built successfully.")