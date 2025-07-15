# build_faiss_index.py
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

print("--- Starting FAISS Index Build ---")

# Define the path where the index will be saved
FAISS_INDEX_PATH = "faiss_index"

# Get the Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("FATAL: GOOGLE_API_KEY environment variable not found for building the index.")

# Initialize the embeddings model
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    print(f"Error initializing embeddings: {e}")
    raise

# --- Create a small, placeholder dataset to initialize the index ---
# In a real-world scenario, you would load this from a database or a file
# containing historical PR data. For now, this is enough to create a valid index.
initial_texts = [
    "feat: Add user authentication endpoint",
    "fix: Correct calculation in payment processing",
    "refactor: Simplify database connection logic",
    "docs: Update installation instructions"
]

print(f"Creating FAISS index from {len(initial_texts)} initial documents...")

# Create the vector store from the initial documents
vector_store = FAISS.from_texts(texts=initial_texts, embedding=embeddings)

print(f"Saving FAISS index to: {FAISS_INDEX_PATH}")

# Save the newly created index to the specified path
vector_store.save_local(FAISS_INDEX_PATH)

print("--- FAISS Index Build Completed Successfully ---")