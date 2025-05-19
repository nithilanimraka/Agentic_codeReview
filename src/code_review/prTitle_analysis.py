import os
import json
import hmac
import hashlib
import requests
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")  # Your Notion database ID

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("FATAL: GOOGLE_API_KEY environment variable not found.")
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

FAISS_INDEX_PATH = "faiss_index"
MAX_STORED_PRS = 11  # Maximum number of PRs to store in the vector store

app = FastAPI()

if os.path.exists(FAISS_INDEX_PATH):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings, # Pass the initialized embeddings object
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully.")
    except Exception as e:
        # Log the error and fallback
        logger.error(f"Error loading FAISS index: {e}. Vector store will be None.")
        vector_store = None
else:
    logger.warning(f"FAISS index file not found at {FAISS_INDEX_PATH}. Vector store will be initialized on first update.")
    vector_store = None


# async def verify_signature(request: Request):
#     signature = request.headers.get("X-Hub-Signature-256")
#     if not WEBHOOK_SECRET:
#         raise HTTPException(status_code=500, detail="WEBHOOK_SECRET is not set.")
#     if not signature:
#         raise HTTPException(status_code=403, detail="No GitHub signature provided.")
    
#     payload_body = await request.body()
#     computed_signature = "sha256=" + hmac.new(
#         WEBHOOK_SECRET.encode(), payload_body, hashlib.sha256
#     ).hexdigest()
    
#     if not hmac.compare_digest(computed_signature, signature):
#         raise HTTPException(status_code=403, detail="Invalid signature")

def load_pr_rules_from_notion():
    """Fetch PR rules from Notion with proper error handling and debugging"""
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        logger.error("Notion credentials not configured")
        return []

    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    try:
        logger.debug(f"Querying Notion database: {NOTION_DATABASE_ID}")
        response = requests.post(url, headers=headers, json={}, timeout=10)
        response.raise_for_status()
        data = response.json()

        logger.debug(f"Notion API response: {json.dumps(data, indent=2)}")

        rules = []
        for page in data.get("results", []):
            try:
                props = page.get("properties", {})

                rule = {
    "id": int(props.get("ID", {}).get("title", [{}])[0].get("plain_text", "0")),
    "rule": extract_notion_text(props.get("Rule")),
    "category": props.get("Category", {}).get("select", {}).get("name", ""),
    "severity": props.get("Severity", {}).get("select", {}).get("name", ""),
    "example_good": extract_notion_text(props.get("Good Example")),
    "example_bad": extract_notion_text(props.get("Bad Example"))
}


                if rule["rule"].strip():
                    rules.append(rule)
                    logger.debug(f"Loaded rule: {rule['id']} - {rule['rule'][:50]}...")

            except Exception as page_error:
                logger.error(f"Error processing page {page.get('id')}: {str(page_error)}")
                continue

        logger.info(f"Loaded {len(rules)} rules from Notion")
        return sorted(rules, key=lambda x: x["id"])

    except requests.exceptions.RequestException as e:
        logger.error(f"Notion API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"API response: {e.response.text}")
        return []

def extract_notion_text(property_obj):
    """Extract text from Notion property types"""
    if not property_obj:
        return ""
    value_type = property_obj.get("type")
    return " ".join([
        t.get("plain_text", "")
        for t in property_obj.get(value_type, [])
        if isinstance(t, dict)
    ])

def get_conversational_chain():
    prompt_template = """
*🔍 PR Analysis Report*

You are provided with:
•⁠  ⁠A *Pull Request Title*: "{title}"
•⁠  ⁠A *Rules Check* result validating the title: "{context}"
•⁠  ⁠The *Code Diff* for the current PR: "{diff}"
•⁠  ⁠A list of *Similar Past PRs*: "{similar_prs}"

*Your task is to:*
1.⁠ ⁠Rigorously analyze the title against three key factors:
   - Compliance with repository rules and conventions
   - Alignment with actual code changes
   - Consistency with historical PR patterns
2.⁠ ⁠Suggest an improved title when needed, incorporating:
   - Specific technical details from the diff
   - Clarity and action-oriented language
3.⁠ ⁠Provide concise, actionable feedback that:
   - Explains any title shortcomings

   📏 *Limit your total response to 250–300 words. Be concise but informative.*

*Response Format:*

*1️⃣ PR Title Analysis*
•⁠  ⁠*Current Title:* "{title}"
•⁠  ⁠*Rules Assessment:* 
  ✅/❌ [Concise] | ✅/❌ [Descriptive] | ✅/❌ [Follows conventions]
  [Detailed explanation of rule compliance]
•⁠  ⁠*Code Alignment:* 
  [How well the title matches the actual changes in the diff]

*2️⃣ Suggested Improvements*
•⁠  ⁠*Optimal Title:* 
  "[Your suggested title following best practices]"
•⁠  ⁠*Improvement Rationale:* 
  [Specific reasons why this is better, referencing rules/code]

*3️⃣ PR Purpose Summary*
•⁠  ⁠[One clear sentence explaining what the PR actually does]
•⁠  ⁠[Optional: Key technical details that should be highlighted]

*Example Output:*

*1️⃣ PR Title Analysis*
•⁠  ⁠*Current Title:* "Update config"
•⁠  ⁠*Rules Assessment:* 
  ❌ [Concise] | ❌ [Descriptive] | ❌ [Follows conventions]
  Title is too vague - good titles specify both what's changing and why
•⁠  ⁠*Code Alignment:* 
  Diff shows significant changes to database connection pooling settings

*2️⃣ Suggested Improvements*
•⁠  ⁠*Optimal Title:* 
  "Increase database connection pool size to 50 and add 30s timeout"
•⁠  ⁠*Improvement Rationale:* 
  Specifies both configuration changes (pool size + timeout) that match the diff

*3️⃣ PR Purpose Summary*
•⁠  ⁠Modifies database configuration to handle more concurrent connections with a timeout fallback
•⁠  ⁠Key changes: MAX_POOL_SIZE=50, CONN_TIMEOUT=30000
"""
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY 
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return model, prompt
    except Exception as e:
        logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
        # Propagate the error so the calling function knows initialization failed
        raise HTTPException(status_code=500, detail=f"Failed to initialize AI model: {e}") from e


def analyze_pr_with_diff(pr_title, pr_diff):
    """
    Analyze the PR title and code diff, and provide feedback using historical context.
    """
    rules = load_pr_rules_from_notion()
    context = "\n".join(
        f"Rule {rule['id']} ({rule['severity']}): {rule['rule']}\n"
        f"  Good: {rule['example_good']}\n"
        f"  Bad: {rule['example_bad']}\n"
        for rule in rules
    )

    # Retrieve similar past PRs from the FAISS vector store
    query = f"{pr_title}\n{pr_diff}"  # Combine title and diff for the query
    similar_prs = retrieve_relevant_prs(query, k=3)  # Retrieve top 3 similar PRs
    similar_prs_context = "\n\n".join(similar_prs)  # Combine into a single string

    # Prepare the input for the model
    model, prompt = get_conversational_chain()
    formatted_input = prompt.format(
        title=pr_title,
        context=context,
        diff=pr_diff,
        similar_prs=similar_prs_context
    )

    # Get the model's response
    response = model.invoke(formatted_input)
    return response.content

def update_faiss_store(pr_title, pr_diff, feedback):
    """
    Update the FAISS vector store with the PR title, diff, and feedback.
    Dynamically remove the least relevant PR if the maximum number of PRs is exceeded.
    """
    global vector_store
    embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY 
        )
    new_text = f"{pr_title}\n{pr_diff}\n{feedback}"

    # Check if the PR already exists in the vector store
    if vector_store is not None:
        stored_texts = vector_store.docstore._dict
        if new_text in [doc.page_content for doc in stored_texts.values()]:
            return  # Skip if the PR already exists

    # Initialize or update the vector store
    if vector_store is None:
        vector_store = FAISS.from_texts([new_text], embeddings)
    else:
        vector_store.add_texts([new_text])

    # Remove the oldest PR if the limit is exceeded
    if vector_store.index.ntotal > MAX_STORED_PRS:
        # Retrieve stored texts
        stored_texts = vector_store.docstore._dict
        all_texts = [doc.page_content for doc in stored_texts.values()]

        # Compute embeddings for all stored texts
        all_embeddings = [embeddings.embed_query(text) for text in all_texts]

        # Compute the embedding for the new text
        new_embedding = embeddings.embed_query(new_text)

        # Convert embeddings to NumPy arrays
        new_embedding_np = np.array(new_embedding)
        all_embeddings_np = np.array(all_embeddings)

        # Compute L2 distances between the new embedding and all stored embeddings
        distances = np.linalg.norm(all_embeddings_np - new_embedding_np, axis=1)

        # Find the index of the least relevant PR (farthest from the new PR)
        least_relevant_index = np.argmax(distances)

        # Remove the least relevant PR
        keys = list(stored_texts.keys())
        key_to_remove = keys[least_relevant_index]
        del stored_texts[key_to_remove]

        # Rebuild the FAISS index with the remaining PRs
        remaining_texts = [doc.page_content for doc in stored_texts.values()]
        if remaining_texts:  # Only rebuild if there are remaining PRs
            vector_store = FAISS.from_texts(remaining_texts, embeddings)
        else:  # If no PRs are left, reset the vector store to None
            vector_store = None

    # Save the updated FAISS index if it exists
    if vector_store is not None:
        vector_store.save_local(FAISS_INDEX_PATH)

def retrieve_relevant_prs(query, k=3):
    """
    Retrieve the top-k most relevant past PRs from the FAISS vector store.
    """
    if vector_store is None:
        return []  # Return empty list if the vector store is not initialized

    # Perform a similarity search
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]  # Return the content of the top-k PRs

# @app.post("/analyze")
# async def analyze_pr_request(request: Request):
#     await verify_signature(request)
#     data = await request.json()
    
#     if "pull_request" not in data:
#         return {"message": "Not a PR event"}
    
#     pr = data["pull_request"]
#     pr_title = pr.get("title", "No title provided.")
#     pr_diff = fetch_pr_diff(pr["diff_url"])
    
#     feedback = analyze_pr_with_diff(pr_title, pr_diff)
#     update_faiss_store(pr_title, pr_diff, feedback)
    
#     comments_url = pr["comments_url"]
#     comment_payload = {"body": feedback}
#     comment_response = requests.post(
#         comments_url,
#         headers={"Authorization": f"token {GITHUB_TOKEN}"},
#         json=comment_payload
#     )
    
#     if comment_response.status_code == 201:
#         return {"message": "Feedback posted successfully"}
#     else:
#         raise HTTPException(status_code=500, detail="Failed to post feedback")