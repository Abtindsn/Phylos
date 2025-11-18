"""
Constructs the LangGraph StateGraph for the Narrative DNA Sequencer.

This file defines the core logic nodes, their interactions, and the overall
control flow of the recursive analysis.
"""
import os
import operator
import uuid
import hashlib
import difflib
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# --- API and Model Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OFFLINE_MODE = os.getenv("PHYLOS_OFFLINE_MODE", os.getenv("PHYLOS_OFFLINE", "auto")).lower()
FORCE_OFFLINE = OFFLINE_MODE in {"1", "true", "yes", "on", "offline", "stub"}
REQUEST_TIMEOUT = float(os.getenv("PHYLOS_GEMINI_TIMEOUT", "8"))

logger = logging.getLogger("phylos.graph")

if FORCE_OFFLINE:
    logger.info("Offline mode enforced via PHYLOS_OFFLINE flag.")

USE_GEMINI = bool(GEMINI_API_KEY) and not FORCE_OFFLINE

if USE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel('gemini-pro')
else:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found - running in offline stub mode.")
    llm = None

embedding_model = "models/embedding-001"
GRAPH_RECURSION_LIMIT = int(os.getenv("PHYLOS_RECURSION_LIMIT", "500"))
_executor = ThreadPoolExecutor(max_workers=2)
_gemini_embeddings_available = USE_GEMINI
_gemini_text_available = USE_GEMINI
STUB_EMBED_DIM = 128

# --- Core Utilities ---

def _stub_embedding(text: str) -> List[float]:
    """Deterministic hashing-based embedding for offline mode."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector: List[float] = []
    seed = digest
    while len(vector) < STUB_EMBED_DIM:
        for byte in seed:
            normalized = (byte / 255.0) * 2 - 1  # map to [-1, 1]
            vector.append(normalized)
            if len(vector) == STUB_EMBED_DIM:
                break
        seed = hashlib.sha256(seed).digest()
    return vector

def _stub_summary(parent: str, child: str) -> str:
    """Simple textual diff summary without LLM access."""
    if parent == child:
        return "Child text is identical to parent."
    matcher = difflib.SequenceMatcher(None, parent, child)
    overlap = matcher.quick_ratio()
    longest = matcher.find_longest_match(0, len(parent), 0, len(child))
    snippet = child[longest.b:longest.b + min(80, longest.size)]
    snippet = snippet or child[:80]
    return (
        f"Offline summary: similarity {overlap:.2f}. "
        f"New emphasis around: \"{snippet.strip()}\""
    )

def embedder(text: str) -> List[float]:
    """Generates embeddings using the Gemini API."""
    global _gemini_embeddings_available
    if not _gemini_embeddings_available:
        return _stub_embedding(text)

    logger.debug("Embedding content (first 50 chars): '%s...'", text[:50])
    def _call():
        return genai.embed_content(model=embedding_model, content=text, task_type="RETRIEVAL_DOCUMENT")

    try:
        embedding = _executor.submit(_call).result(timeout=REQUEST_TIMEOUT)
        return embedding['embedding']
    except (TimeoutError, Exception) as e:
        logger.warning("Embedding failed (%s). Falling back to offline vectors.", e)
        _gemini_embeddings_available = False
        return _stub_embedding(text)

def fetch_article_content(url: str) -> Dict[str, Any]:
    """Placeholder for a web scraping and content extraction service (e.g., using BeautifulSoup, Jina Reader)."""
    logger.info("Fetching content from URL: %s", url)
    # Dummy content for architectural purposes
    return {
        "id": url,
        "content": f"This is the simulated content for the article at {url}. It mentions another article: http://example.com/{uuid.uuid4()}",
        "author": "Author Name",
        "timestamp": "2025-11-18T12:00:00Z",
    }

def extract_links(content: str, base_url: str) -> List[str]:
    """Placeholder for a hyperlink extraction utility."""
    logger.debug("Extracting links from content.")
    import re
    urls = re.findall(r'https?://[^\s,"]+', content)
    return urls[:2] # Limit branching factor

def calculate_semantic_drift(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    import numpy as np
    if not isinstance(vec1, list) or not isinstance(vec2, list) or not vec1 or not vec2:
        return 0.0
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # We want drift, which is 1 - similarity
    drift = 1.0 - similarity
    logger.debug("Calculated semantic drift: %s", drift)
    return drift

def summarize_mutation(parent_content: str, child_content: str) -> str:
    """Uses the LLM to generate a summary of the semantic difference."""
    fallback = _stub_summary(parent_content, child_content)
    prompt = f"""
    Analyze the semantic difference between the two following texts.
    PARENT TEXT:
    ---
    {parent_content[:2000]}
    ---

    CHILD TEXT:
    ---
    {child_content[:2000]}
    ---

    Concisely describe the mutation or change in narrative, tone, or key information.
    Focus on what makes the CHILD TEXT different from the PARENT TEXT.
    """

    return generate_text_response(prompt, fallback)

def generate_text_response(prompt: str, fallback: str) -> str:
    """Generic helper to request text from the LLM with graceful fallback."""
    global _gemini_text_available
    if llm is None or not _gemini_text_available:
        return fallback

    def _call():
        return llm.generate_content(prompt)

    try:
        response = _executor.submit(_call).result(timeout=REQUEST_TIMEOUT)
        return response.text
    except (TimeoutError, Exception) as e:
        logger.warning("LLM text generation failed (%s). Falling back to stub.", e)
        _gemini_text_available = False
        return fallback


# --- Graph Node Definitions ---

def node_acquire(state: "GraphState") -> Dict[str, Any]:
    """
    Acquires the next article from the traversal queue.
    """
    logger.info("Node: Acquire")
    queue = list(state["traversal_queue"])
    url, parent_id, depth = queue.pop(0)

    article_data = fetch_article_content(url)
    article_data["embedding"] = embedder(article_data["content"])
    article_data["depth"] = depth

    return {
        "traversal_queue": queue,
        "current_article": article_data,
        "parent_article_id": parent_id,
        "current_depth": depth,
        "knowledge_graph": {
            "nodes": {article_data["id"]: article_data},
            "edges": []
        }
    }

def node_sequence(state: "GraphState") -> Dict[str, Any]:
    """
    Compares the current article to its parent to detect semantic mutations.
    """
    logger.info("Node: Sequence")
    current_article = state["current_article"]
    parent_id = state["parent_article_id"]
    knowledge_graph = state["knowledge_graph"]
    summary = "Initial article."

    if not current_article:
        return {}

    if not parent_id:
        # "Patient Zero" comparison to global context
        drift_score = calculate_semantic_drift(state["global_context"], current_article["embedding"])
        parent_id = "GLOBAL_CONTEXT"
    else:
        parent_article = knowledge_graph["nodes"].get(parent_id)
        if not parent_article:
            logger.error("Parent article %s not found in knowledge graph.", parent_id)
            return {}
        drift_score = calculate_semantic_drift(parent_article["embedding"], current_article["embedding"])
        summary = summarize_mutation(parent_article["content"], current_article["content"])

    # Define mutation threshold
    MUTATION_THRESHOLD = 0.3 # Adjusted for real embeddings

    relation_type = "Mutation" if drift_score > MUTATION_THRESHOLD else "Replication"
    logger.debug("Relation to parent: %s (Score: %s)", relation_type, drift_score)

    new_edge = {
        "source": parent_id,
        "target": current_article["id"],
        "attributes": {
            "mutation_score": drift_score,
            "relation_type": relation_type,
            "summary": summary
        }
    }

    return {
        "knowledge_graph": {
            "nodes": {},
            "edges": [new_edge]
        }
    }

def node_branch(state: "GraphState") -> Dict[str, Any]:
    """
    Extracts new hyperlinks to continue the traversal.
    """
    logger.info("Node: Branch")
    current_article = state["current_article"]
    current_depth = 0 if not current_article else current_article.get("depth", 0)

    if current_depth >= state["max_depth"]:
        logger.info("Max depth (%s) reached. Halting branching.", state["max_depth"])
        return {}

    new_links = extract_links(current_article["content"], current_article["id"])
    logger.info("Found %s new links to explore.", len(new_links))
    
    new_depth = current_depth + 1
    next_level_queue = [(link, current_article["id"], new_depth) for link in new_links]
    updated_queue = list(state["traversal_queue"]) + next_level_queue

    return {
        "traversal_queue": updated_queue,
    }


# --- Graph Control Flow ---

def should_continue(state: "GraphState") -> str:
    """
    Determines whether to continue the traversal or end.
    """
    logger.info("Control Node: Should Continue?")
    if not state["traversal_queue"]:
        logger.info("Queue empty. Ending traversal.")
        return END
    
    # Check depth of the *next* item without removing it
    _, _, next_depth = state["traversal_queue"][0]
    if next_depth > state["max_depth"]:
        logger.info("Next queue depth %s exceeds max depth %s. Ending.", next_depth, state["max_depth"])
        return END
        
    logger.info("Queue not empty and depth within limits. Continuing.")
    return "acquire"

# --- Graph Assembly ---

from state import GraphState

workflow = StateGraph(GraphState)

workflow.add_node("acquire", node_acquire)
workflow.add_node("sequence", node_sequence)
workflow.add_node("branch", node_branch)

workflow.set_entry_point("acquire")
workflow.add_edge("acquire", "sequence")
workflow.add_edge("sequence", "branch")

workflow.add_conditional_edges(
    "branch",
    should_continue,
    {
        "acquire": "acquire",
        END: END
    }
)

app = workflow.compile()

# For debugging: visualize the graph
try:
    graph_image = app.get_graph().draw_mermaid_png()
    with open("/home/abtin/Phylos/graph_visualization.png", "wb") as f:
        f.write(graph_image)
    logger.info("Graph visualization saved to graph_visualization.png")
except Exception as e:
    logger.warning("Could not draw graph visualization: %s", e)
