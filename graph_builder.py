"""
Constructs the LangGraph StateGraph for the Narrative DNA Sequencer.

This file defines the core logic nodes, their interactions, and the overall
control flow of the recursive analysis.
"""
import operator
import uuid
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List

# Assuming a generic embedding function and web content utilities exist
# In a real implementation, these would be robust services.
def embedder(text: str) -> List[float]:
    """Placeholder for a sentence transformer or OpenAI embedding model."""
    # This is a dummy implementation.
    # A real one would call an actual model.
    print(f"--- Embedding content (first 50 chars): '{text[:50]}...'")
    return [len(text) / 1000.0] + [0.0] * 767 # Dummy 768-dim vector

def fetch_article_content(url: str) -> Dict[str, Any]:
    """Placeholder for a web scraping and content extraction service (e.g., using BeautifulSoup, Jina Reader)."""
    print(f"--- Fetching content from URL: {url}")
    # Dummy content for architectural purposes
    return {
        "id": url,
        "content": f"This is the simulated content for the article at {url}. It mentions another article: http://example.com/{uuid.uuid4()}",
        "author": "Author Name",
        "timestamp": "2025-11-18T12:00:00Z",
    }

def extract_links(content: str, base_url: str) -> List[str]:
    """Placeholder for a hyperlink extraction utility."""
    print(f"--- Extracting links from content...")
    # Dummy extraction for architectural purposes
    # In reality, this would use regex or a library like BeautifulSoup.
    import re
    # A very basic regex to find URLs
    urls = re.findall(r'https?://[^\s,"]+', content)
    return urls[:2] # Limit branching factor for this example

def calculate_semantic_drift(vec1: List[float], vec2: List[float]) -> float:
    """Placeholder for cosine similarity or other vector distance metric."""
    # Dummy calculation
    if not vec1 or not vec2:
        return 0.0
    drift = abs(vec1[0] - vec2[0])
    print(f"--- Calculated semantic drift: {drift}")
    return drift

# --- Graph Node Definitions ---

def node_acquire(state: "GraphState") -> Dict[str, Any]:
    """
    Acquires the next article from the traversal queue.
    This is the entry point for each recursive step.
    """
    print(">>> In Node: Acquire")
    queue = state["traversal_queue"]
    url, parent_id, depth = queue.pop(0)

    article_data = fetch_article_content(url)
    article_data["embedding"] = embedder(article_data["content"])

    return {
        "traversal_queue": queue,
        "current_article": article_data,
        "parent_article_id": parent_id,
        "knowledge_graph": {
            "nodes": {article_data["id"]: article_data},
            "edges": []
        }
    }

def node_sequence(state: "GraphState") -> Dict[str, Any]:
    """
    The "brain" of the operation. Compares the current article to its parent
    to detect semantic mutations.
    """
    print(">>> In Node: Sequence")
    current_article = state["current_article"]
    parent_id = state["parent_article_id"]
    knowledge_graph = state["knowledge_graph"]

    if not current_article or not parent_id:
        # This is "Patient Zero", it has no parent to compare against.
        # Its relationship is to the global context.
        drift_score = calculate_semantic_drift(state["global_context"], current_article["embedding"])
        parent_id = "GLOBAL_CONTEXT"
    else:
        parent_article = knowledge_graph["nodes"].get(parent_id)
        if not parent_article:
            # Should not happen in a well-formed graph
            print(f"!!! ERROR: Parent article {parent_id} not found in knowledge graph.")
            return {}
        drift_score = calculate_semantic_drift(parent_article["embedding"], current_article["embedding"])

    # Define mutation threshold
    MUTATION_THRESHOLD = 0.1 # Dummy value

    relation_type = "Mutation" if drift_score > MUTATION_THRESHOLD else "Replication"
    print(f"--- Relation to parent: {relation_type} (Score: {drift_score})")

    new_edge = {
        "source": parent_id,
        "target": current_article["id"],
        "attributes": {
            "mutation_score": drift_score,
            "relation_type": relation_type,
            "summary": f"Drift of {drift_score:.2f} detected." # In reality, an LLM would generate this.
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
    Extracts new, valid hyperlinks from the current article to continue
    the recursive traversal.
    """
    print(">>> In Node: Branch")
    current_article = state["current_article"]
    queue = state["traversal_queue"]
    
    # Find the depth of the current article to calculate child depth
    # This is a bit clunky and could be improved by passing depth more directly
    current_depth = 0
    # This is a simplified way to get the depth. A more robust solution might pass
    # the current depth within the state explicitly.
    if state.get('parent_article_id'):
        # A real system would need a more reliable way to track depth.
        # For this scaffold, we assume depth increases by 1 from parent.
        # This part of the logic is complex to get right without a proper
        # state flow for depth, so we'll approximate.
        # Let's assume the depth was passed correctly in the queue tuple.
        # The 'traversal_queue_history' is a hack to reconstruct this.
        pass


    if state.get("current_depth", 0) >= state["max_depth"]:
        print("--- Max depth reached. Halting branching from this node.")
        return {}

    new_links = extract_links(current_article["content"], current_article["id"])
    print(f"--- Found {len(new_links)} new links to explore.")
    
    # Add new links to the queue for the next level of traversal
    # The depth of these new links is the current article's depth + 1.
    # This requires knowing the current depth accurately.
    # Let's assume the depth is part of the state that gets passed around.
    # The tuple in traversal_queue is (url, parent_id, depth).
    # The 'acquire' node pops it, but we need to track it.
    
    # This is a simplification. A robust implementation would manage depth more explicitly.
    # For now, we'll assume a depth of 1 for any children found.
    new_depth = state.get("current_depth", 0) + 1
    next_level_queue = [(link, current_article["id"], new_depth) for link in new_links]

    return {
        "traversal_queue": next_level_queue,
    }


# --- Graph Control Flow ---

def should_continue(state: "GraphState") -> str:
    """
    Determines whether to continue the traversal or end.
    """
    print(">>> In Control Node: Should Continue?")
    if state["traversal_queue"]:
        # We also need to check the depth constraint of the items in the queue
        next_url, _, next_depth = state["traversal_queue"][0]
        if next_depth > state["max_depth"]:
            print(f"--- Next item in queue (depth {next_depth}) exceeds max depth ({state['max_depth']}). Ending.")
            return END
        print("--- Queue is not empty and depth is within limits. Continuing.")
        return "acquire"
    else:
        print("--- Queue is empty. Ending graph traversal.")
        return END

# --- Graph Assembly ---

from state import GraphState

# The global state reducer graph
workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("acquire", node_acquire)
workflow.add_node("sequence", node_sequence)
workflow.add_node("branch", node_branch)

# Define the edges
workflow.set_entry_point("acquire")
workflow.add_edge("acquire", "sequence")
workflow.add_edge("sequence", "branch")

# Add the conditional edge for recursion
workflow.add_conditional_edges(
    "branch",
    should_continue,
    {
        "acquire": "acquire",
        END: END
    }
)

# Compile the state machine
app = workflow.compile()

# For debugging: visualize the graph
try:
    # The `get_graph` method returns a `Graph` object that can be drawn
    graph_image = app.get_graph().draw_mermaid_png()
    with open("/home/abtin/Phylos/graph_visualization.png", "wb") as f:
        f.write(graph_image)
    print("--- Graph visualization saved to graph_visualization.png")
except Exception as e:
    print(f"Could not draw graph: {e}. Please ensure graphviz and its dependencies are installed.")
