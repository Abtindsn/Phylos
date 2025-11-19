"""
Defines the strict, typed state for the narrative analysis graph and the data models for API communication.
"""
from typing import TypedDict, List, Dict, Any, Annotated, Tuple
from pydantic import BaseModel, Field

# --- Pydantic Data Models for API and Data Structuring ---

class MutationEvent(BaseModel):
    """
    Represents a semantic mutation event when a narrative diverges significantly.
    This is the "signal" we are looking for.
    """
    source_id: str = Field(..., description="The ID (URL) of the parent article.")
    target_id: str = Field(..., description="The ID (URL) of the current, mutated article.")
    semantic_diff_summary: str = Field(
        ..., description="A brief, AI-generated summary of how the narrative changed."
    )
    drift_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The calculated semantic drift score, from 0.0 (identical) to 1.0 (completely different).",
    )

class InitialArticleRequest(BaseModel):
    """
    The request model for initiating a new narrative trace.
    """
    start_url: str = Field(..., description="The URL of the 'Patient Zero' article to begin the trace from.")
    max_depth: int = Field(default=3, description="The maximum recursion depth for tracing links.")


# --- LangGraph State Definition ---

class Article(TypedDict):
    """Represents a single node in our knowledge graph."""
    id: str  # URL
    content: str
    author: str | None
    timestamp: str | None
    embedding: List[float]
    outbound_links: List[str] | None

class KnowledgeGraph(TypedDict):
    """The accumulated knowledge graph."""
    nodes: Dict[str, Article] # Node ID (URL) -> Article data
    edges: List[Dict[str, Any]] # List of edges with metadata

def reduce_knowledge_graph(left: KnowledgeGraph, right: KnowledgeGraph) -> KnowledgeGraph:
    """Reducer to merge knowledge graph updates."""
    # A simple merge, assuming no conflicting updates to the same node/edge
    # In a real-world scenario, this might need more sophisticated logic
    return {
        "nodes": {**left.get("nodes", {}), **right.get("nodes", {})},
        "edges": left.get("edges", []) + right.get("edges", []),
    }

def replace_queue(_: List[Tuple[str, str | None, int]], right: List[Tuple[str, str | None, int]]) -> List[Tuple[str, str | None, int]]:
    """Reducer that always prefers the latest traversal queue."""
    return right

def merge_visited_urls(left: List[str] | None, right: List[str] | None) -> List[str]:
    """Reducer that keeps a deduplicated visitation list."""
    seen: set[str] = set()
    merged: List[str] = []
    for bucket in (left or []), (right or []):
        for url in bucket or []:
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged

def merge_host_counts(left: Dict[str, int] | None, right: Dict[str, int] | None) -> Dict[str, int]:
    """Reducer that keeps the max observed visit count per host."""
    merged: Dict[str, int] = dict(left or {})
    for host, count in (right or {}).items():
        merged[host] = max(merged.get(host, 0), count)
    return merged

class GraphState(TypedDict):
    """
    The central state of our recursive analysis engine.
    It's passed between nodes and updated at each step.
    """
    # The queue of (URL, parent_URL, current_depth) to visit
    traversal_queue: Annotated[List[Tuple[str, str | None, int]], replace_queue]

    # The accumulated graph of articles and their relationships
    knowledge_graph: Annotated[KnowledgeGraph, reduce_knowledge_graph]

    # Track visited URLs to prevent infinite loops
    visited_urls: Annotated[List[str], merge_visited_urls]

    # Track per-host visit counts so we can limit recursive hops on a single domain
    host_visit_counts: Annotated[Dict[str, int], merge_host_counts]

    # The vector embedding of the "Patient Zero" article
    global_context: List[float]

    # The current article being processed
    current_article: Article | None

    # The parent of the current article
    parent_article_id: str | None

    # Maximum depth for traversal
    max_depth: int

    # A list of warnings about data integrity issues encountered during the trace.
    data_warnings: List[str]
