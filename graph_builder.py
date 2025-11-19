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
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import re

# --- API and Model Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    # Sanitize the key in case it was loaded with quotes or whitespace
    GEMINI_API_KEY = GEMINI_API_KEY.strip().strip('"').strip("'")

ECO_MODE = os.getenv("PHYLOS_ECO_MODE", "true").lower() in {"1", "true", "yes", "on"}
OFFLINE_MODE = os.getenv("PHYLOS_OFFLINE_MODE", os.getenv("PHYLOS_OFFLINE", "auto")).lower()
FORCE_OFFLINE = OFFLINE_MODE in {"1", "true", "yes", "on", "offline", "stub"}
REQUEST_TIMEOUT = float(os.getenv("PHYLOS_GEMINI_TIMEOUT", "8"))
HOST_VISIT_LIMIT = int(os.getenv("PHYLOS_MAX_VISITS_PER_HOST", "8"))
ORIGIN_INSIGHT_MODEL_NAME = os.getenv("PHYLOS_ORIGIN_INSIGHT_MODEL", "gemini-2.0-flash-exp")
HTTP_HEADERS = {
    "User-Agent": os.getenv(
        "PHYLOS_HTTP_USER_AGENT",
        "PhylosCrawler/1.0 (+https://github.com/abtin/Phylos)"
    )
}
# Block obvious stub/share domains. Users can override via PHYLOS_LINK_BLOCKLIST.
DEFAULT_BLOCKLIST = {
    "example.com",
    "offline.phylos",
}
LINK_BLOCKLIST_SUFFIXES = {
    host.strip().lower()
    for host in os.getenv("PHYLOS_LINK_BLOCKLIST", ",".join(DEFAULT_BLOCKLIST)).split(",")
    if host.strip()
}
SOCIAL_SHARE_HOSTS = {
    "facebook.com",
    "m.facebook.com",
    "twitter.com",
    "x.com",
    "t.co",
    "reddit.com",
    "www.reddit.com",
    "pinterest.com",
    "instagram.com",
    "threads.net",
    "linkedin.com",
    "youtube.com",
    "youtu.be",
    "bsky.app",
    "tiktok.com",
    "whatsapp.com",
}
SOCIAL_SHARE_KEYWORDS = {"share", "sharer", "intent", "compose", "bookmark", "login", "post"}
RAW_URL_PATTERN = re.compile(r'https?://[^\s"\'>)]+')
DISALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".webm", ".mp4", ".mp3", ".mov",
    ".pdf", ".zip", ".rar", ".iso", ".dmg", ".exe", ".bin", ".apk", ".tar", ".gz"
}

logger = logging.getLogger("phylos.graph")

def _normalize_model_name(value: str | None) -> str | None:
    if not value:
        return None
    name = value.strip().strip('"').strip("'")
    if not name:
        return None
    if "/" in name:
        name = name.split("/")[-1]
    return name

def _unique_preserve(values):
    seen = set()
    deduped = []
    for item in values:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped

MODEL_LABEL_OVERRIDES = {
    "gemini-1.5-flash": "Gemini 1.5 Flash",
    "gemini-1.5-flash-latest": "Gemini 1.5 Flash",
    "gemini-1.5-flash-8b": "Gemini 1.5 Flash 8B",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
    "gemini-1.5-pro-latest": "Gemini 1.5 Pro",
    "gemini-2.0-flash-exp": "Gemini 2.0 Flash (Experimental)",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite",
    "gemini-2.0-flash-lite-preview": "Gemini 2.0 Flash Lite (Preview)",
    "gemini-2.0-flash-thinking": "Gemini 2.0 Flash Thinking",
    "gemini-2.0-flash-thinking-exp": "Gemini 2.0 Flash Thinking (Experimental)",
    "gemini-2.0-flash-thinking-exp-01-21": "Gemini 2.0 Flash Thinking (Exp 01-21)",
    "gemini-2.0-pro": "Gemini 2.0 Pro",
    "gemini-2.0-pro-exp": "Gemini 2.0 Pro (Experimental)",
    "gemini-2.0-pro-exp-02-05": "Gemini 2.0 Pro (Exp 02-05)",
    "gemini-1.0-pro": "Gemini 1.0 Pro",
    "gemini-pro": "Gemini Pro",
    "gemini-3.0": "Gemini 3.0",
    "gemini-3.0-flash": "Gemini 3.0 Flash",
    "gemini-3.0-pro": "Gemini 3.0 Pro",
}

def _prettify_model_label(name: str) -> str:
    if not name:
        return "Gemini"
    if name in MODEL_LABEL_OVERRIDES:
        return MODEL_LABEL_OVERRIDES[name]
    parts = name.replace("models/", "").replace("_", "-").split("-")
    pretty = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            pretty.append(part)
        elif len(part) == 1:
            pretty.append(part.upper())
        else:
            pretty.append(part.capitalize())
    return " ".join(pretty) or name

DEFAULT_CHAT_MODEL = _normalize_model_name(os.getenv("PHYLOS_DEFAULT_CHAT_MODEL", "gemini-1.5-flash"))
if not DEFAULT_CHAT_MODEL:
    DEFAULT_CHAT_MODEL = "gemini-pro"

_default_fallback_order = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-preview",
    "gemini-2.0-flash-thinking",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-exp",
    "gemini-2.0-pro",
    "gemini-2.0-pro-exp-02-05",
    "gemini-3.0",
    "gemini-3.0-flash",
    "gemini-3.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",
    "gemini-pro",
]
_fallback_env = os.getenv(
    "PHYLOS_CHAT_MODEL_FALLBACKS",
    ",".join(_default_fallback_order),
)
CHAT_MODEL_FALLBACKS = _unique_preserve(
    _normalize_model_name(item.strip())
    for item in _fallback_env.split(",")
    if item.strip()
)
AVAILABLE_GEMINI_MODELS: Dict[str, str] = {}
_MODEL_CLIENTS: Dict[str, genai.GenerativeModel] = {}

def _refresh_available_models() -> None:
    if not USE_GEMINI:
        return
    try:
        for model in genai.list_models():
            name = _normalize_model_name(getattr(model, "name", None))
            if not name:
                continue
            supported = set(getattr(model, "supported_generation_methods", []))
            if "generateContent" not in supported:
                continue
            AVAILABLE_GEMINI_MODELS[name] = getattr(model, "display_name", name)
    except Exception as exc:
        logger.debug("Unable to list Gemini models: %s", exc)

def _model_priority(preferred: str | None = None) -> List[str]:
    return _unique_preserve(
        [
            _normalize_model_name(preferred),
            DEFAULT_CHAT_MODEL,
            *CHAT_MODEL_FALLBACKS,
        ]
    )

def _get_model_client(name: str | None):
    normalized = _normalize_model_name(name)
    if not normalized:
        return None
    if normalized in _MODEL_CLIENTS:
        return _MODEL_CLIENTS[normalized]
    try:
        client = genai.GenerativeModel(normalized)
        _MODEL_CLIENTS[normalized] = client
        return client
    except Exception as exc:
        logger.debug("Failed to initialize Gemini model %s: %s", normalized, exc)
        return None

if FORCE_OFFLINE:
    logger.info("Offline mode enforced via PHYLOS_OFFLINE flag.")

USE_GEMINI = bool(GEMINI_API_KEY) and not FORCE_OFFLINE

if USE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)
    _refresh_available_models()
    llm = _get_model_client(DEFAULT_CHAT_MODEL)
    try:
        origin_llm = genai.GenerativeModel(ORIGIN_INSIGHT_MODEL_NAME)
    except Exception as exc:
        logger.warning("Unable to initialize origin insight model (%s). Falling back to stub.", exc)
        origin_llm = None
else:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found - running in offline stub mode.")
    llm = None
    origin_llm = None

embedding_model = "models/embedding-001"
GRAPH_RECURSION_LIMIT = int(os.getenv("PHYLOS_RECURSION_LIMIT", "500"))
_executor = ThreadPoolExecutor(max_workers=1)
_gemini_embeddings_available = USE_GEMINI
_gemini_text_available = USE_GEMINI
_gemini_origin_available = USE_GEMINI
STUB_EMBED_DIM = 128

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _normalized_host(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host:
        return ""
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host

def _has_disallowed_extension(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in DISALLOWED_EXTENSIONS)

def _is_blocked_link(url: str) -> bool:
    host = _normalized_host(url)
    if not host:
        return False
    if _has_disallowed_extension(url):
        return True
    if any(host == suffix or host.endswith(f".{suffix}") for suffix in LINK_BLOCKLIST_SUFFIXES):
        return True
    if any(host == suffix or host.endswith(f".{suffix}") for suffix in SOCIAL_SHARE_HOSTS):
        parsed = urlparse(url)
        haystack = f"{parsed.path} {parsed.query}".lower()
        if any(keyword in haystack for keyword in SOCIAL_SHARE_KEYWORDS):
            return True
    return False

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
    if ECO_MODE or not _gemini_embeddings_available:
        return _stub_embedding(text)

    logger.debug("Embedding content (first 50 chars): '%s...'", text[:50])
    
    # Optimization: Truncate text to save tokens and avoid large payloads
    truncated_text = text[:2000]
    
    def _call():
        import time
        time.sleep(1.0) # Throttle requests to avoid rate limits
        return genai.embed_content(model=embedding_model, content=truncated_text, task_type="RETRIEVAL_DOCUMENT")

    try:
        embedding = _executor.submit(_call).result(timeout=REQUEST_TIMEOUT)
        return embedding['embedding']
    except (TimeoutError, Exception) as e:
        logger.warning("Embedding failed (%s). Falling back to offline vectors.", e)
        _gemini_embeddings_available = False
        return _stub_embedding(text)

def fetch_article_content(url: str) -> Dict[str, Any]:
    """Fetches an article body from the public web with a graceful offline fallback."""
    logger.info("Fetching content from URL: %s", url)

    def _stub():
        random_tail = uuid.uuid4()
        simulated = f"https://offline.phylos/simulated/{random_tail}"
        return {
            "id": url,
            "content": (
                f"This is simulated fallback content for {url}. "
                f"Real content could not be retrieved, but an offline reference is provided: {simulated}"
            ),
            "author": "Unknown",
            "timestamp": _now_iso(),
            "outbound_links": [simulated],
        }

    if FORCE_OFFLINE:
        return _stub()

    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch %s (%s). Skipping.", url, exc)
        return None

    content_type = (response.headers.get("Content-Type") or "").lower()
    allowed_types = ("text/html", "application/xhtml", "application/xml", "text/plain")
    if content_type and not any(t in content_type for t in allowed_types):
        logger.info("Skipping %s due to unsupported content type: %s", url, content_type)
        return None

    raw_html = response.text
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "header", "footer", "nav"]):
        tag.decompose()

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ]
    content = "\n\n".join(paragraphs) or soup.get_text(" ", strip=True)
    content = content.strip()
    if not content:
        logger.warning("No readable text extracted from %s. Skipping.", url)
        return None

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    def _meta_content(keys):
        for attr in ("name", "property"):
            for key in keys:
                tag = soup.find("meta", attrs={attr: key})
                if tag and tag.get("content"):
                    return tag["content"].strip()
        return None

    author = _meta_content(["author", "article:author", "og:author"]) or "Unknown"
    published = _meta_content(["article:published_time", "og:published_time"]) or response.headers.get("Date")
    if published:
        try:
            published = datetime.fromisoformat(published.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        except Exception:
            published = _now_iso()
    else:
        published = _now_iso()

    # Collect outbound links (anchors + canonical/meta references) for downstream extraction.
    outbound_links: List[str] = []
    seen_links: set[str] = set()

    def _append_link(href: str):
        if not href:
            return
        normalized = href.strip()
        if not normalized:
            return
        absolute = urljoin(url, normalized)
        if not absolute.startswith("http"):
            return
        if _is_blocked_link(absolute):
            return
        if absolute in seen_links:
            return
        seen_links.add(absolute)
        outbound_links.append(absolute)

    for anchor in soup.find_all("a", href=True):
        _append_link(anchor["href"])

    for link_tag in soup.find_all("link", href=True):
        rels = link_tag.get("rel") or []
        if isinstance(rels, str):
            rels = [rels]
        rel_set = {rel.lower() for rel in rels}
        if rel_set & {"canonical", "amphtml", "shortlink"}:
            _append_link(link_tag["href"])

    meta_url_keys = [
        "og:url",
        "og:see_also",
        "twitter:url",
        "article:publisher",
        "article:source",
        "citation_reference",
    ]
    for key in meta_url_keys:
        for attr in ("property", "name"):
            for tag in soup.find_all("meta", attrs={attr: key}):
                content = tag.get("content")
                if content and "http" in content:
                    _append_link(content)

    base_host = _normalized_host(url)
    def _discover_hidden_links():
        if not raw_html:
            return []
        discovered: List[str] = []
        for match in RAW_URL_PATTERN.finditer(raw_html):
            candidate = match.group(0).rstrip(').,;"')
            if not candidate.startswith("http"):
                continue
            if _is_blocked_link(candidate):
                continue
            host = _normalized_host(candidate)
            if not host or host == base_host:
                continue
            if candidate in seen_links:
                continue
            seen_links.add(candidate)
            discovered.append(candidate)
            if len(discovered) >= 24:
                break
        return discovered

    prioritized_links: List[str] = []
    if outbound_links:
        off_domain_links: List[str] = []
        on_domain_links: List[str] = []
        for link in outbound_links:
            host = _normalized_host(link)
            if host == base_host:
                on_domain_links.append(link)
            else:
                off_domain_links.append(link)
        prioritized_links = off_domain_links + on_domain_links

    if len([link for link in prioritized_links if _normalized_host(link) != base_host]) < 2:
        hidden = _discover_hidden_links()
        prioritized_links.extend(hidden)

    max_preview = int(os.getenv("PHYLOS_REFERENCE_PREVIEW_LIMIT", "40"))
    if prioritized_links and max_preview > 0:
        preview = "\n".join(prioritized_links[:max_preview])
        if preview:
            content = f"{content}\n\nReferenced URLs:\n{preview}"

    return {
        "id": url,
        "content": f"{title}\n\n{content}",
        "author": author,
        "timestamp": published,
        "outbound_links": prioritized_links,
    }

def extract_links(content: str, base_url: str) -> List[str]:
    """Extracts outbound links, preferring sources outside the current domain."""
    logger.debug("Extracting links from content.")

    raw_urls = RAW_URL_PATTERN.findall(content)
    seen: set[str] = set()
    cleaned: List[str] = []
    for raw in raw_urls:
        candidate = raw.rstrip(').,;"')
        if candidate in seen:
            continue
        if _is_blocked_link(candidate):
            continue
        seen.add(candidate)
        cleaned.append(candidate)

    base_host = urlparse(base_url).netloc
    off_domain: List[str] = []
    on_domain: List[str] = []
    for url in cleaned:
        host = urlparse(url).netloc
        if not host:
            continue
        if host == base_host:
            on_domain.append(url)
        else:
            off_domain.append(url)

    ordered = off_domain + on_domain
    return ordered[:4]

def _tokenize(text: str) -> set[str]:
    """Simple tokenizer that removes common stopwords."""
    if not text:
        return set()
    # Basic English stopwords
    STOPWORDS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you",
        "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one",
        "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    }
    words = re.findall(r'\b\w\w+\b', text.lower())
    return {w for w in words if w not in STOPWORDS}

def _jaccard_drift(text1: str, text2: str) -> float:
    """Calculates drift (1 - Jaccard Similarity) between two texts."""
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)
    if not tokens1 or not tokens2:
        return 1.0 # Max drift if one is empty
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    similarity = intersection / union if union else 0.0
    return 1.0 - similarity

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
    if ECO_MODE:
        return fallback

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

def generate_text_response(prompt: str, fallback: str, model_name: Optional[str] = None) -> str:
    """
    Generates a text response using the configured Gemini model.
    Allows overriding the default model via model_name.
    """
    if not USE_GEMINI:
        return fallback

    attempt_errors = []
    for candidate in _model_priority(model_name):
        client = _get_model_client(candidate)
        if not client:
            continue
        try:
            response = client.generate_content(prompt)
            if response and getattr(response, "text", None):
                return response.text
        except Exception as exc:
            attempt_errors.append((candidate, str(exc)))
            logger.warning("Gemini generation failed via %s: %s", candidate, exc)

    if attempt_errors:
        details = "; ".join(f"{name}: {err}" for name, err in attempt_errors)
        logger.error("All Gemini model attempts failed. Using fallback. Details: %s", details)
    return fallback

def generate_origin_insight(prompt: str, fallback: str) -> str:
    """Uses a higher-context Gemini model for origin-difference analysis."""
    global _gemini_origin_available
    if ECO_MODE or origin_llm is None or not _gemini_origin_available:
        return fallback

    def _call():
        return origin_llm.generate_content(prompt)

    try:
        response = _executor.submit(_call).result(timeout=REQUEST_TIMEOUT)
        return response.text
    except (TimeoutError, Exception) as e:
        logger.warning("Origin insight generation failed (%s). Falling back to stub.", e)
        _gemini_origin_available = False
        return fallback

def get_chat_model_options() -> List[Dict[str, str]]:
    """
    Returns the list of chat-capable Gemini models prioritized by the current configuration.
    Each entry includes an `id`, `label`, and `available` flag indicating whether the API
    confirmed access to that model (if discoverable).
    """
    options: List[Dict[str, str]] = []
    known = AVAILABLE_GEMINI_MODELS.copy()
    seen: set[str] = set()

    def _append(name: str):
        if not name or name in seen:
            return
        seen.add(name)
        label = known.get(name) or _prettify_model_label(name)
        options.append({
            "id": name,
            "label": label,
            "available": (not known) or (name in known),
        })

    for candidate in _model_priority():
        _append(candidate)

    if not options and known:
        for name, label in known.items():
            options.append({
                "id": name,
                "label": label,
                "available": True,
            })

    if not options:
        for candidate in _model_priority():
            _append(candidate)

    return options

def get_default_chat_model() -> str | None:
    """
    Returns the preferred chat model id after filtering out unavailable options.
    """
    options = get_chat_model_options()
    if not options:
        return DEFAULT_CHAT_MODEL
    for option in options:
        if option.get("available", True):
            return option["id"]
    return options[0]["id"]

# --- Graph Node Definitions ---

def node_acquire(state: "GraphState") -> Dict[str, Any]:
    """
    Acquires the next article from the traversal queue.
    """
    logger.info("Node: Acquire")
    queue = list(state["traversal_queue"])
    article_data = None
    url = parent_id = None
    depth = 0

    while queue:
        url, parent_id, depth = queue.pop(0)
        candidate = fetch_article_content(url)
        if candidate:
            article_data = candidate
            break
        logger.info("Skipping %s due to missing or unsupported content.", url)

    if not article_data:
        return {
            "traversal_queue": queue,
            "current_article": None,
            "parent_article_id": None,
            "current_depth": 0,
            "knowledge_graph": {"nodes": {}, "edges": []},
        }

    article_data["embedding"] = embedder(article_data["content"])
    article_data["depth"] = depth
    article_data.setdefault("outbound_links", [])

    visited = list(state.get("visited_urls", []))
    if url and url not in visited:
        visited.append(url)

    host_counts = dict(state.get("host_visit_counts", {}))
    if url:
        host = urlparse(url).netloc.lower()
        if host:
            host_counts[host] = host_counts.get(host, 0) + 1

    return {
        "traversal_queue": queue,
        "current_article": article_data,
        "parent_article_id": parent_id,
        "current_depth": depth,
        "knowledge_graph": {
            "nodes": {article_data["id"]: article_data},
            "edges": []
        },
        "visited_urls": visited,
        "host_visit_counts": host_counts,
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
        if ECO_MODE:
             # In Eco Mode, we don't have a global context embedding, so we assume 0 drift for the root
            drift_score = 0.0
        else:
            drift_score = calculate_semantic_drift(state["global_context"], current_article["embedding"])
        parent_id = "GLOBAL_CONTEXT"
    else:
        parent_article = knowledge_graph["nodes"].get(parent_id)
        if not parent_article:
            logger.error("Parent article %s not found in knowledge graph.", parent_id)
            return {}
        
        if ECO_MODE:
             # Use Jaccard Similarity for better semantic approximation
             drift_score = _jaccard_drift(parent_article["content"], current_article["content"])
             summary = _stub_summary(parent_article["content"], current_article["content"])
        else:
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
    if not current_article:
        logger.info("No current article available for branching.")
        return {}
    current_depth = 0 if not current_article else current_article.get("depth", 0)

    if current_depth >= state["max_depth"]:
        logger.info("Max depth (%s) reached. Halting branching.", state["max_depth"])
        return {}

    raw_links = current_article.get("outbound_links") or []
    if raw_links:
        new_links = raw_links[:4]  # limit breadth via prioritization
    else:
        new_links = extract_links(current_article["content"], current_article["id"])
    logger.info("Found %s new links to explore.", len(new_links))

    visited = set(state.get("visited_urls") or [])
    existing_queue = list(state["traversal_queue"])
    queued_urls = {item[0] for item in existing_queue}
    queued_host_counts: Dict[str, int] = {}
    for queued_url, _, _ in existing_queue:
        host = urlparse(queued_url).netloc.lower()
        if host:
            queued_host_counts[host] = queued_host_counts.get(host, 0) + 1

    host_counts = dict(state.get("host_visit_counts") or {})
    filtered_links: List[str] = []
    for link in new_links:
        host = urlparse(link).netloc.lower()
        if not host:
            continue
        if link in visited or link in queued_urls:
            logger.debug("Skipping already seen link: %s", link)
            continue
        total_host_visits = host_counts.get(host, 0) + queued_host_counts.get(host, 0)
        if HOST_VISIT_LIMIT > 0 and total_host_visits >= HOST_VISIT_LIMIT:
            logger.debug("Host %s reached visit limit (%s). Skipping %s.", host, HOST_VISIT_LIMIT, link)
            continue
        filtered_links.append(link)
        queued_urls.add(link)
        queued_host_counts[host] = queued_host_counts.get(host, 0) + 1

    if filtered_links:
        logger.info("Queued %s new unique links after filtering.", len(filtered_links))
    else:
        logger.info("No new unique links to queue from this article.")

    new_depth = current_depth + 1
    next_level_queue = [(link, current_article["id"], new_depth) for link in filtered_links]
    updated_queue = existing_queue + next_level_queue

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
