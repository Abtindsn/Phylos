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
from datetime import datetime, timezone, timedelta
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
    "schema.org",
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

if FORCE_OFFLINE:
    logger.info("Offline mode enforced via PHYLOS_OFFLINE flag.")

USE_GEMINI = bool(GEMINI_API_KEY) and not FORCE_OFFLINE

if USE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel('gemini-2.0-flash')
    try:
        origin_llm = genai.GenerativeModel(ORIGIN_INSIGHT_MODEL_NAME)
    except Exception:
        origin_llm = None
else:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found - running in offline stub mode.")
    llm = None
    origin_llm = None

    llm = None
    origin_llm = None

embedding_model = "models/text-embedding-004"
GRAPH_RECURSION_LIMIT = int(os.getenv("PHYLOS_RECURSION_LIMIT", "500"))
_executor = ThreadPoolExecutor(max_workers=1)
_gemini_embeddings_available = USE_GEMINI

# ---# Model fallback sequence when API calls fail
FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.5-pro",
]

def _call_with_fallback(operation_name: str, func, *args, **kwargs):
    """
    Executes a function with automatic model fallback for 429/404 errors.
    'func' should accept a 'model_name' keyword argument if it's a generation call.
    """
    if not USE_GEMINI:
        raise RuntimeError("Gemini API is disabled.")

    last_exception = None
    
    # Try the requested model first if specified, otherwise start with the list
    models_to_try = list(FALLBACK_MODELS)
    
    # If a specific model was requested in kwargs (e.g. for embedding), try that first
    # For generation, we iterate through models and pass the model name to the function
    
    for model_name in models_to_try:
        try:
            # If the function is 'genai.embed_content', we pass the model argument
            if func == genai.embed_content:
                 return func(model=embedding_model, *args, **kwargs)
            
            # For generation, we instantiate the model here or pass the name
            # Assuming 'func' is a wrapper that takes 'model_name'
            return func(model_name=model_name, *args, **kwargs)

        except Exception as e:
            error_str = str(e)
            is_quota = "429" in error_str or "Quota" in error_str
            is_not_found = "404" in error_str or "not found" in error_str
            
            if is_quota or is_not_found:
                logger.warning(f"{operation_name} failed with {model_name}: {e}. Retrying with next model...")
                last_exception = e
                continue
            else:
                # If it's another error (e.g. 500, invalid argument), raise immediately
                raise e
                
    logger.error(f"{operation_name} failed on all models. Last error: {last_exception}")
    raise last_exception
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

def _are_domains_related(url1: str, url2: str) -> bool:
    """
    Checks if two URLs belong to related domains (e.g., bbc.com and bbc.co.uk).
    """
    host1 = _normalized_host(url1)
    host2 = _normalized_host(url2)
    
    if host1 == host2:
        return True
        
    # Remove TLDs to compare base names
    # Simple heuristic: split by dot and compare the second to last part if length > 2
    # or just check if one is a substring of another for now, but that's risky.
    # Better approach: check for common known related domains or shared root.
    
    # Heuristic 1: Shared root domain (e.g. cnbc.com, cnbc.eu - though cnbc.eu isn't common, but you get the idea)
    # Actually, let's look at the prompt examples: bbc.com and bbc.co.uk.
    # They share 'bbc'.
    
    parts1 = host1.split('.')
    parts2 = host2.split('.')
    
    # Filter out common TLDs/SLDs for comparison
    common_suffixes = {'com', 'co', 'uk', 'org', 'net', 'gov', 'edu', 'io', 'ai', 'app', 'dev'}
    
    root1 = [p for p in parts1 if p not in common_suffixes]
    root2 = [p for p in parts2 if p not in common_suffixes]
    
    # If they share a significant root part
    if set(root1) & set(root2):
        return True
        
    return False

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

def _has_future_date(url: str) -> bool:
    """Checks if a URL contains a date string that is in the future."""
    # Regex to find YYYY/MM/DD or YYYY-MM-DD patterns
    match = re.search(r'(\d{4})[/-](\d{2})[/-](\d{2})', url)
    if not match:
        return False
    
    year, month, day = map(int, match.groups())
    
    try:
        # Check for plausible dates
        if not (1990 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
            return False
        
        url_date = datetime(year, month, day, tzinfo=timezone.utc)
        # If the date is more than a day in the future, flag it.
        if url_date > datetime.now(timezone.utc) + timedelta(days=1):
            return True
    except ValueError:
        return False # Invalid date like 2025/02/30
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
        f"[Non-AI Analysis] Offline summary: similarity {overlap:.2f}. "
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
        return _call_with_fallback("Embedding", genai.embed_content, content=truncated_text, task_type="RETRIEVAL_DOCUMENT")

    try:
        embedding = _executor.submit(_call).result(timeout=REQUEST_TIMEOUT)
        return embedding['embedding']
    except (TimeoutError, Exception) as e:
        logger.warning("Embedding failed (%s). Falling back to offline vectors.", e)
        _gemini_embeddings_available = False
        return _stub_embedding(text)

def fetch_article_content(url: str) -> tuple[dict[str, Any], str] | None:
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
        }, ""

    if FORCE_OFFLINE:
        return _stub()

    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch %s (%s). Skipping.", url, exc)
        return None, ""

    content_type = (response.headers.get("Content-Type") or "").lower()
    allowed_types = ("text/html", "application/xhtml", "application/xml", "text/plain")
    if content_type and not any(t in content_type for t in allowed_types):
        logger.info("Skipping %s due to unsupported content type: %s", url, content_type)
        return None, ""

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
        return None, ""

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

    return ({
        "id": url,
        "content": f"{title}\n\n{content}",
        "author": author,
        "timestamp": published,
        "outbound_links": prioritized_links,
    }, raw_html)

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
    if vec1.shape != vec2.shape:
        logger.warning("Embedding dimension mismatch: %s vs %s. Returning default drift.", vec1.shape, vec2.shape)
        return 1.0 # Assume high drift if we can't compare

    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # We want drift, which is 1 - similarity
    drift = 1.0 - similarity
    logger.debug("Calculated semantic drift: %s", drift)
    return drift

def summarize_mutation(parent_content: str, child_content: str, parent_url: str = "", child_url: str = "", domains_related: bool = False, drift_score: float = 0.0) -> str:
    """Uses the LLM to generate a summary of the semantic difference."""
    fallback = _stub_summary(parent_content, child_content)
    if ECO_MODE:
        return fallback

    if drift_score > 0.9:
        prompt = f"""
        PERFORM A DEEP NARRATIVE ANALYSIS of the divergence between these two texts.
        The semantic drift score is {drift_score:.2f} (High Anomaly).
        
        PARENT TEXT (Source: {parent_url}):
        ---
        {parent_content[:3000]}
        ---

        CHILD TEXT (Source: {child_url}):
        ---
        {child_content[:3000]}
        ---

        Your task is to explain WHY the mutation score is so high.
        
        Structure your response exactly as follows:
        
        **Analysis:**
        *   **Parent Focus:** [Concise summary of the parent's main topic/angle]
        *   **Child Focus:** [Concise summary of the child's main topic/angle]
        
        **Conclusion:**
        [Explain the specific nature of the divergence. Is it a complete topic shift? A different entity being discussed? A "hallucinated" link? Be specific.]
        """
    else:
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
        Avoid generic phrases like "The text discusses". Be specific about names, events, and details.
        
        CONTEXT:
        Parent URL: {parent_url}
        Child URL: {child_url}
        Are domains related: {domains_related}
        
        If the domains are related but the content is very different, explain WHY (e.g., regional variation, updated article, different topic).
        """

    return generate_text_response(prompt, fallback)

def generate_text_response(prompt: str, fallback: str, model_name: str = None) -> str:
    """
    Generates a text response using the configured Gemini model.
    Allows overriding the default model via model_name.
    """
    if not USE_GEMINI:
        return fallback

    def _generate(model_name=None):
        # Use the requested model if provided, otherwise use the default llm
        active_model = genai.GenerativeModel(model_name) if model_name else llm
        return active_model.generate_content(prompt)

    try:
        # We don't pass model_name in kwargs here because _call_with_fallback will pass it as a kwarg to _generate
        # when it iterates. However, if we want to start with a specific model, we should pass it.
        # But _call_with_fallback logic says: "If a specific model was requested in kwargs... try that first".
        # The issue is that _call_with_fallback calls func(model_name=model_name, *args, **kwargs).
        # If kwargs ALREADY contains model_name, it crashes.
        
        # Correct usage: Pass model_name only if we want to override the start. 
        # But _call_with_fallback iterates anyway.
        # Let's just NOT pass model_name in kwargs to _call_with_fallback for _generate, 
        # because _generate signature is def _generate(model_name=None).
        
        # Wait, if I pass model_name to _call_with_fallback(..., model_name=foo), 
        # then inside _call_with_fallback, kwargs has {'model_name': foo}.
        # Then it calls func(model_name=current_model, **kwargs).
        # So it calls func(model_name=current_model, model_name=foo) -> TypeError!
        
        # Fix: Remove model_name from kwargs before calling func inside _call_with_fallback?
        # Or just don't pass it here if we want fallback logic to handle it.
        # If we want to force a start model, we should handle it differently.
        # For now, let's just NOT pass it in kwargs here, and let fallback logic iterate.
        response = _call_with_fallback("Text Generation", _generate)
        if response.text:
            return response.text
        return fallback
    except Exception as e:
        logger.error("Gemini generation failed: %s", e)
        # Always return fallback (which is unique) on error to avoid repetitive error messages
        return fallback

def generate_origin_insight(prompt: str, fallback: str) -> str:
    """Uses a higher-context Gemini model for origin-difference analysis."""
    global _gemini_origin_available
    if ECO_MODE or not _gemini_origin_available:
        return fallback

    def _call(model_name=None):
        # Use the passed model_name from fallback logic, or default to ORIGIN_INSIGHT_MODEL_NAME
        active_model = genai.GenerativeModel(model_name or ORIGIN_INSIGHT_MODEL_NAME)
        return active_model.generate_content(prompt)

    try:
        response = _executor.submit(_call_with_fallback, "Origin Insight", _call).result(timeout=REQUEST_TIMEOUT)
        return response.text
    except (TimeoutError, Exception) as e:
        logger.warning("Origin insight generation failed (%s). Falling back to stub.", e)
        _gemini_origin_available = False
        return fallback


# --- Graph Node Definitions ---

def _add_raw_html_to_article(article_data: Dict[str, Any], raw_html: str) -> Dict[str, Any]:
    """Helper to attach raw HTML for later analysis without bloating logs."""
    if article_data:
        article_data["raw_html"] = raw_html
    return article_data

def node_acquire(state: "GraphState") -> Dict[str, Any]:
    """
    Acquires the next article from the traversal queue.
    """
    logger.info("Node: Acquire")
    queue = list(state["traversal_queue"])
    article_data = None
    url = parent_id = None
    raw_html_content = ""
    depth = 0

    while queue:
        url, parent_id, depth = queue.pop(0)
        response_tuple = fetch_article_content(url)
        candidate, raw_html_content = response_tuple if response_tuple else (None, "")
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
    article_data = _add_raw_html_to_article(article_data, raw_html_content)
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
        # "Patient Zero" is the original source - no mutation by definition
        # It should always be considered authentic (green) with 0.0 drift
        drift_score = 0.0
        parent_id = "GLOBAL_CONTEXT"
        divergence_reason = "Patient Zero (Root)"
        domains_related = True # Self-referential effectively
    else:
        parent_article = knowledge_graph["nodes"].get(parent_id)
        if not parent_article:
            logger.error("Parent article %s not found in knowledge graph.", parent_id)
            return {}
        
        
        domains_related = _are_domains_related(parent_article["id"], current_article["id"])
        divergence_reason = "Standard Drift"

        if ECO_MODE:
             # Use Jaccard Similarity for better semantic approximation
             drift_score = _jaccard_drift(parent_article["content"], current_article["content"])
             summary = _stub_summary(parent_article["content"], current_article["content"])
             divergence_reason = "Jaccard Drift (Eco Mode)"
        else:
            drift_score = calculate_semantic_drift(parent_article["embedding"], current_article["embedding"])
            
            # --- Refined Mutation Logic ---
            
            if drift_score > 0.90:
                if domains_related:
                    # High drift but related domains -> Likely a regional variation or homepage vs article
                    # We soften the score slightly to avoid breaking the trace, but flag it.
                    logger.info("High drift (%.2f) between related domains (%s -> %s). Suspected Regional/Format Variation.", 
                                drift_score, parent_article["id"], current_article["id"])
                    divergence_reason = "Regional/Format Variation (Related Sources)"
                    # Optional: Soften score if we want to treat it as less severe?
                    # For now, we keep the score but change the reason.
                else:
                    divergence_reason = "High Mutation Hotspot (Unrelated Sources)"
            elif domains_related and drift_score > 0.5:
                 divergence_reason = "Routine Update/Variation (Related Sources)"

            summary = summarize_mutation(parent_article["content"], current_article["content"], 
                                         parent_url=parent_article["id"], child_url=current_article["id"],
                                         domains_related=domains_related, drift_score=drift_score)

    # Define mutation threshold
    MUTATION_THRESHOLD = 0.3 # Adjusted for real embeddings

    relation_type = "Mutation" if drift_score > MUTATION_THRESHOLD else "Replication"
    
    # Override relation type for specific cases
    if drift_score > 0.90:
        relation_type = "Anomaly" if not domains_related else "Major Variation"

    logger.debug("Relation to parent: %s (Score: %s, Reason: %s)", relation_type, drift_score, divergence_reason)

    new_edge = {
        "source": parent_id,
        "target": current_article["id"],
        "attributes": {
            "mutation_score": drift_score,
            "relation_type": relation_type,
            "divergence_reason": divergence_reason,
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
    warnings = list(state.get("data_warnings") or [])
    raw_html = current_article.get("raw_html", "")
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
        if _has_future_date(link):
            # --- Enhanced Anomaly Investigation ---
            context_snippet = "Context not available."
            anchor_text = "N/A"
            if raw_html:
                soup = BeautifulSoup(raw_html, "html.parser")
                anchor_tag = soup.find("a", href=re.compile(re.escape(link.split('?')[0])))
                if anchor_tag:
                    anchor_text = anchor_tag.get_text(strip=True)
                    # Get the parent tag's text to provide context
                    parent_context = anchor_tag.parent.get_text(" ", strip=True) if anchor_tag.parent else ""
                    context_snippet = parent_context[:200]
            
            logger.warning("Skipping future-dated link: %s (Anchor: '%s')", link, anchor_text)
            warnings.append(f"Future-Dated URL Skipped: Found in '{current_article['id']}'. URL: '{link}'. Anchor Text: '{anchor_text}'. Context: '{context_snippet}...'. This could be a placeholder, typo, or pre-publication link.")
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
        "data_warnings": warnings,
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


try:
    graph_image = app.get_graph().draw_mermaid_png()
    with open("/home/abtin/Phylos/graph_visualization.png", "wb") as f:
        f.write(graph_image)
    logger.info("Graph visualization saved to graph_visualization.png")
except Exception as e:
    logger.warning("Could not draw graph visualization: %s", e)

# --- Model Listing Functions ---

def get_chat_model_options() -> List[Dict[str, Any]]:
    """
    Returns a list of available Gemini models for chat.
    Fetches from API if possible, otherwise returns a curated fallback list.
    """
    # Curated list of known working Gemini models as of 2025
    FALLBACK_MODELS = [
        {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
        {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
        {"id": "gemini-2.0-flash", "label": "Gemini 2.0 Flash"},
        {"id": "gemini-2.0-flash-exp", "label": "Gemini 2.0 Flash (Experimental)"},
    ]
    
    if not USE_GEMINI:
        return FALLBACK_MODELS
    
    try:
        # Try to fetch available models from the API
        models = genai.list_models()
        chat_models = []
        
        for model in models:
            # Filter for models that support generateContent
            if 'generateContent' in model.supported_generation_methods:
                model_id = model.name.replace('models/', '')
                # Create a human-readable label
                label = model_id.replace('-', ' ').title()
                chat_models.append({
                    "id": model_id,
                    "label": label,
                })
        
        # Sort models by ID (newest versions first)
        import re
        def model_sort_key(model):
            """Sort by version number (descending), then alphabetically"""
            model_id = model["id"]
            # Extract version numbers like 2.5, 2.0, 1.5, 1.0
            version_match = re.search(r'(\d+)\.(\d+)', model_id)
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2))
                # Return negative to sort descending
                return (-major, -minor, model_id)
            return (0, 0, model_id)
        
        chat_models.sort(key=model_sort_key)
        
        # If we successfully got models from the API, return them
        if chat_models:
            logger.info("Fetched %d chat models from Gemini API", len(chat_models))
            return chat_models
        else:
            logger.warning("No chat models found from API, using fallback list")
            return FALLBACK_MODELS
            
    except Exception as e:
        logger.warning("Failed to fetch models from API (%s), using fallback list", e)
        return FALLBACK_MODELS

def get_default_chat_model() -> str:
    """Returns the default chat model ID."""
    # Prefer gemini-2.5-flash if available, otherwise try to use the main LLM's model
    if "gemini-2.5-flash" in [m["id"] for m in get_chat_model_options()]:
        return "gemini-2.5-flash"
    elif USE_GEMINI and llm:
        try:
            # The model name might be in the format 'models/gemini-1.5-flash-latest'
            model_name = getattr(llm, '_model_name', 'gemini-1.5-flash-latest')
            return model_name.replace('models/', '')
        except Exception:
            pass
    
    # Fallback to a reliable default
    return "gemini-2.5-flash"
