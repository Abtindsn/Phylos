
# FastAPI server for the Narrative DNA Sequencer.
# Provides a WebSocket endpoint so clients can stream traversal events.

import os
import uuid
from typing import Dict, Any, List
import json
import logging
from pathlib import Path
import difflib
import re
from urllib.parse import urlparse
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("PHYLOS_LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("PHYLOS_LOG_FILE")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
handlers = [logging.StreamHandler()]
if LOG_FILE:
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
        except (FileNotFoundError, OSError):
            pass
    handlers.append(logging.FileHandler(LOG_FILE))

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=handlers)
logger = logging.getLogger("phylos.server")
BASE_DIR = Path(__file__).resolve().parent
REACT_HTML = (BASE_DIR / "frontend" / "chronicle.html").read_text()

# --- Helpers ---
MAX_LIST_ITEMS = int(os.getenv("PHYLOS_EVENT_MAX_ITEMS", "6"))
MAX_STRING_LENGTH = int(os.getenv("PHYLOS_EVENT_MAX_STRING", "160"))
MAX_RECURSION_DEPTH = int(os.getenv("PHYLOS_EVENT_MAX_DEPTH", "3"))

def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"

def _brief_article(article: dict | None) -> dict | None:
    if not isinstance(article, dict):
        return article
    return {
        "id": article.get("id"),
        "depth": article.get("depth"),
        "timestamp": article.get("timestamp"),
        "author": article.get("author"),
        "content_preview": _shorten(article.get("content", ""), MAX_STRING_LENGTH),
    }

def _short_host(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url).netloc
    return host or url

def _strip_references(text: str) -> str:
    if not text:
        return ""
    parts = text.split("Referenced URLs:")
    return parts[0].strip()

def _parse_insight_json(raw: str) -> dict | None:
    """Try to parse Gemini output into JSON even if wrapped."""
    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None

def _summarize_event_data(data, edge_updates=None):
    if not isinstance(data, dict):
        return data
    summary = dict(data)
    if "knowledge_graph" in summary and isinstance(summary["knowledge_graph"], dict):
        kg = summary["knowledge_graph"]
        nodes = kg.get("nodes") or {}
        edges = kg.get("edges") or []
        summary["knowledge_graph"] = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "sample_nodes": list(nodes.keys())[:MAX_LIST_ITEMS],
            "sample_edges": [
                {
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "attributes": {
                        "relation_type": edge.get("attributes", {}).get("relation_type"),
                        "mutation_score": edge.get("attributes", {}).get("mutation_score"),
                        "summary_preview": _shorten(edge.get("attributes", {}).get("summary", ""), MAX_STRING_LENGTH // 2),
                    }
                }
                for edge in edges[:MAX_LIST_ITEMS]
            ],
        }
    if "current_article" in summary:
        summary["current_article"] = _brief_article(summary["current_article"])
    if "knowledge_graph" in summary and isinstance(summary["knowledge_graph"], dict):
        kg = summary["knowledge_graph"]
        if "sample_nodes" in kg:
            kg["sample_nodes"] = kg["sample_nodes"][:MAX_LIST_ITEMS]
    if "global_context" in summary and isinstance(summary["global_context"], list):
        summary["global_context"] = {
            "length": len(summary["global_context"]),
            "preview": summary["global_context"][:MAX_LIST_ITEMS],
        }
    if "traversal_queue" in summary and isinstance(summary["traversal_queue"], list):
        summary["traversal_queue"] = summary["traversal_queue"][:MAX_LIST_ITEMS]

    clean_updates = []
    mutation_count = 0
    replication_count = 0
    edge_updates = edge_updates or []
    for edge in edge_updates[:MAX_LIST_ITEMS]:
        attrs = edge.get("attributes") or {}
        relation = attrs.get("relation_type")
        if relation == "Mutation":
            mutation_count += 1
        elif relation == "Replication":
            replication_count += 1
        clean_updates.append({
            "source": edge.get("source"),
            "target": edge.get("target"),
            "attributes": {
                "relation_type": relation,
                "mutation_score": attrs.get("mutation_score"),
                "summary_preview": _shorten(attrs.get("summary", ""), MAX_STRING_LENGTH // 2),
            }
        })
    if clean_updates:
        summary["edge_updates"] = clean_updates
    if mutation_count or replication_count:
        summary["edge_stats"] = {
            "mutation_count": mutation_count,
            "replication_count": replication_count,
        }

    return summary

def _sanitize_payload(value, depth=0):
    """Trim large numeric arrays and long strings before streaming to the UI."""
    if depth > MAX_RECURSION_DEPTH:
        return "<truncated>"

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, str):
        return value if len(value) <= MAX_STRING_LENGTH else value[:MAX_STRING_LENGTH] + "…"

    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, (int, float)) for item in value):
            if len(value) > MAX_LIST_ITEMS:
                return value[:MAX_LIST_ITEMS] + [f"...({len(value) - MAX_LIST_ITEMS} more)"]
            return value
        trimmed = [_sanitize_payload(item, depth + 1) for item in value[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            trimmed.append(f"...({len(value) - MAX_LIST_ITEMS} more)")
        return trimmed

    if isinstance(value, dict):
        return {key: _sanitize_payload(val, depth + 1) for key, val in value.items()}

    return str(value)

class ChatRequest(BaseModel):
    session_id: str
    message: str

def _create_session() -> tuple[str, Dict[str, Any]]:
    session_id = str(uuid.uuid4())
    SESSION_CONTEXTS[session_id] = {
        "knowledge_graph": {"nodes": {}, "edges": []},
        "summary": None,
        "history": [],
        "investigation": [],
        "investigation_map": {},
    }
    return session_id, SESSION_CONTEXTS[session_id]

def _gather_graphs(obj, acc, depth=0, max_depth=4):
    if depth > max_depth or obj is None:
        return
    if isinstance(obj, dict):
        if "nodes" in obj and "edges" in obj:
            acc.append(obj)
        for value in obj.values():
            _gather_graphs(value, acc, depth + 1, max_depth)
    elif isinstance(obj, list):
        for item in obj:
            _gather_graphs(item, acc, depth + 1, max_depth)

def _accumulate_graph(accumulator: Dict[str, Any], data: Dict[str, Any] | None):
    if not data:
        return
    graphs: List[Dict[str, Any]] = []
    _gather_graphs(data, graphs)
    for graph in graphs:
        nodes = graph.get("nodes") or {}
        if isinstance(nodes, dict):
            accumulator["nodes"].update(nodes)
        edges = graph.get("edges") or []
        if isinstance(edges, list):
            accumulator["edges"].extend(edges)

def _generate_graph_summary(graph: Dict[str, Any]) -> str:
    node_count = len(graph.get("nodes", {}))
    edge_count = len(graph.get("edges", []))
    sample_edges = [
        {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "relation": edge.get("attributes", {}).get("relation_type"),
            "score": edge.get("attributes", {}).get("mutation_score"),
        }
        for edge in graph.get("edges", [])[:5]
    ]
    edge_lines = []
    for edge in sample_edges:
        rel = edge["relation"] or "Unknown"
        score = edge["score"]
        try:
            score_text = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_text = "-"
        edge_lines.append(f"{edge['source']} -> {edge['target']} ({rel}, score {score_text})")
    edge_snippet = "; ".join(edge_lines) if edge_lines else "No mutation edges captured."
    fallback = (
        f"The trace captured {node_count} unique sources tied together by {edge_count} narrative relationships. "
        f"Mutation hotspots observed along: {edge_snippet}. Use these as jumping-off points for deeper review."
    )
    prompt = f"""
    You are Gemini. Provide a concise but insightful summary of a narrative trace.
    Node count: {node_count}
    Edge count: {edge_count}
    Representative edges (source -> target, relation, score):
    {json.dumps(sample_edges, indent=2)}

    Describe overall findings in 2 short paragraphs and highlight potential mutation hotspots.
    """
    return generate_text_response(prompt, fallback)

def _generate_chat_reply(summary: str, history: List[Dict[str, str]], question: str) -> str:
    summary_preview = _shorten(summary, 220)
    fallback = (
        f"Here's what the trace uncovered: {summary_preview} "
        f"In response to your question \"{question}\", focus on the mutations mentioned above—they mark where the narrative diverges most. "
        "Trace those sources to validate their claims or find corroborating evidence."
    )
    history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:])
    prompt = f"""
    You are Gemini acting as an investigative analyst. Use the summary below to answer follow-up questions.
    SUMMARY:
    {summary}

    CHAT HISTORY:
    {history_text}

    USER QUESTION:
    {question}

    Respond concisely (<=120 words) and suggest next investigative steps when useful.
    """
    return generate_text_response(prompt, fallback)

def _collect_edges_from_section(section: Any) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    def traverse(obj: Any):
        if isinstance(obj, dict):
            kg = obj.get("knowledge_graph")
            if isinstance(kg, dict):
                edge_list = kg.get("edges")
                if isinstance(edge_list, list) and edge_list:
                    edges.extend(edge_list)
            for key, value in obj.items():
                if key == "knowledge_graph":
                    continue
                traverse(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item)
    traverse(section)
    return edges

def _extract_edge_updates(raw_data: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_data, dict):
        return []
    updates: List[Dict[str, Any]] = []
    for key in ("chunk", "output"):
        section = raw_data.get(key)
        if section:
            updates.extend(_collect_edges_from_section(section))
    return updates

def _clean_score(value):
    try:
        return float(value)
    except Exception:
        return value

def _build_edge_snapshots(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for edge in edges or []:
        attr = edge.get("attributes", {})
        cleaned.append({
            "source": edge.get("source"),
            "target": edge.get("target"),
            "attributes": {
                "relation_type": attr.get("relation_type"),
                "mutation_score": _clean_score(attr.get("mutation_score")),
                "summary": attr.get("summary"),
            }
        })
    return cleaned

def _origin_similarity(origin_embedding, target_embedding):
    if not origin_embedding or not target_embedding:
        return None
    try:
        drift = calculate_semantic_drift(origin_embedding, target_embedding)
        similarity = 1.0 - drift
        return max(0.0, min(1.0, similarity))
    except Exception:
        return None

def _strip_references(text: str) -> str:
    if not text:
        return ""
    parts = text.split("Referenced URLs:")
    return parts[0].strip()

def _first_unique_sentence(origin_text: str, node_text: str) -> str:
    sentence_splitter = r'(?<=[.!?])\s+'
    origin_sentences = [s.strip() for s in re.split(sentence_splitter, origin_text) if s.strip()]
    node_sentences = [s.strip() for s in re.split(sentence_splitter, node_text) if s.strip()]
    origin_norm = {re.sub(r'\W+', '', s).lower() for s in origin_sentences}
    for sentence in node_sentences:
        normalized = re.sub(r'\W+', '', sentence).lower()
        if normalized and normalized not in origin_norm:
            return sentence
    return node_sentences[0] if node_sentences else node_text[:200]

def _fallback_origin_summary(origin_text: str, node_text: str, article_title: str | None = None) -> tuple[str, str | None]:
    clean_origin = _strip_references(origin_text)
    clean_node = _strip_references(node_text)
    matcher = difflib.SequenceMatcher(None, clean_origin, clean_node)
    overlap = matcher.quick_ratio()
    origin_focus = clean_origin[:220].strip().replace("\n", " ")
    unique_sentence = _first_unique_sentence(clean_origin, clean_node)
    changed_focus = unique_sentence.strip().replace("\n", " ")

    if overlap > 0.85:
        delta = "The later article largely mirrors the original but reiterates it for a new audience."
    elif overlap > 0.6:
        delta = "It keeps most of the framing but adds fresh context or emphasis in the highlighted excerpt."
    else:
        delta = "It meaningfully reframes the story and draws attention to new angles in the highlighted excerpt."

    article_label = article_title or "Follow-on article"
    summary = (
        f"{article_label}: Original focus centered on {origin_focus}... "
        f"This piece highlights {changed_focus}... "
        f"Net effect: {delta}"
    )
    hidden = (
        f"Hidden investigation: compared normalized sentences; first unique sentence in article "
        f"was '{changed_focus}'. Overlap score {overlap:.2f}."
    )
    return summary, hidden

def _summarize_origin_difference(origin_text: str, node_text: str, article_title: str | None = None) -> tuple[str, str | None]:
    """
    Ask Gemini (via generate_text_response) for a JSON payload describing the
    public summary + hidden investigation for ORIGINAL vs FOLLOW-UP article.
    Falls back to our deterministic diff summarizer if anything goes wrong.
    """
    if not origin_text or not node_text:
        return "", None

    # Deterministic fallback ensures we always have a reasonable baseline
    fallback_summary, fallback_hidden = _fallback_origin_summary(origin_text, node_text, article_title)
    fallback_payload = json.dumps(
        {
            "summary": fallback_summary,
            "investigation": fallback_hidden or "",
        },
        ensure_ascii=False,
    )

    prompt = (
        "Primary Objective: You are an elite Signal Investigator performing a forensic, differential analysis "
        "between the ORIGINAL STORY and the FOLLOW-UP ARTICLE shown below. Your task is to isolate verifiable Signal "
        "(facts, causal intent) from Noise (editorial tone, omissions) and reconstruct how and why the narrative evolved.\n\n"
        "You MUST respond STRICTLY in compact JSON with two string fields only:\n"
        "{\"summary\": \"public-facing insight\", \"investigation\": \"private chain-of-investigation\"}.\n\n"
        "SUMMARY (2-3 sentences) must:\n"
        " • Describe the Core Power (facts shared across both pieces) and highlight the most significant Δ (difference).\n"
        " • State whether the follow-up reinforces, contradicts, or reframes the original, referencing the source host.\n"
        f" • Explicitly mention the follow-up article id or title: {(article_title or 'Unknown')!r}.\n"
        "INVESTIGATION (3-4 sentences, hidden) must follow the First-Principles playbook:\n"
        " • Evidence Set & Source Profiles (identify URLs + timestamps, biases).\n"
        " • Commonality Grid (Shared facts) and Discrepancy Matrix items labeled Δ_F, Δ_I, Δ_P.\n"
        " • Hypothesis testing: propose at least two motives for the change, then argue which is superior using the evidence.\n"
        " • Explicitly define Signal vs. Noise in your reasoning and reference the original URLs when citing evidence.\n"
        "No steps may be skipped.\n\n"
        f"ARTICLE TITLE: {article_title or 'Unknown'}\n"
        "ORIGINAL STORY:\n"
        f"{_strip_references(origin_text)[:3000]}\n\n"
        "FOLLOW-UP ARTICLE:\n"
        f"{_strip_references(node_text)[:3000]}"
    )

    try:
        raw = generate_text_response(prompt, fallback_payload)
    except Exception as exc:
        logger.warning("generate_text_response failed for origin insight: %s", exc)
        return fallback_summary, fallback_hidden

    summary = fallback_summary
    hidden = fallback_hidden

    try:
        data = json.loads(raw)
    except Exception:
        data = None

    if isinstance(data, dict):
        summary_candidate = (data.get("summary") or "").strip()
        investigation_candidate = (
            data.get("investigation")
            or data.get("analysis")
            or data.get("reason")
            or ""
        )
        investigation_candidate = investigation_candidate.strip()

        if summary_candidate:
            summary = summary_candidate
        if investigation_candidate:
            hidden = investigation_candidate
    elif data is not None:
        text = str(data).strip()
        if text:
            summary = text
    else:
        raw_text = (raw or "").strip()
        if raw_text:
            summary = raw_text

    if not hidden:
        hidden = fallback_hidden
    if hidden and summary and hidden.strip() == summary.strip():
        hidden = fallback_hidden
    return summary, hidden

def _build_node_snapshots(
    graph: Dict[str, Any],
    origin_embedding: List[float] | None = None,
    origin_content: str | None = None,
    origin_url: str | None = None,
) -> List[Dict[str, Any]]:
    edges = graph.get("edges") or []
    score_map: Dict[str, float] = {}
    for edge in edges:
        target = edge.get("target")
        if target:
            score_map[target] = _clean_score(edge.get("attributes", {}).get("mutation_score")) or 0.0

    nodes = []
    for node_id, payload in (graph.get("nodes") or {}).items():
        resolved_id = payload.get("id", node_id)
        similarity = _origin_similarity(origin_embedding, payload.get("embedding"))
        difference = 1.0 - similarity if similarity is not None else None
        nodes.append({
            "id": resolved_id,
            "content": payload.get("content", ""),
            "author": payload.get("author"),
            "timestamp": payload.get("timestamp"),
            "depth": payload.get("depth", 0),
            "mutation_score": score_map.get(resolved_id, score_map.get(node_id, 0.0)),
            "outbound_links": payload.get("outbound_links") or [],
            "origin_similarity": similarity,
            "origin_difference": difference,
            "origin_summary": None,
            "origin_hidden_reason": None,
        })

    origin_text = (origin_content or "").strip()
    if origin_text:
        for entry in nodes:
            if not entry.get("content"):
                continue
            if origin_url and entry["id"] == origin_url:
                entry["origin_summary"] = "Reference article supplied as the starting point."
                entry["origin_hidden_reason"] = "Origin node – investigation not needed."
                continue
            try:
                article_title = entry.get("title") or entry.get("id")
                summary, hidden = _summarize_origin_difference(origin_text, entry["content"], article_title)
            except Exception:
                summary, hidden = _fallback_origin_summary(origin_text, entry["content"], entry.get("title"))
            entry["origin_summary"] = summary
            entry["origin_hidden_reason"] = hidden
    return nodes

def _build_investigation_entry(origin: dict | None, article: dict | None, rapid: bool = False) -> dict | None:
    if not origin or not article:
        return None
    origin_content = origin.get("content")
    if not origin_content:
        return None
    article_content = article.get("content")
    if not article_content:
        return None
    title = (article_content.split("\n", 1)[0] or article.get("id") or "").strip()
    if rapid:
        summary, hidden = _fallback_origin_summary(origin_content, article_content, title)
    else:
        summary, hidden = _summarize_origin_difference(origin_content, article_content, title)
    if not hidden or hidden.strip() == summary.strip():
        host = _short_host(article.get("id")) or "source"
        hidden = f"Internal investigation note: awaiting deeper comparison for {host}."
    entry = {
        "id": article.get("id"),
        "title": title or article.get("id"),
        "timestamp": article.get("timestamp"),
        "summary": summary,
        "reason": hidden or summary,
        "investigation": hidden or summary,
        "url": article.get("id"),
    }
    return entry

def _build_investigation_timeline(nodes: list[dict], session_context: dict) -> list[dict]:
    investigation_map = session_context.get("investigation_map") or {}
    timeline = session_context.setdefault("investigation", [])
    # ensure we have entries for nodes missing them
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        if node_id not in investigation_map:
            reason = node.get("origin_hidden_reason")
            if not reason:
                continue
            entry = {
                "id": node_id,
                "title": node.get("title") or node_id,
                "timestamp": node.get("timestamp"),
                "summary": node.get("origin_summary"),
                "reason": reason,
                "url": node.get("id"),
            }
            investigation_map[node_id] = entry
            timeline.append(entry)
    return sorted(
        [
        {
            "id": entry.get("id"),
            "title": entry.get("title"),
            "timestamp": entry.get("timestamp"),
            "summary": entry.get("summary"),
            "reason": entry.get("reason"),
            "investigation": entry.get("investigation", entry.get("reason")),
            "url": entry.get("url"),
        }
        for entry in timeline
    ],
    key=lambda item: (item.get("timestamp") or "", item.get("title") or "")
    )

# --- Local Imports ---
from state import GraphState, InitialArticleRequest
from graph_builder import (
    app,
    embedder,
    fetch_article_content,
    GRAPH_RECURSION_LIMIT,
    generate_text_response,
    generate_origin_insight,
    calculate_semantic_drift,
)

# --- Session Storage ---
SESSION_CONTEXTS: Dict[str, Dict[str, Any]] = {}

# --- FastAPI App Initialization ---
api = FastAPI(
    title="Narrative DNA Sequencer",
    description="A backend for tracing semantic mutations in narrative networks.",
    version="0.1.0",
)

# Single-page console (classic) to interact with the agent
CLASSIC_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Phylos - Narrative DNA Console</title>
    <style>
        :root {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #f8fbff;
            background: #020a13;
        }
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            min-height: 100vh;
            background: radial-gradient(circle at 20% 20%, rgba(16, 82, 147, 0.35), transparent 60%),
                        radial-gradient(circle at 80% 0%, rgba(103, 20, 134, 0.25), transparent 55%),
                        #020a13;
            padding: 32px;
        }
        h1, h2, h3, h4 {
            margin: 0;
        }
        .page {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        header {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: baseline;
            gap: 8px;
        }
        header p {
            color: #c6d5e5;
            max-width: 640px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }
        .panel {
            background: rgba(3, 14, 27, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 20px 35px rgba(3, 17, 32, 0.55);
        }
        label {
            font-size: 0.9rem;
            color: #a0b9d5;
            display: block;
            margin-bottom: 6px;
        }
        input, select {
            width: 100%;
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.05);
            color: #f1f6ff;
            font-size: 1rem;
        }
        input:focus {
            outline: 2px solid rgba(84, 179, 255, 0.5);
            border-color: transparent;
        }
        .controls {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 12px;
        }
        button {
            border: none;
            border-radius: 999px;
            padding: 10px 20px;
            font-weight: 600;
            cursor: pointer;
            color: #01101f;
            background: linear-gradient(135deg, #6dffd6, #2ba6ff);
            box-shadow: 0 10px 25px rgba(43, 166, 255, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 15px 30px rgba(43, 166, 255, 0.45);
        }
        button.secondary {
            background: rgba(255, 255, 255, 0.15);
            color: #e6eef8;
            box-shadow: none;
        }
        .hint {
            color: #7f93b1;
            font-size: 0.8rem;
            margin: 6px 0 0;
        }
        .hint code {
            color: #c8d8ee;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }
        .badge {
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge.idle {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }
        .badge.connecting {
            background: rgba(255, 193, 59, 0.2);
            color: #ffc13b;
        }
        .badge.connected {
            background: rgba(98, 248, 208, 0.2);
            color: #62f8d0;
        }
        .badge.error {
            background: rgba(255, 87, 127, 0.25);
            color: #ff577f;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
            margin-top: 18px;
        }
        .stat {
            background: rgba(255, 255, 255, 0.03);
            padding: 12px 14px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stat span {
            display: block;
        }
        .stat span.value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #fefefe;
            margin-bottom: 4px;
        }
        .stat span.label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.7rem;
            color: #a5bad5;
        }
        .log {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 360px;
            overflow: auto;
        }
        .log-entry {
            padding: 12px 16px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
        }
        .log-entry h4 {
            font-size: 0.95rem;
            margin-bottom: 6px;
            color: #e2ecf8;
        }
        .log-entry small {
            color: #7f93b1;
        }
        .summary-box {
            padding: 12px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.04);
            margin-top: 10px;
            min-height: 90px;
            white-space: pre-wrap;
        }
        .chat-panel {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .chat-messages {
            max-height: 240px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 12px;
        }
        .chat-entry {
            display: flex;
            gap: 10px;
            align-items: flex-start;
        }
        .chat-entry .avatar {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .chat-entry .bubble {
            border-radius: 16px;
            padding: 10px 14px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.03);
            max-width: 75%;
        }
        .chat-entry.user {
            flex-direction: row-reverse;
            text-align: right;
        }
        .chat-entry.user .bubble {
            background: linear-gradient(135deg, rgba(109, 255, 214, 0.25), rgba(43, 166, 255, 0.2));
            border-color: rgba(109, 255, 214, 0.4);
        }
        .chat-entry.user .avatar {
            background: linear-gradient(135deg, #6dffd6, #2ba6ff);
            color: #062335;
        }
        .chat-entry.assistant .bubble {
            background: rgba(146, 112, 255, 0.18);
            border-color: rgba(146, 112, 255, 0.4);
        }
        .chat-entry.assistant .avatar {
            background: rgba(146, 112, 255, 0.35);
            color: #ffffff;
        }
        .chat-entry.system .bubble {
            background: rgba(255, 193, 59, 0.15);
            border-color: rgba(255, 193, 59, 0.4);
        }
        .chat-entry.system .avatar {
            background: rgba(255, 193, 59, 0.3);
            color: #2d1a00;
        }
        .chat-entry strong {
            display: block;
            font-size: 0.85rem;
            margin-bottom: 4px;
            color: #d6e3ff;
        }
        .chat-entry p {
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.35;
        }
        pre {
            margin: 0;
            margin-top: 8px;
            padding: 12px;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.35);
            overflow-x: auto;
            font-size: 0.8rem;
        }
        .queue-list {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-height: 180px;
            overflow: auto;
        }
        .queue-item {
            border-radius: 10px;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .queue-item span {
            display: block;
            font-size: 0.85rem;
        }
        #chat-input {
            width: 100%;
            min-height: 70px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.05);
            color: #f6f8fb;
            padding: 10px;
            font-size: 0.95rem;
        }
        .empty {
            color: #7689a3;
            font-style: italic;
        }
        @media (max-width: 640px) {
            body {
                padding: 16px;
            }
        }
    </style>
</head>
<body>
    <div class="page">
        <header>
            <div>
                <h1>Phylos - Narrative DNA Console</h1>
                <p>Initiate a trace, watch LangGraph events stream in real-time, and inspect how the agent builds its mutation graph.</p>
            </div>
            <small>WebSocket endpoint: <code>/ws/dna-stream</code></small>
        </header>

        <div class="grid">
            <section class="panel">
                <h3>Trace Configuration</h3>
                <form id="trace-form">
                    <label for="start-url">Start URL</label>
                    <input type="url" id="start-url" value="https://example.com/patient-zero" placeholder="https://..." required />

                    <label for="max-depth">Max Depth</label>
                    <input type="number" id="max-depth" min="1" max="6" value="2" />

                    <label for="ws-endpoint">WebSocket Endpoint (optional)</label>
                    <input type="text" id="ws-endpoint" placeholder="auto (derived from this page)" />
                    <p class="hint">Auto endpoint: <code id="ws-endpoint-auto"></code></p>

                    <div class="controls">
                        <button type="submit" id="start-btn">Stream Narrative DNA</button>
                        <button type="button" class="secondary" id="stop-btn">Stop</button>
                        <button type="button" class="secondary" id="clear-btn">Clear Log</button>
                    </div>
                </form>

                <h4 style="margin-top:18px;">Current Payload</h4>
                <pre id="payload-preview">{}</pre>
            </section>

            <section class="panel">
                <div class="status-row">
                    <h3>Connection</h3>
                    <span id="status-badge" class="badge idle">Idle</span>
                </div>
                <p class="hint">Active endpoint: <code id="status-endpoint"></code></p>
                <div class="stats">
                    <div class="stat">
                        <span class="value" id="stat-events">0</span>
                        <span class="label">Events</span>
                    </div>
                    <div class="stat">
                        <span class="value" id="stat-nodes">0</span>
                        <span class="label">Nodes Seen</span>
                    </div>
                    <div class="stat">
                        <span class="value" id="stat-mutations">0</span>
                        <span class="label">Mutations</span>
                    </div>
                    <div class="stat">
                        <span class="value" id="stat-replications">0</span>
                        <span class="label">Replications</span>
                    </div>
                </div>

                <h4 style="margin-top:20px;">Traversal Queue</h4>
                <ul id="queue" class="queue-list">
                    <li class="empty">Queue is empty</li>
                </ul>
            </section>
        </div>

        <section class="panel">
            <div class="status-row">
                <h3>Event Stream</h3>
                <small>Latest LangGraph + system messages</small>
            </div>
            <div id="log" class="log">
                <div class="empty">No events yet - start a trace to watch the stream.</div>
            </div>
        </section>
        <section class="panel">
            <h3>Gemini Debrief & Chat</h3>
            <p class="hint">Summary appears after each trace. Use the chat to debate or ask follow-ups.</p>
            <div id="summary-text" class="summary-box">Run a trace to see Gemini's findings.</div>
            <div class="chat-panel">
                <div id="chat-messages" class="chat-messages">
                    <div class="empty">Chat unlocks once a summary is ready.</div>
                </div>
                <form id="chat-form">
                    <textarea id="chat-input" placeholder="Ask Gemini about this narrative..." disabled></textarea>
                    <div class="controls">
                        <button type="submit" class="secondary" id="chat-send" disabled>Ask Gemini</button>
                    </div>
                </form>
            </div>
        </section>
    </div>

    <script>
        (function() {
            const urlInput = document.getElementById('start-url');
            const depthInput = document.getElementById('max-depth');
            const wsInput = document.getElementById('ws-endpoint');
            const form = document.getElementById('trace-form');
            const stopBtn = document.getElementById('stop-btn');
            const clearBtn = document.getElementById('clear-btn');
            const statusBadge = document.getElementById('status-badge');
            const statusEndpoint = document.getElementById('status-endpoint');
            const autoEndpointEl = document.getElementById('ws-endpoint-auto');
            const logContainer = document.getElementById('log');
            const queueList = document.getElementById('queue');
            const payloadPreview = document.getElementById('payload-preview');
            const statEvents = document.getElementById('stat-events');
            const statNodes = document.getElementById('stat-nodes');
            const statMutations = document.getElementById('stat-mutations');
            const statReplications = document.getElementById('stat-replications');
            const summaryBox = document.getElementById('summary-text');
            const chatMessages = document.getElementById('chat-messages');
            const chatForm = document.getElementById('chat-form');
            const chatInput = document.getElementById('chat-input');
            const chatSend = document.getElementById('chat-send');

            let ws = null;
            let stats = { events: 0, nodes: 0, mutations: 0, replications: 0 };
            const seenNodes = new Set();
            const fallbackEndpoint = null;
            let manualEndpointSupplied = false;
            let attemptedFallback = false;
            let activePayload = null;
            let currentSessionId = null;
            let summaryReady = false;

            function computeDefaultEndpoint() {
                const baseProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
                let hostname = window.location.hostname;
                if (!hostname || hostname === '0.0.0.0' || hostname === '::' || hostname === '[::]') {
                    hostname = '127.0.0.1';
                }
                const port = window.location.port ? `:${window.location.port}` : '';
                return `${baseProtocol}://${hostname}${port}/ws/dna-stream`;
            }

            function refreshEndpointHints() {
                const auto = computeDefaultEndpoint();
                if (autoEndpointEl) {
                    autoEndpointEl.textContent = auto;
                }
                if (statusEndpoint && !statusEndpoint.textContent) {
                    statusEndpoint.textContent = auto;
                }
                if (!wsInput.value.trim() && statusEndpoint) {
                    statusEndpoint.textContent = auto;
                }
            }
            refreshEndpointHints();

            function updateStatus(state, text) {
                statusBadge.className = `badge ${state}`;
                statusBadge.textContent = text;
            }

            function resetStats() {
                stats = { events: 0, nodes: 0, mutations: 0, replications: 0 };
                seenNodes.clear();
                updateStats();
                renderQueue([]);
            }

            function updateStats() {
                statEvents.textContent = stats.events;
                statNodes.textContent = stats.nodes;
                statMutations.textContent = stats.mutations;
                statReplications.textContent = stats.replications;
            }

            function setChatEnabled(enabled) {
                chatInput.disabled = !enabled;
                chatSend.disabled = !enabled;
            }

            setChatEnabled(false);

            function renderQueue(queue) {
                queueList.innerHTML = '';
                if (!queue || !queue.length) {
                    queueList.innerHTML = '<li class="empty">Queue is empty</li>';
                    return;
                }
                queue.forEach(function(item, idx) {
                    let url = item;
                    let parent = '';
                    let depth = '?';
                    if (Array.isArray(item)) {
                        [url, parent, depth] = item;
                    } else if (typeof item === 'object' && item !== null) {
                        url = item.url || item[0];
                        parent = item.parent || item[1] || '';
                        depth = item.depth ?? item[2] ?? '?';
                    }
                    const li = document.createElement('li');
                    li.className = 'queue-item';
                    li.innerHTML = `
                        <span><strong>#${idx + 1}</strong> Depth ${depth}</span>
                        <span>${url}</span>
                        <span style="color:#7f93b1;">Parent: ${parent || 'GLOBAL'}</span>
                    `;
                    queueList.appendChild(li);
                });
            }

            function addLogEntry(kind, title, payload) {
                if (logContainer.querySelector('.empty')) {
                    logContainer.innerHTML = '';
                }

                const entry = document.createElement('article');
                entry.className = 'log-entry';
                const timestamp = new Date().toLocaleTimeString();
                entry.innerHTML = `
                    <h4>${title}</h4>
                    <small>${kind} - ${timestamp}</small>
                `;
                if (payload !== undefined) {
                    const block = document.createElement('pre');
                    block.textContent = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2);
                    entry.appendChild(block);
                }
                logContainer.prepend(entry);

                while (logContainer.children.length > 50) {
                    logContainer.removeChild(logContainer.lastChild);
                }
            }

            function appendChatMessage(author, text, type = 'user') {
                if (chatMessages.querySelector('.empty')) {
                    chatMessages.innerHTML = '';
                }
                const entry = document.createElement('div');
                entry.className = `chat-entry ${type}`;

                const avatar = document.createElement('div');
                avatar.className = 'avatar';
                avatar.textContent = type === 'assistant' ? 'G' : type === 'user' ? 'You' : '!';

                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                const authorEl = document.createElement('strong');
                authorEl.textContent = author;
                const textEl = document.createElement('p');
                textEl.textContent = text;
                bubble.appendChild(authorEl);
                bubble.appendChild(textEl);

                entry.appendChild(avatar);
                entry.appendChild(bubble);

                chatMessages.appendChild(entry);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function handleIncoming(message) {
                if (message.status) {
                    if (message.status === 'session') {
                        currentSessionId = message.session_id;
                        addLogEntry('system', 'Session established', `Chat session ${currentSessionId.slice(0, 8)}…`);
                        return;
                    }
                    if (message.status === 'summary') {
                        summaryReady = true;
                        summaryBox.textContent = message.summary;
                        setChatEnabled(true);
                        appendChatMessage('Gemini', 'Summary updated. Ask about any part of the trace.', 'assistant');
                        addLogEntry('system', 'Gemini summary ready', message.summary);
                        return;
                    }
                    const kind = message.status === 'error' ? 'error' : 'system';
                    addLogEntry(kind, message.message || message.status, message);
                    return;
                }

                stats.events += 1;
                const data = message.data || {};

                if (data.current_article && data.current_article.id) {
                    seenNodes.add(data.current_article.id);
                    stats.nodes = seenNodes.size;
                }

                const edgeStats = data.edge_stats;
                if (edgeStats) {
                    stats.mutations += edgeStats.mutation_count || 0;
                    stats.replications += edgeStats.replication_count || 0;
                } else if (data.knowledge_graph && Array.isArray(data.knowledge_graph.edges)) {
                    data.knowledge_graph.edges.forEach(function(edge) {
                        const relation = edge.attributes && edge.attributes.relation_type;
                        if (relation === 'Mutation') {
                            stats.mutations += 1;
                        } else if (relation === 'Replication') {
                            stats.replications += 1;
                        }
                    });
                }

                if (Array.isArray(data.traversal_queue)) {
                    renderQueue(data.traversal_queue);
                }

                updateStats();

                const title = `[${message.event}] ${message.name}`;
                addLogEntry('event', title, message);
            }

            function teardownSocket() {
                if (ws) {
                    try {
                        ws.onclose = null;
                        ws.onerror = null;
                        ws.onmessage = null;
                        ws.close(1000, 'Restarting trace');
                    } catch (err) {}
                    ws = null;
                }
            }

            function connectWebSocket(endpoint) {
                if (!endpoint) {
                    endpoint = computeDefaultEndpoint();
                }
                teardownSocket();
                updateStatus('connecting', 'Connecting...');
                if (statusEndpoint) {
                    statusEndpoint.textContent = endpoint;
                }

                ws = new WebSocket(endpoint);

                ws.onopen = function() {
                    updateStatus('connected', 'Streaming');
                    addLogEntry('system', 'Connected - streaming events', { endpoint });
                    if (activePayload) {
                        ws.send(JSON.stringify(activePayload));
                    }
                };

                ws.onmessage = function(event) {
                    try {
                        const msg = JSON.parse(event.data);
                        handleIncoming(msg);
                    } catch (err) {
                        addLogEntry('error', 'Failed to parse event', { endpoint, raw: event.data });
                    }
                };

                ws.onerror = function(event) {
                    const detail = event?.message || (event?.error && event.error.message) || `readyState=${ws?.readyState ?? 'n/a'}`;
                    addLogEntry('error', 'WebSocket error', { endpoint, detail });
                    if (!manualEndpointSupplied && !attemptedFallback && fallbackEndpoint) {
                        attemptedFallback = true;
                        addLogEntry('system', 'Retrying via fallback', { endpoint: fallbackEndpoint });
                        connectWebSocket(fallbackEndpoint);
                        return;
                    }
                    updateStatus('error', 'Error');
                };

                ws.onclose = function(evt) {
                    const info = { endpoint, code: evt.code, reason: evt.reason || 'n/a' };
                    const wasIntentional = evt.code === 1000;
                    addLogEntry('system', wasIntentional ? 'Trace ended' : 'Connection closed unexpectedly', info);
                    if (ws === this) {
                        ws = null;
                    }
                    updateStatus('idle', 'Idle');
                };
            }

            function startTrace(evt) {
                evt.preventDefault();
                const payload = {
                    start_url: urlInput.value.trim(),
                    max_depth: parseInt(depthInput.value, 10) || 1,
                };

                if (!payload.start_url) {
                    addLogEntry('system', 'Start URL required', 'Please provide a valid URL before starting.');
                    return;
                }

                manualEndpointSupplied = Boolean(wsInput.value.trim());
                attemptedFallback = false;
                activePayload = payload;

                payloadPreview.textContent = JSON.stringify(payload, null, 2);
                resetStats();
                summaryReady = false;
                summaryBox.textContent = 'Trace running... summary will appear here.';
                setChatEnabled(false);
                chatMessages.innerHTML = '<div class="empty">Waiting for Gemini summary...</div>';
                currentSessionId = null;

                const targetEndpoint = wsInput.value.trim() || computeDefaultEndpoint();
                addLogEntry('system', 'Connecting to agent', { ...payload, endpoint: targetEndpoint });
                connectWebSocket(targetEndpoint);
            }

            function stopTrace() {
                if (ws) {
                    ws.onclose = null;
                    ws.onerror = null;
                    ws.close(1000, 'Stopped by user');
                    ws = null;
                }
                addLogEntry('system', 'Trace stopped', 'Stopped by user');
                updateStatus('idle', 'Idle');
            }

            function clearLog() {
                logContainer.innerHTML = '<div class="empty">Log cleared.</div>';
            }

            if (wsInput) {
                wsInput.addEventListener('input', refreshEndpointHints);
            }
            form.addEventListener('submit', startTrace);
            stopBtn.addEventListener('click', stopTrace);
            clearBtn.addEventListener('click', clearLog);
            chatForm.addEventListener('submit', async function(evt) {
                evt.preventDefault();
                const message = chatInput.value.trim();
                if (!message) {
                    return;
                }
                if (!currentSessionId || !summaryReady) {
                    appendChatMessage('System', 'Chat is unavailable until a trace finishes.', 'system');
                    return;
                }
                appendChatMessage('You', message, 'user');
                chatInput.value = '';
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({session_id: currentSessionId, message}),
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        appendChatMessage('System', data.detail || 'Chat request failed.', 'system');
                        return;
                    }
                    appendChatMessage('Gemini', data.reply || 'No response.', 'assistant');
                } catch (error) {
                    appendChatMessage('System', 'Unable to reach chat endpoint.', 'system');
                }
            });
        })();
    </script>
</body>
</html>
"""

@api.get("/")
async def get():
    """Serves the modern React-based UI."""
    return HTMLResponse(REACT_HTML)

@api.get("/classic")
async def get_classic():
    """Serves the classic console UI."""
    return HTMLResponse(CLASSIC_HTML)

@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handles follow-up chat requests about the latest trace summary."""
    context = SESSION_CONTEXTS.get(request.session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Session not found. Run a trace first.")
    if not context.get("summary"):
        raise HTTPException(status_code=400, detail="Summary not ready yet. Please wait for the trace to finish.")

    history: List[Dict[str, str]] = context.setdefault("history", [])
    history.append({"role": "user", "content": request.message})
    reply = _generate_chat_reply(context["summary"], history, request.message)
    history.append({"role": "assistant", "content": reply})
    return {"reply": reply}

@api.get("/session/{session_id}/graph")
async def session_graph(session_id: str):
    """Returns the accumulated knowledge graph for a completed session."""
    context = SESSION_CONTEXTS.get(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Session not found.")

    graph = context.get("knowledge_graph") or {"nodes": {}, "edges": []}
    origin_info = context.get("origin") or {}
    node_snapshots = _build_node_snapshots(
        graph,
        origin_embedding=origin_info.get("embedding"),
        origin_content=origin_info.get("content"),
        origin_url=origin_info.get("url"),
    )
    edge_snapshots = _build_edge_snapshots(graph.get("edges") or [])
    stats = {
        "nodes": len(node_snapshots),
        "edges": len(edge_snapshots),
        "mutations": sum(1 for edge in edge_snapshots if edge["attributes"].get("relation_type") == "Mutation"),
        "replications": sum(1 for edge in edge_snapshots if edge["attributes"].get("relation_type") == "Replication"),
    }
    return {
        "nodes": node_snapshots,
        "edges": edge_snapshots,
        "stats": stats,
        "origin": {"url": origin_info.get("url")},
        "investigation_timeline": _build_investigation_timeline(node_snapshots, context),
    }


@api.websocket("/ws/dna-stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    The main WebSocket endpoint for streaming graph analysis events.
    """
    logger.info("WebSocket connection attempt from %s", websocket.client)
    await websocket.accept()
    session_id, session_context = _create_session()
    await websocket.send_json({"status": "session", "session_id": session_id})
    try:
        # 1. Receive the initial request from the client
        initial_data = await websocket.receive_json()
        request = InitialArticleRequest(**initial_data)
        logger.info("WebSocket trace request received: %s", request.model_dump())
        
        await websocket.send_json({"status": "info", "message": f"Received request to trace: {request.start_url}"})
        logger.debug("Acknowledgement sent to client.")

        # 2. Prepare the initial state for the graph
        patient_zero_content = fetch_article_content(request.start_url)
        global_context_embedding = embedder(patient_zero_content["content"])
        logger.info("Prepared patient zero content for %s", request.start_url)

        initial_state: GraphState = {
            "traversal_queue": [(request.start_url, None, 0)], # (url, parent_id, depth)
            "knowledge_graph": {"nodes": {}, "edges": []},
            "visited_urls": [],
            "host_visit_counts": {},
            "global_context": global_context_embedding,
            "current_article": None,
            "parent_article_id": None,
            "max_depth": request.max_depth,
        }
        session_context["origin"] = {
            "url": request.start_url,
            "embedding": global_context_embedding,
            "content": patient_zero_content["content"],
        }
        logger.debug(
            "Initial graph state seeded: queue=%s max_depth=%s",
            initial_state["traversal_queue"],
            request.max_depth,
        )

        # 3. Stream the graph execution events back to the client
        async for event in app.astream_events(
            initial_state,
            version="v1",
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        ):
            raw_data = event.get("data")
            edge_updates = _extract_edge_updates(raw_data)
            _accumulate_graph(session_context["knowledge_graph"], raw_data)
            event_keys = list((event.get("data") or {}).keys())
            logger.debug(
                "Streaming event: event=%s name=%s data_keys=%s",
                event.get("event"),
                event.get("name"),
                event_keys,
            )
            summarized_data = _summarize_event_data(raw_data, edge_updates=edge_updates)
            sanitized_data = _sanitize_payload(summarized_data)
            await websocket.send_json({
                "event": event["event"],
                "name": event["name"],
                "data": sanitized_data,
            })
                if raw_data and isinstance(raw_data, dict):
                    article = raw_data.get("current_article")
                    investigation_entry = _maybe_record_investigation(session_context, article, rapid=True)
                    if investigation_entry and not investigation_entry.get("_sent"):
                        sanitized_entry = {
                            "id": investigation_entry.get("id"),
                            "title": investigation_entry.get("title"),
                            "timestamp": investigation_entry.get("timestamp"),
                            "summary": _sanitize_payload(investigation_entry.get("summary")),
                            "reason": _sanitize_payload(investigation_entry.get("reason")),
                            "investigation": _sanitize_payload(
                                investigation_entry.get("investigation", investigation_entry.get("reason"))
                            ),
                            "url": investigation_entry.get("url"),
                        }
                    await websocket.send_json({
                        "status": "investigation",
                        "entry": sanitized_entry,
                    })
                    investigation_entry["_sent"] = True
                if article:
                    asyncio.create_task(
                        _upgrade_investigation_entry(session_context, article, websocket)
                    )
        
        await websocket.send_json({"status": "info", "message": "Graph traversal complete."})
        logger.info("WebSocket trace completed successfully for %s", request.start_url)
        summary_text = _generate_graph_summary(session_context["knowledge_graph"])
        session_context["summary"] = summary_text
        await websocket.send_json({
            "status": "summary",
            "session_id": session_id,
            "summary": summary_text,
        })
        logger.info("Graph summary ready for session %s", session_id)

    except WebSocketDisconnect:
        logger.warning("Client disconnected prematurely.")
    except Exception as e:
        error_message = f"An error occurred: {type(e).__name__} - {e}"
        logger.exception("Unhandled exception during WebSocket trace: %s", error_message)
        await websocket.send_json({"status": "error", "message": error_message})
    finally:
        if websocket.client_state.name != 'DISCONNECTED':
            await websocket.close()
        logger.info("WebSocket connection closed for client %s", websocket.client)

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
def _maybe_record_investigation(
    session_context: Dict[str, Any],
    article: Dict[str, Any] | None,
    *,
    rapid: bool = False,
    force: bool = False,
) -> Dict[str, Any] | None:
    if not article:
        return None
    article_id = article.get("id")
    if not article_id:
        return None
    origin = session_context.get("origin")
    if origin and article_id == origin.get("url"):
        return None
    investigation_map = session_context.setdefault("investigation_map", {})
    existing = investigation_map.get(article_id)
    if existing and not force:
        if "timestamp" not in existing and article.get("timestamp"):
            existing["timestamp"] = article.get("timestamp")
        return existing

    origin = session_context.get("origin")
    if not origin or article_id == origin.get("url"):
        return None

    entry = _build_investigation_entry(origin, article, rapid=rapid)
    if not entry:
        return None

    last_summary = session_context.get("last_investigation_summary")
    if entry["summary"] and entry["summary"] == last_summary:
        host = _short_host(article_id)
        entry["summary"] = f"{entry['summary']} (Source: {host or article_id})"
    session_context["last_investigation_summary"] = entry["summary"]

    if existing:
        existing.update(entry)
        entry = existing
    else:
        investigation_map[article_id] = entry
        session_context.setdefault("investigation", []).append(entry)
    return entry

async def _upgrade_investigation_entry(
    session_context: Dict[str, Any],
    article: Dict[str, Any],
    websocket: WebSocket
) -> None:
    loop = asyncio.get_running_loop()
    entry = await loop.run_in_executor(
        None,
        lambda: _maybe_record_investigation(session_context, article, rapid=False, force=True)
    )
    if not entry:
        return
    sanitized_entry = {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "timestamp": entry.get("timestamp"),
        "summary": _sanitize_payload(entry.get("summary")),
        "reason": _sanitize_payload(entry.get("reason")),
        "investigation": _sanitize_payload(entry.get("investigation", entry.get("reason"))),
        "url": entry.get("url"),
    }
    try:
        await websocket.send_json({
            "status": "investigation_update",
            "entry": sanitized_entry,
        })
    except Exception as exc:
        logger.debug("Failed to push investigation update: %s", exc)
