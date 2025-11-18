
# FastAPI server for the Narrative DNA Sequencer.
# Provides a WebSocket endpoint so clients can stream traversal events.

import os
import json
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

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

# --- Local Imports ---
from state import GraphState, InitialArticleRequest
from graph_builder import app, embedder, fetch_article_content, GRAPH_RECURSION_LIMIT

# --- FastAPI App Initialization ---
api = FastAPI(
    title="Narrative DNA Sequencer",
    description="A backend for tracing semantic mutations in narrative networks.",
    version="0.1.0",
)

# Single-page console to interact with the agent
HTML = """
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

            let ws = null;
            let stats = { events: 0, nodes: 0, mutations: 0, replications: 0 };
            const seenNodes = new Set();
            const fallbackEndpoint = null;
            let manualEndpointSupplied = false;
            let attemptedFallback = false;
            let activePayload = null;

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

            function handleIncoming(message) {
                stats.events += 1;
                const data = message.data || {};

                if (data.current_article && data.current_article.id) {
                    seenNodes.add(data.current_article.id);
                    stats.nodes = seenNodes.size;
                }

                if (data.knowledge_graph && Array.isArray(data.knowledge_graph.edges)) {
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
        })();
    </script>
</body>
</html>
"""

@api.get("/")
async def get():
    """Serves a simple HTML page to interact with the WebSocket."""
    return HTMLResponse(HTML)


@api.websocket("/ws/dna-stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    The main WebSocket endpoint for streaming graph analysis events.
    """
    logger.info("WebSocket connection attempt from %s", websocket.client)
    await websocket.accept()
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
            "global_context": global_context_embedding,
            "current_article": None,
            "parent_article_id": None,
            "max_depth": request.max_depth,
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
            logger.debug("Streaming event to client: %s", event)
            await websocket.send_json({
                "event": event["event"],
                "name": event["name"],
                "data": event["data"],
            })
        
        await websocket.send_json({"status": "info", "message": "Graph traversal complete."})
        logger.info("WebSocket trace completed successfully for %s", request.start_url)

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
