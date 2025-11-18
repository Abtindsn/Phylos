
FastAPI server to expose the Narrative DNA Sequencer via a WebSocket.

This server provides a real-time streaming endpoint for clients to observe
the graph traversal and analysis as it happens.

import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# --- Local Imports ---
from state import GraphState, InitialArticleRequest
from graph_builder import app, embedder, fetch_article_content

# --- FastAPI App Initialization ---
api = FastAPI(
    title="Narrative DNA Sequencer",
    description="A backend for tracing semantic mutations in narrative networks.",
    version="0.1.0",
)

# A simple HTML page for testing the WebSocket connection
HTML = """
<!DOCTYPE html>
<html>
    <head>
        <title>Narrative DNA Sequencer</title>
    </head>
    <body>
        <h1>WebSocket Log</h1>
        <form action="" onsubmit="sendMessage(event)">
            <label for="url">Start URL:</label>
            <input type="text" id="url" name="url" value="http://example.com/patient-zero" autocomplete="off"/>
            <br/>
            <label for="depth">Max Depth:</label>
            <input type="number" id="depth" name="depth" value="2" autocomplete="off"/>
            <br/><br/>
            <button>Start Trace</button>
        </form>
        <h2>Log:</h2>
        <pre id="messages"></pre>
        <script>
            var ws = null;
            function sendMessage(event) {
                event.preventDefault();
                if (ws) {
                    ws.close();
                }
                var url = document.getElementById('url').value;
                var depth = parseInt(document.getElementById('depth').value);
                document.getElementById('messages').textContent = ''; // Clear log

                ws = new WebSocket(`ws://localhost:8000/ws/dna-stream`);
                ws.onopen = function() {
                    console.log("WebSocket connection established.");
                    ws.send(JSON.stringify({start_url: url, max_depth: depth}));
                };
                ws.onmessage = function(event) {
                    var messages = document.getElementById('messages');
                    var data = JSON.parse(event.data);
                    messages.textContent += JSON.stringify(data, null, 2) + '\n\n';
                };
                ws.onerror = function(event) {
                    console.error("WebSocket error observed:", event);
                    var messages = document.getElementById('messages');
                    messages.textContent += 'Error connecting to WebSocket.\n';
                }
                ws.onclose = function(event) {
                    console.log("WebSocket connection closed.");
                    ws = null;
                }
            }
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
    await websocket.accept()
    try:
        # 1. Receive the initial request from the client
        initial_data = await websocket.receive_json()
        request = InitialArticleRequest(**initial_data)
        
        await websocket.send_json({"status": "info", "message": f"Received request to trace: {request.start_url}"})

        # 2. Prepare the initial state for the graph
        patient_zero_content = fetch_article_content(request.start_url)
        global_context_embedding = embedder(patient_zero_content["content"])

        initial_state: GraphState = {
            "traversal_queue": [(request.start_url, "GLOBAL_CONTEXT", 0)], # (url, parent_id, depth)
            "knowledge_graph": {"nodes": {}, "edges": []},
            "global_context": global_context_embedding,
            "current_article": None,
            "parent_article_id": None,
            "max_depth": request.max_depth,
        }

        # 3. Stream the graph execution events back to the client
        async for event in app.astream_events(initial_state, version="v1"):
            await websocket.send_json({
                "event": event["event"],
                "name": event["name"],
                "data": event["data"],
            })
        
        await websocket.send_json({"status": "info", "message": "Graph traversal complete."})

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        error_message = f"An error occurred: {type(e).__name__} - {e}"
        print(error_message)
        await websocket.send_json({"status": "error", "message": error_message})
    finally:
        if websocket.client_state.name != 'DISCONNECTED':
            await websocket.close()
        print("WebSocket connection closed.")

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
