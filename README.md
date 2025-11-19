# Phylos - Narrative DNA Sequencer

This project is a backend service designed to trace the evolution of a narrative through a network of articles. It uses LangGraph and FastAPI to identify where "semantic mutations" (disinformation or spin) occur in the narrative chain.

## Core Concept

- **Nodes**: Articles/Posts (with attributes like content, timestamp, author, and vector embedding).
- **Edges**: Citations/Links (with attributes like mutation score and relation type).
- **Signal**: High Mutation Score, indicating a significant semantic drift from the source.
- **Noise**: Low Mutation Score, indicating verbatim copying or reposting.

## Features

- **State Management**: A `GraphState` schema to track the knowledge graph, traversal queue, and global context.
- **Graph Nodes**:
    - `Node_Acquire`: Fetches content from a URL.
    - `Node_Sequence`: Compares articles to calculate semantic drift and identify mutations.
    - `Node_Branch`: Extracts hyperlinks to continue the analysis recursively.
- **FastAPI Integration**: A WebSocket endpoint (`/ws/dna-stream`) to stream graph updates in real-time.
- **Data Models**: Pydantic V2 models for data structures like `MutationEvent`.

## Getting Started

This guide provides a complete, copy-paste-friendly set of commands to get the agent running locally.

### Prerequisites

-   **Git**: To clone the repository.
-   **Docker**: To build and run the application container.
-   **Gemini API Key**: You need an active API key from Google AI Studio.

### Setup & Run

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-repo/phylos.git
    cd phylos
    ```
    *(Note: Replace `your-repo/phylos` with the actual repository path if you have one).*

2.  **Configure Environment**

    Create a `.env` file from the provided example. This file will securely store your API key.

    ```bash
    cp .env.example .env
    ```

    Now, open the newly created `.env` file and paste your Gemini API key into it:

    ```env
    # .env
    GEMINI_API_KEY="PASTE_YOUR_GEMINI_API_KEY_HERE"
    PHYLOS_OFFLINE_MODE=0   # keep at 0 to fetch real-world sites
    # Optional: stop the crawler from looping endlessly on the same domain
    # PHYLOS_MAX_VISITS_PER_HOST=8
    # Optional: limit how many referenced URLs per article we surface (default 40)
    # PHYLOS_REFERENCE_PREVIEW_LIMIT=40
    # Optional: extend the link blocklist (comma-separated host suffixes)
    # PHYLOS_LINK_BLOCKLIST=example.com,offline.phylos
    # Optional: choose the Gemini model used for origin-difference insights (defaults to gemini-2.5-pro)
    # PHYLOS_ORIGIN_INSIGHT_MODEL=gemini-2.5-pro
    ```

    > **Important:** Setting `PHYLOS_OFFLINE_MODE=1` forces the crawler to use stubbed
    > `example.com` content. Leave it at `0` (or remove it entirely) whenever you want
    > to analyze live news sources such as CNN, Fox News, etc. Because the server runs
    > inside Docker, make sure the container has outbound network access so the HTTP
    > fetcher can reach those sites.

3.  **Build and Run the Docker Container**

    The following command builds the Docker image and runs the container in one step. It uses the `--env-file` flag to securely pass your API key to the application.

    ```bash
    docker build -t phylos-app . && docker run --env-file .env -p 8000:8000 phylos-app
    ```

> ⚠️ **Real-web crawling vs. stub mode**
>
> The command above uses whatever value you placed in `.env`. Do **not** add
> `-e PHYLOS_OFFLINE_MODE=1` unless you explicitly want the offline stubs (the
> ones that emit `http://example.com/...`). When the variable is `0`, Phylos will
> fetch the actual article content and outbound links from the live site.
>
> You can still control the traversal depth per run. The Chronicle UI now
> exposes a **Depth Limit** dropdown (1–6 hops, default 3) next to the URL
> field, so you no longer have to hand-edit the WebSocket payload. If you're
> scripting against the WebSocket endpoint directly, continue to include a
> `max_depth` in the payload:
>
> ```json
> { "start_url": "https://www.cnn.com/...", "max_depth": 2 }
> ```
>
> When omitted, the backend falls back to the server default (3).
>
> Bad/binary links are skipped automatically now. If a URL resolves to an image,
> video, or any unsupported content type, Phylos discards it instead of inventing
> placeholder "simulated" articles—keeping the Chronicle free of nonsense blobs.

    The FastAPI server will now be running and accessible at `http://localhost:8000`. You can connect to the WebSocket at `ws://localhost:8000/ws/dna-stream`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
