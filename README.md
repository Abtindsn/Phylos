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

### Prerequisites

- Docker
- Python 3.10+

### Installation and Running

1.  **Build the Docker image:**

    ```bash
    docker build -t phylos-app .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -p 8000:8000 phylos-app
    ```

The FastAPI server will be available at `http://localhost:8000`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
