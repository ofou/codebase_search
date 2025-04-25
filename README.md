# Codebase Search MCP Server

A Model Context Protocol (MCP) server for semantic codebase search using Ollama embeddings.

## Overview

This MCP server allows Cursor to search your codebase semantically using the Ollama embedding model. It provides a powerful way to find relevant code snippets based on natural language queries.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- The `nomic-embed-text` model pulled in Ollama

## Installation

````
git clone git@github.com:ofou/codebase_search.git
cd codebase_search
brew install ollama uv
uv venv && uv pip install -r requirements.txt
```

3. Ensure Ollama is running and has the required model:

```bash
ollama serve
```
4. Pull the `nomic-embed-text` model from another terminal:

```bash
ollama pull nomic-embed-text
```

5. Make sure to copy the `.cursor/mcp.json` file to your project directory. This file contains the configuration for the MCP server.

```bash
cp .kilocode/mcp.json
```

## Configuration

The server is configured to:

- Use the `nomic-embed-text` model for embeddings
- Return the top 20 most relevant files
- Skip binary and large files
- Cache embeddings for better performance

You can modify these settings in the `mcp_codebase_server.py` file.

## Usage

### Running as a standalone server

To run the server directly:

```bash
python mcp_codebase_server.py
```

This will start the server on port 8086.

### Integration with Cursor

The `.cursor/mcp.json` file is already configured to use the server. When you open Cursor in this project, it will automatically detect and use the MCP server.

To use it in Cursor:

1. Open Cursor in this project
2. In chat, ask Cursor to search your codebase with a natural language query
3. Example: "Find code related to handling API authentication"

### API

The server exposes a single tool:

- `codebase_search`: Searches the codebase for files relevant to a query
  - Parameters:
    - `query`: The search query (required)
    - `target_directories`: List of directories to search (optional)
    - `explanation`: Reason for the search (optional)

## Testing

Run the automated tests:

```bash
pytest
```

## Troubleshooting

- If you see embedding errors, make sure Ollama is running and has the `nomic-embed-text` model
- For large codebases, the initial search may take some time as embeddings are generated
- Subsequent searches will be faster due to caching
