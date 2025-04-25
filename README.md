# 🔍 Local `codebase_search` MCP Server

Semantic codebase search using Ollama embeddings via Model Context Protocol (MCP).

## 🌟 Overview
Search your codebase semantically with Ollama's `nomic-embed-text` model. Works with local Ollama server and integrates with VSCode.

## 📋 Prerequisites
- 🐍 Python 3.8+
- 🤖 [Ollama](https://ollama.ai/) installed and running
- 📊 `nomic-embed-text` model in Ollama (274MB)

## 🚀 Installation
```bash
git clone git@github.com:ofou/codebase_search.git
cd codebase_search
brew install ollama uv
uv venv && uv pip install -r requirements.txt

# Start Ollama
ollama serve
```

```bash
# In another terminal, pull the model
ollama pull nomic-embed-text
```

## 🛠️ Setup
Open the `mcp_codebase_server.py` file and set the `codebase_path` variable to your codebase directory. This is where the server will look for files to search.

## ⚙️ Configuration
Server uses `nomic-embed-text` for embeddings, returns top 20 relevant files, skips binary/large files, and caches for performance. Modify settings in `mcp_codebase_server.py`.

## 💻 Usage

### 🔗 Kilo Code Integration
The `.kilocode/mcp.json` file enables automatic MCP server detection. Just open VSCode and Kilo Code in your project and ask natural language questions like "Find code related to handling API authentication"

### 🛠️ API
Single tool: `codebase_search`
- Parameters:
  - `query`: Search query (required)
  - `target_directories`: Directories to search (optional)
  - `explanation`: Search reason (optional)

## ✅ To Do
- [ ] Add a lightweight vector store
- [ ] Implement a chunking strategy for large files
- [ ] Index files in the background on init
- [ ] Add a Reranker for better results
- [ ] Testing and validation
- [ ] Add more file types to ignore

## ❓ Troubleshooting
- 🔄 Embedding errors? Ensure Ollama is running with `nomic-embed-text`
- ⏱️ Initial search may be slow for large codebases
- 💨 Subsequent searches are faster thanks to caching
