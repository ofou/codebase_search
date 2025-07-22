# 🔍 Local `codebase_search` MCP Server

Semantic codebase search using Ollama embeddings via Model Context Protocol (MCP) with **file watching** and **background indexing**.

## 🌟 Overview
Search your codebase semantically with Ollama's `nomic-embed-text` model. Works with local Ollama server and integrates with VSCode. Now with **automatic file indexing** and **real-time updates**!

## ✨ New Features
- 🐳 **Docker Support**: Run in containers for easy deployment
- 👀 **File Watching**: Automatically updates index when files change
- 🔄 **Background Indexing**: Indexes files in the background for faster searches
- 📊 **Index Status**: Monitor indexing progress and status
- 🏥 **Health Checks**: Built-in health monitoring

## 📋 Prerequisites
- 🐍 Python 3.8+ (for local development)
- 🐳 Docker & Docker Compose (for containerized deployment)
- 🤖 [Ollama](https://ollama.ai/) installed and running

## 🚀 Quick Start (Docker)

### Option 1: Docker Compose (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd codebase_search

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Option 2: Docker Build
```bash
# Build the image
docker build -t codebase-search .

# Run the container
docker run -d \
  --name codebase-search \
  -p 8086:8086 \
  -v $(pwd):/app \
  codebase-search
```

## 🛠️ Local Development

### Installation
```bash
git clone <repository-url>
cd codebase_search
pip install -r requirements.txt

# Start Ollama
ollama serve

# In another terminal, pull the model
ollama pull nomic-embed-text
```

### Running Locally
```bash
# Start the server
python mcp_codebase_server.py

# Or use the startup script
./start_server.sh
```

## ⚙️ Configuration

### Environment Variables
- `OLLAMA_HOST`: Ollama server host (default: localhost)
- `ENABLE_BACKGROUND_INDEXING`: Enable background indexing (default: true)
- `ENABLE_FILE_WATCHING`: Enable file watching (default: true)
- `INDEX_UPDATE_DELAY`: Delay before re-indexing changed files (default: 2.0s)

### File Filtering
The server automatically skips:
- Binary files (`.pyc`, `.jpg`, `.png`, etc.)
- Large files (>5MB)
- Hidden files and directories
- Common ignored directories (`.git`, `node_modules`, etc.)

## 💻 Usage

### 🔗 Kilo Code Integration
The `.kilocode/mcp.json` file enables automatic MCP server detection. Just open VSCode and Kilo Code in your project and ask natural language questions like:
- "Find code related to handling API authentication"
- "Show me database connection code"
- "Where is the user login function?"

### 🛠️ API Tools

#### `codebase_search`
Find code snippets by semantic meaning.
- **Parameters:**
  - `query`: Search query (required)
  - `target_directories`: Directories to search (optional)
  - `explanation`: Search reason (optional)

#### `get_index_status`
Get the current status of the file index.
- **Returns:** Index statistics and status information

## 🔧 Monitoring

### Health Check
```bash
# Check if the server is healthy
curl http://localhost:8086/health

# Or use the health check script
python health_check.py
```

### Index Status
Use the `get_index_status` tool to monitor:
- Number of indexed files
- Cache size
- Background indexing status
- File watching status

## 🐳 Docker Commands

### Build and Run
```bash
# Build the image
docker build -t codebase-search .

# Run with volume mount for file watching
docker run -d \
  --name codebase-search \
  -p 8086:8086 \
  -v $(pwd):/app \
  -v ollama_data:/root/.ollama \
  codebase-search
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f codebase-search

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## 📊 Performance

### Indexing
- **Initial Indexing**: Indexes all files on startup (configurable)
- **Background Updates**: Automatically re-indexes changed files
- **Batch Processing**: Processes files in batches for memory efficiency
- **Caching**: Caches embeddings in memory and on disk

### Search
- **Fast Queries**: Uses pre-indexed embeddings for instant results
- **Similarity Scoring**: Cosine similarity for accurate results
- **Configurable Results**: Returns top N most relevant files

## ✅ To Do
- [x] Add Docker support
- [x] Implement file watching
- [x] Add background indexing
- [x] Add health checks
- [x] Add index status monitoring
- [ ] Add a lightweight vector store
- [ ] Implement a chunking strategy for large files
- [ ] Add a Reranker for better results
- [ ] Testing and validation
- [ ] Add more file types to ignore

## ❓ Troubleshooting

### Container Issues
- **Ollama not starting**: Check if port 11434 is available
- **Permission errors**: Ensure proper volume mounts
- **Model not found**: The startup script automatically pulls the model

### Performance Issues
- **Slow initial indexing**: Normal for large codebases, check logs for progress
- **High memory usage**: Adjust batch size in configuration
- **Slow searches**: Ensure embeddings are cached

### General Issues
- �� Embedding errors? Ensure Ollama is running with `nomic-embed-text`
- ⏱️ Initial search may be slow for large codebases
- 💨 Subsequent searches are faster thanks to caching

## 📝 Logs

The server provides detailed logging:
- **INFO**: General operations and status
- **DEBUG**: Detailed file processing information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors

View logs with:
```bash
# Docker
docker-compose logs -f

# Local
tail -f *.log
```
