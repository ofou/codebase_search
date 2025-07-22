#!/bin/bash

# Start script for the codebase search MCP server

echo "🚀 Starting Codebase Search MCP Server..."

# Check if Ollama is running
echo "🔍 Checking Ollama connection..."
if ! ollama list > /dev/null 2>&1; then
    echo "❌ Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Pull the required model if not already present
echo "📥 Ensuring nomic-embed-text model is available..."
ollama pull nomic-embed-text

# Start the MCP server
echo "🔧 Starting MCP server..."
python3 mcp_codebase_server.py
