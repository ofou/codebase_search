import asyncio
import json
import os
import glob
import ollama
import numpy as np
from typing import Optional, List
from mcp.server.fastmcp import FastMCP

# Constants
EMBEDDING_MODEL = "nomic-embed-text"
TOP_N_RESULTS = 20
USE_FALLBACK = False  # Changed to False to require Ollama embeddings


async def get_embedding(text: str) -> list[float] | None:
    """Generates embedding for the given text using Ollama."""
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response.get("embedding")
    except Exception as e:
        print(f"Error getting embedding from Ollama: {e}")
        return None


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


async def find_all_files(search_dirs):
    """Find all files in the specified directories."""
    all_files = []
    for dir_pattern in search_dirs:
        # Expand user home directory if needed
        expanded_pattern = os.path.expanduser(dir_pattern)
        # Use recursive glob to find all files
        pattern = os.path.join(expanded_pattern, "**", "*")
        try:
            # Find all files recursively
            found_paths = glob.glob(pattern, recursive=True)
            # Filter out directories, keep only files
            all_files.extend([p for p in found_paths if os.path.isfile(p)])
        except Exception as e:
            print(f"Error scanning directory {dir_pattern}: {e}")
            continue  # Skip problematic directories

    return all_files


# Create a FastMCP server instance with SSE transport settings
mcp = FastMCP("CodebaseSearch", sse_port=8086)


@mcp.tool()
async def codebase_search(
    query: str,
    target_directories: Optional[List[str]] = None,
    explanation: Optional[str] = None,
) -> str:
    """
    Finds snippets of code from the codebase most relevant to the search query.

    Args:
        query: The semantic search query.
        target_directories: Optional list of glob patterns for directories to search within.
        explanation: One sentence explanation as to why this tool is being used, and how it contributes to the goal.

    Returns:
        JSON string with search results.
    """
    print(f"Searching codebase for: '{query}' using {EMBEDDING_MODEL}")
    if explanation:
        print(f"Explanation: {explanation}")
    search_dirs = target_directories if target_directories else ["."]
    print(f"Target directories: {search_dirs}")

    # 1. Find all files
    all_files = await find_all_files(search_dirs)

    if not all_files:
        return json.dumps(
            {"error": "No files found in specified directories."}, indent=2
        )

    print(f"Found {len(all_files)} files.")

    # 2. Get query embedding
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return json.dumps(
            {
                "error": "Failed to get query embedding. Please make sure Ollama is running and the nomic-embed-text model is available."
            },
            indent=2,
        )

    # 3. Process files and calculate similarities
    file_embeddings = {}
    print("Generating embeddings for files...")
    for filepath in all_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if not content.strip():  # Skip empty files
                continue

            file_embedding = await get_embedding(content)
            if file_embedding:
                file_embeddings[filepath] = file_embedding
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    if not file_embeddings:
        return json.dumps(
            {
                "error": "Failed to generate embeddings for any files. Please check if Ollama is running properly."
            },
            indent=2,
        )

    print(f"Generated embeddings for {len(file_embeddings)} files.")

    # 4. Calculate similarities and rank
    similarities = {}
    for filepath, embedding in file_embeddings.items():
        similarity = cosine_similarity(query_embedding, embedding)
        similarities[filepath] = similarity

    # 5. Get top N results
    sorted_files = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_results = [
        {"file": filepath, "similarity": score}
        for filepath, score in sorted_files[:TOP_N_RESULTS]
    ]

    print(f"Top {len(top_results)} results: {top_results}")
    return json.dumps(top_results, indent=2)


@mcp.tool()
async def folder_similarity_search(
    folder_path: str,
    target_directories: Optional[List[str]] = None,
    explanation: Optional[str] = None,
) -> str:
    """
    Finds files that are semantically similar to the contents of a specified folder.

    Args:
        folder_path: Path to the folder containing files to use as the query.
        target_directories: Optional list of glob patterns for directories to search within.
        explanation: One sentence explanation as to why this tool is being used, and how it contributes to the goal.

    Returns:
        JSON string with search results.
    """
    print(f"Finding files similar to folder: '{folder_path}' using {EMBEDDING_MODEL}")
    if explanation:
        print(f"Explanation: {explanation}")

    # Ensure folder exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return json.dumps(
            {"error": f"Folder '{folder_path}' does not exist or is not a directory."},
            indent=2,
        )

    # Find all files in the query folder
    folder_files = await find_all_files([folder_path])

    if not folder_files:
        return json.dumps(
            {"error": f"No files found in the query folder '{folder_path}'."},
            indent=2,
        )

    print(f"Found {len(folder_files)} files in query folder.")

    # Create a combined embedding for the folder by averaging embeddings of all files
    folder_file_embeddings = []
    for filepath in folder_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if not content.strip():  # Skip empty files
                continue

            file_embedding = await get_embedding(content)
            if file_embedding:
                folder_file_embeddings.append(file_embedding)
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    if not folder_file_embeddings:
        return json.dumps(
            {
                "error": f"Failed to generate embeddings for files in the folder '{folder_path}'."
            },
            indent=2,
        )

    # Average the embeddings to get a single embedding for the folder
    folder_embedding = np.mean(np.array(folder_file_embeddings), axis=0).tolist()

    # Find target files to compare with
    search_dirs = target_directories if target_directories else ["."]
    print(f"Target directories: {search_dirs}")

    # Exclude the query folder from the search
    search_dirs = [dir_path for dir_path in search_dirs if dir_path != folder_path]

    # Find all files
    all_files = await find_all_files(search_dirs)

    # Filter out files from the query folder
    all_files = [f for f in all_files if not f.startswith(folder_path)]

    if not all_files:
        return json.dumps(
            {
                "error": "No files found in specified directories (excluding query folder)."
            },
            indent=2,
        )

    print(f"Found {len(all_files)} files to compare with.")

    # Process target files and calculate similarities
    file_embeddings = {}
    print("Generating embeddings for target files...")
    for filepath in all_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if not content.strip():  # Skip empty files
                continue

            file_embedding = await get_embedding(content)
            if file_embedding:
                file_embeddings[filepath] = file_embedding
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    if not file_embeddings:
        return json.dumps(
            {"error": "Failed to generate embeddings for any target files."}, indent=2
        )

    print(f"Generated embeddings for {len(file_embeddings)} target files.")

    # Calculate similarities and rank
    similarities = {}
    for filepath, embedding in file_embeddings.items():
        similarity = cosine_similarity(folder_embedding, embedding)
        similarities[filepath] = similarity

    # Get top N results
    sorted_files = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_results = [
        {"file": filepath, "similarity": score}
        for filepath, score in sorted_files[:TOP_N_RESULTS]
    ]

    print(f"Top {len(top_results)} results: {top_results}")
    return json.dumps(top_results, indent=2)


if __name__ == "__main__":
    # Check if Ollama is running and model is available (required now, not optional)
    try:
        ollama.list()
        print(f"Ollama is running. Ensure '{EMBEDDING_MODEL}' is pulled.")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print(
            "Please ensure the Ollama service is running and the nomic-embed-text model is installed."
        )
        print("You can install it with: ollama pull nomic-embed-text")
        exit(1)  # Exit if Ollama is not available since fallback is removed

    # Run the MCP server with debugging info and SSE transport
    print("Starting MCP server with SSE transport on port 8086")
    mcp.run()
