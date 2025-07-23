import asyncio
import json
import os
import hashlib
import glob
import ollama
import numpy as np
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("codebase_search_mcp")

# Configuration
EMBEDDING_MODEL = "nomic-embed-text"
TOP_N_RESULTS = 20
USE_FALLBACK = False  # Changed to False to require Ollama embeddings
# File extensions to ignore (binary files, large data files, etc.)
IGNORE_EXTENSIONS = [
    ".pyc",
    ".jpg",
    ".png",
    ".gif",
    ".pdf",
    ".zip",
    ".gz",
    ".class",
    ".jar",
    ".*",
]
# Max file size in MB to process (to avoid very large files)
MAX_FILE_SIZE_MB = 5
# Enable embedding caching for better performance (in-memory and on-disk)
CACHE_EMBEDDINGS = True
# Directory for persistent embedding cache
CACHE_DIR = "./.cache"

# Ensure cache directory exists
if CACHE_EMBEDDINGS:
    os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory cache for embeddings
embedding_cache: Dict[str, list] = {}


async def get_embedding(text: str) -> list[float] | None:
    """Generates embedding for the given text using Ollama.

    Args:
        text: The text to generate an embedding for

    Returns:
        A list of floats representing the embedding, or None if there was an error
    """
    # Check cache first if enabled
    if CACHE_EMBEDDINGS:
        # In-memory cache lookup
        if text in embedding_cache:
            return embedding_cache[text]
        # On-disk cache lookup
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    embedding = json.load(f)
                embedding_cache[text] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_path}: {e}")

    try:
        start_time = time.time()
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embedding = response.get("embedding")

        # Cache the embedding if enabled
        if CACHE_EMBEDDINGS and embedding:
            embedding_cache[text] = embedding
            # Persist to on-disk cache
            try:
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                cache_path = os.path.join(CACHE_DIR, f"{key}.json")
                with open(cache_path, "w", encoding="utf-8") as cf:
                    json.dump(embedding, cf)
            except Exception as e:
                logger.warning(f"Failed to write cache file {cache_path}: {e}")

        elapsed = time.time() - start_time
        logger.debug(f"Embedding generation took {elapsed:.2f} seconds")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding from Ollama: {e}")
        return None


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculates cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity score between 0 and 1
    """
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def should_process_file(filepath: str) -> bool:
    """Determines if a file should be processed based on extension and size.

    Args:
        filepath: Path to the file

    Returns:
        Boolean indicating whether the file should be processed
    """
    # Check extension
    ext = os.path.splitext(filepath)[1].lower()
    if ext in IGNORE_EXTENSIONS:
        return False

    # Check file size
    try:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            logger.info(f"Skipping large file: {filepath} ({size_mb:.2f} MB)")
            return False
    except Exception as e:
        logger.warning(f"Error checking file size for {filepath}: {e}")
        return False

    return True


async def find_all_files(search_dirs):
    """Find all files in the specified directories.

    Args:
        search_dirs: List of directory patterns to search

    Returns:
        List of file paths found
    """
    all_files = []
    for dir_pattern in search_dirs:
        # Expand user home directory if needed
        expanded_pattern = os.path.expanduser(dir_pattern)
        # Convert to absolute path if it's a relative path
        expanded_pattern = os.path.abspath(expanded_pattern)

        if not os.path.exists(expanded_pattern):
            logger.warning(f"Directory does not exist: {expanded_pattern}")
            continue

        # Use recursive glob to find all files
        pattern = os.path.join(expanded_pattern, "**", "*")
        try:
            # Find all files recursively
            found_paths = glob.glob(pattern, recursive=True)
            # Filter out directories, keep only files
            files = [
                p for p in found_paths if os.path.isfile(p) and should_process_file(p)
            ]
            all_files.extend(files)
            logger.info(
                f"Found {len(files)} processable files in {dir_pattern} ({expanded_pattern})"
            )
        except Exception as e:
            logger.error(f"Error scanning directory {dir_pattern}: {e}")
            continue  # Skip problematic directories

    return all_files


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
    start_time = time.time()
    logger.info(f"Searching codebase for: '{query}' using {EMBEDDING_MODEL}")
    if explanation:
        logger.info(f"Explanation: {explanation}")

    # Fix: Handle target_directories properly, including the ability to search parent directories
    if target_directories:
        search_dirs = target_directories
    else:
        # Default to current directory and parent directory if not specified
        search_dirs = [".", ".."]

    logger.info(f"Target directories: {search_dirs}")

    # 1. Find all files
    all_files = await find_all_files(search_dirs)

    if not all_files:
        return json.dumps(
            {"error": "No files found in specified directories."}, indent=2
        )

    logger.info(f"Found {len(all_files)} files to process.")

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
    logger.info("Generating embeddings for files...")

    # Process files in smaller batches to avoid memory issues
    batch_size = 10
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i : i + batch_size]
        logger.debug(
            f"Processing batch {i // batch_size + 1}/{len(all_files) // batch_size + 1}"
        )

        for filepath in batch:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if not content.strip():  # Skip empty files
                    continue

                file_embedding = await get_embedding(content)
                if file_embedding:
                    file_embeddings[filepath] = file_embedding
            except Exception as e:
                logger.warning(f"Error processing file {filepath}: {e}")

    if not file_embeddings:
        return json.dumps(
            {
                "error": "Failed to generate embeddings for any files. Please check if Ollama is running properly."
            },
            indent=2,
        )

    logger.info(f"Generated embeddings for {len(file_embeddings)} files.")

    # 4. Calculate similarities and rank
    similarities = {}
    for filepath, embedding in file_embeddings.items():
        similarity = cosine_similarity(query_embedding, embedding)
        similarities[filepath] = similarity

    # 5. Get top N results
    sorted_files = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_results = [
        {
            "file": filepath,
            "similarity": score,
            "relative_path": os.path.relpath(filepath, os.getcwd()),
        }
        for filepath, score in sorted_files[:TOP_N_RESULTS]
    ]

    elapsed_time = time.time() - start_time
    logger.info(f"Search completed in {elapsed_time:.2f} seconds")
    logger.info(f"Top {len(top_results)} results found")

    return json.dumps(top_results, indent=2)


if __name__ == "__main__":
    try:
        ollama.list()
        logger.info(f"Ollama is running. Ensure '{EMBEDDING_MODEL}' is pulled.")
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        exit(1)

    # Run the MCP server with debugging info and SSE transport
    logger.info("Starting MCP server with SSE transport on port 8086")
    mcp.run()
