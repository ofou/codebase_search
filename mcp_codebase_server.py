import asyncio
import json
import os
import hashlib
import glob
import ollama
import numpy as np
from typing import Optional, List, Dict, Any, Set
from mcp.server.fastmcp import FastMCP
import logging
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import queue
import signal
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("codebase_search_mcp")

# Configuration
EMBEDDING_MODEL = "nomic-embed-text"
TOP_N_RESULTS = 20
USE_FALLBACK = False
# File extensions to ignore (binary files, large data files, etc.)
IGNORE_EXTENSIONS = [
    ".pyc", ".jpg", ".png", ".gif", ".pdf", ".zip", ".gz", ".class", ".jar", ".*",
    ".bin", ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".lib",
    ".git", ".cache", ".kilocode", "__pycache__", "node_modules", ".venv"
]
# Max file size in MB to process
MAX_FILE_SIZE_MB = 5
# Enable embedding caching for better performance
CACHE_EMBEDDINGS = True
# Directory for persistent embedding cache
CACHE_DIR = "./.cache"
# Background indexing settings
ENABLE_BACKGROUND_INDEXING = True
ENABLE_FILE_WATCHING = True
INDEX_UPDATE_DELAY = 2.0  # seconds to wait after file change before re-indexing

# Ensure cache directory exists
if CACHE_EMBEDDINGS:
    os.makedirs(CACHE_DIR, exist_ok=True)

# Global state
embedding_cache: Dict[str, list] = {}
file_embeddings: Dict[str, list] = {}
indexed_files: Set[str] = set()
file_update_queue = queue.Queue()
indexing_lock = threading.Lock()
observer = None
stop_event = threading.Event()


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events for automatic re-indexing."""
    
    def __init__(self, watch_directories: List[str]):
        super().__init__()
        self.watch_directories = watch_directories
        self.last_modified = {}
    
    def on_created(self, event):
        if not event.is_directory and self._should_watch_file(event.src_path):
            self._queue_file_update(event.src_path, "created")
    
    def on_modified(self, event):
        if not event.is_directory and self._should_watch_file(event.src_path):
            self._queue_file_update(event.src_path, "modified")
    
    def on_deleted(self, event):
        if not event.is_directory and self._should_watch_file(event.src_path):
            self._queue_file_update(event.src_path, "deleted")
    
    def _should_watch_file(self, filepath: str) -> bool:
        """Check if file should be watched and processed."""
        if not should_process_file(filepath):
            return False
        
        # Check if file is in one of our watch directories
        for watch_dir in self.watch_directories:
            if filepath.startswith(os.path.abspath(watch_dir)):
                return True
        return False
    
    def _queue_file_update(self, filepath: str, event_type: str):
        """Queue a file update for background processing."""
        try:
            file_update_queue.put((filepath, event_type, time.time()))
            logger.debug(f"Queued file update: {event_type} - {filepath}")
        except Exception as e:
            logger.error(f"Error queuing file update: {e}")


async def get_embedding(text: str) -> list[float] | None:
    """Generates embedding for the given text using Ollama."""
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
    """Calculates cosine similarity between two vectors."""
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def should_process_file(filepath: str) -> bool:
    """Determines if a file should be processed based on extension and size."""
    # Check extension
    ext = os.path.splitext(filepath)[1].lower()
    if ext in IGNORE_EXTENSIONS:
        return False
    
    # Check if it's a hidden file or directory
    basename = os.path.basename(filepath)
    if basename.startswith('.'):
        return False
    
    # Check if it's in an ignored directory
    for ignored in ['.git', '.cache', '.kilocode', '__pycache__', 'node_modules', '.venv']:
        if ignored in filepath:
            return False

    # Check file size
    try:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                logger.debug(f"Skipping large file: {filepath} ({size_mb:.2f} MB)")
                return False
    except Exception as e:
        logger.warning(f"Error checking file size for {filepath}: {e}")
        return False

    return True


async def find_all_files(search_dirs):
    """Find all files in the specified directories."""
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
            continue

    return all_files


async def index_file(filepath: str) -> bool:
    """Index a single file and store its embedding."""
    try:
        if not should_process_file(filepath):
            return False
        
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        if not content.strip():  # Skip empty files
            return False

        file_embedding = await get_embedding(content)
        if file_embedding:
            with indexing_lock:
                file_embeddings[filepath] = file_embedding
                indexed_files.add(filepath)
            logger.debug(f"Indexed file: {filepath}")
            return True
    except Exception as e:
        logger.warning(f"Error indexing file {filepath}: {e}")
    
    return False


async def remove_file_from_index(filepath: str):
    """Remove a file from the index."""
    with indexing_lock:
        if filepath in file_embeddings:
            del file_embeddings[filepath]
        if filepath in indexed_files:
            indexed_files.remove(filepath)
    logger.debug(f"Removed file from index: {filepath}")


async def background_indexer():
    """Background task that processes file updates and maintains the index."""
    pending_updates = {}
    
    while not stop_event.is_set():
        try:
            # Process file update queue
            while not file_update_queue.empty():
                filepath, event_type, timestamp = file_update_queue.get_nowait()
                pending_updates[filepath] = (event_type, timestamp)
            
            # Process pending updates after delay
            current_time = time.time()
            to_process = []
            
            for filepath, (event_type, timestamp) in pending_updates.items():
                if current_time - timestamp >= INDEX_UPDATE_DELAY:
                    to_process.append((filepath, event_type))
                    del pending_updates[filepath]
            
            # Process the updates
            for filepath, event_type in to_process:
                if event_type == "deleted":
                    await remove_file_from_index(filepath)
                elif event_type in ["created", "modified"]:
                    await index_file(filepath)
            
            await asyncio.sleep(1)  # Check every second
            
        except Exception as e:
            logger.error(f"Error in background indexer: {e}")
            await asyncio.sleep(5)


async def initial_indexing(search_dirs: List[str]):
    """Perform initial indexing of all files."""
    logger.info("Starting initial indexing...")
    start_time = time.time()
    
    all_files = await find_all_files(search_dirs)
    logger.info(f"Found {len(all_files)} files to index")
    
    # Process files in batches
    batch_size = 10
    indexed_count = 0
    
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i + batch_size]
        logger.info(f"Indexing batch {i // batch_size + 1}/{(len(all_files) + batch_size - 1) // batch_size}")
        
        tasks = [index_file(filepath) for filepath in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        indexed_count += sum(1 for result in results if result is True)
    
    elapsed = time.time() - start_time
    logger.info(f"Initial indexing completed in {elapsed:.2f} seconds. Indexed {indexed_count} files.")


def start_file_watcher(search_dirs: List[str]):
    """Start the file system watcher."""
    global observer
    
    if not ENABLE_FILE_WATCHING:
        return
    
    observer = Observer()
    event_handler = FileChangeHandler(search_dirs)
    
    for directory in search_dirs:
        expanded_dir = os.path.abspath(os.path.expanduser(directory))
        if os.path.exists(expanded_dir):
            observer.schedule(event_handler, expanded_dir, recursive=True)
            logger.info(f"Watching directory: {expanded_dir}")
    
    observer.start()
    logger.info("File watcher started")


def stop_file_watcher():
    """Stop the file system watcher."""
    global observer
    if observer:
        observer.stop()
        observer.join()
        logger.info("File watcher stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal, cleaning up...")
    stop_event.set()
    stop_file_watcher()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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

    # Handle target directories
    if target_directories:
        search_dirs = target_directories
    else:
        # Default to current directory and parent directory if not specified
        search_dirs = [".", ".."]

    logger.info(f"Target directories: {search_dirs}")

    # Get query embedding
    query_embedding = await get_embedding(query)
    if not query_embedding:
        return json.dumps(
            {
                "error": "Failed to get query embedding. Please make sure Ollama is running and the nomic-embed-text model is available."
            },
            indent=2,
        )

    # Use indexed files if available, otherwise fall back to scanning
    with indexing_lock:
        if file_embeddings:
            available_files = list(file_embeddings.keys())
            logger.info(f"Using {len(available_files)} indexed files")
        else:
            # Fall back to scanning files
            available_files = await find_all_files(search_dirs)
            logger.info(f"Scanned {len(available_files)} files")

    if not available_files:
        return json.dumps(
            {"error": "No files found in specified directories."}, indent=2
        )

    # Calculate similarities for indexed files
    similarities = {}
    with indexing_lock:
        for filepath in available_files:
            if filepath in file_embeddings:
                similarity = cosine_similarity(query_embedding, file_embeddings[filepath])
                similarities[filepath] = similarity

    if not similarities:
        return json.dumps(
            {
                "error": "No file embeddings available. Please wait for indexing to complete."
            },
            indent=2,
        )

    # Get top N results
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


@mcp.tool()
async def get_index_status() -> str:
    """
    Get the current status of the file index.
    
    Returns:
        JSON string with index status information.
    """
    with indexing_lock:
        status = {
            "indexed_files_count": len(indexed_files),
            "file_embeddings_count": len(file_embeddings),
            "cache_size": len(embedding_cache),
            "indexed_files": list(indexed_files)[:10],  # First 10 files
            "background_indexing_enabled": ENABLE_BACKGROUND_INDEXING,
            "file_watching_enabled": ENABLE_FILE_WATCHING,
        }
    
    return json.dumps(status, indent=2)


async def main():
    """Main function to start the MCP server with background tasks."""
    try:
        # Test Ollama connection
        ollama.list()
        logger.info(f"Ollama is running. Ensure '{EMBEDDING_MODEL}' is pulled.")
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return

    # Default search directories
    search_dirs = [".", ".."]
    
    # Start file watcher
    start_file_watcher(search_dirs)
    
    # Start background tasks
    background_tasks = []
    
    if ENABLE_BACKGROUND_INDEXING:
        # Start initial indexing
        background_tasks.append(asyncio.create_task(initial_indexing(search_dirs)))
        # Start background indexer
        background_tasks.append(asyncio.create_task(background_indexer()))
    
    # Run the MCP server
    logger.info("Starting MCP server with SSE transport on port 8086")
    
    try:
        await mcp.run_async()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Clean up
        stop_event.set()
        stop_file_watcher()
        for task in background_tasks:
            task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
