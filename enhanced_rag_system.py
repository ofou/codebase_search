#!/usr/bin/env python3
"""
Enhanced RAG (Retrieval-Augmented Generation) system for codebase search.
"""

import asyncio
import json
import os
import hashlib
import re
import ast
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger("enhanced_rag")

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    filepath: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'import', 'comment', 'code'
    language: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    docstring: Optional[str] = None
    imports: List[str] = None
    dependencies: List[str] = None

@dataclass
class SearchResult:
    """Represents a search result with enhanced metadata."""
    filepath: str
    relative_path: str
    similarity_score: float
    chunks: List[CodeChunk]
    file_summary: str
    language: str
    file_size: int
    last_modified: float
    function_count: int
    class_count: int

class CodeChunker:
    """Intelligent code chunking with semantic understanding."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk_file(self, filepath: str, content: str) -> List[CodeChunk]:
        """Chunk a file into semantic units."""
        language = self._detect_language(filepath)
        chunks = []
        
        if language == "python":
            chunks = self._chunk_python(content, filepath)
        elif language in ["javascript", "typescript"]:
            chunks = self._chunk_javascript(content, filepath)
        elif language in ["java", "csharp"]:
            chunks = self._chunk_java_like(content, filepath)
        else:
            chunks = self._chunk_generic(content, filepath, language)
        
        return chunks
    
    def _detect_language(self, filepath: str) -> str:
        """Detect programming language from file extension."""
        ext = os.path.splitext(filepath)[1].lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.sh': 'bash',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.conf': 'conf'
        }
        return language_map.get(ext, 'text')
    
    def _chunk_python(self, content: str, filepath: str) -> List[CodeChunk]:
        """Chunk Python code with AST analysis."""
        chunks = []
        lines = content.split('\n')
        
        try:
            tree = ast.parse(content)
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._create_function_chunk(node, lines, filepath)
                    chunks.append(chunk)
                elif isinstance(node, ast.ClassDef):
                    chunk = self._create_class_chunk(node, lines, filepath)
                    chunks.append(chunk)
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    chunk = self._create_import_chunk(node, lines, filepath)
                    chunks.append(chunk)
            
            # Add remaining code as generic chunks
            remaining_lines = self._get_remaining_lines(tree, lines)
            if remaining_lines:
                chunk = self._create_generic_chunk(remaining_lines, filepath, 'code')
                chunks.append(chunk)
                
        except SyntaxError:
            # Fallback to generic chunking for files with syntax errors
            chunks = self._chunk_generic(content, filepath, 'python')
        
        return chunks
    
    def _create_function_chunk(self, node: ast.FunctionDef, lines: List[str], filepath: str) -> CodeChunk:
        """Create a chunk for a function definition."""
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        
        content = '\n'.join(lines[start_line:end_line])
        docstring = ast.get_docstring(node)
        
        return CodeChunk(
            content=content,
            filepath=filepath,
            start_line=start_line + 1,
            end_line=end_line,
            chunk_type='function',
            language='python',
            function_name=node.name,
            docstring=docstring,
            imports=[],
            dependencies=[]
        )
    
    def _create_class_chunk(self, node: ast.ClassDef, lines: List[str], filepath: str) -> CodeChunk:
        """Create a chunk for a class definition."""
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        
        content = '\n'.join(lines[start_line:end_line])
        docstring = ast.get_docstring(node)
        
        return CodeChunk(
            content=content,
            filepath=filepath,
            start_line=start_line + 1,
            end_line=end_line,
            chunk_type='class',
            language='python',
            class_name=node.name,
            docstring=docstring,
            imports=[],
            dependencies=[]
        )
    
    def _create_import_chunk(self, node: ast.AST, lines: List[str], filepath: str) -> CodeChunk:
        """Create a chunk for import statements."""
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        
        content = '\n'.join(lines[start_line:end_line])
        
        return CodeChunk(
            content=content,
            filepath=filepath,
            start_line=start_line + 1,
            end_line=end_line,
            chunk_type='import',
            language='python',
            imports=[],
            dependencies=[]
        )
    
    def _get_remaining_lines(self, tree: ast.AST, lines: List[str]) -> List[str]:
        """Get lines that are not part of functions or classes."""
        # This is a simplified implementation
        # In a full implementation, you'd track all covered lines
        return lines
    
    def _chunk_javascript(self, content: str, filepath: str) -> List[CodeChunk]:
        """Chunk JavaScript/TypeScript code."""
        chunks = []
        lines = content.split('\n')
        
        # Simple regex-based chunking for JS/TS
        function_pattern = r'(?:function\s+(\w+)|(\w+)\s*[:=]\s*(?:function|\([^)]*\)\s*=>))'
        class_pattern = r'class\s+(\w+)'
        
        # Find functions
        for i, line in enumerate(lines):
            if re.search(function_pattern, line):
                chunk = self._create_generic_chunk([line], filepath, 'function', start_line=i+1)
                chunks.append(chunk)
        
        # Find classes
        for i, line in enumerate(lines):
            if re.search(class_pattern, line):
                chunk = self._create_generic_chunk([line], filepath, 'class', start_line=i+1)
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_java_like(self, content: str, filepath: str) -> List[CodeChunk]:
        """Chunk Java-like languages."""
        chunks = []
        lines = content.split('\n')
        
        # Simple regex-based chunking
        function_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:<[^>]+>\s+)?\w+\s+\w+\s*\([^)]*\)'
        class_pattern = r'(?:public\s+)?class\s+(\w+)'
        
        for i, line in enumerate(lines):
            if re.search(function_pattern, line):
                chunk = self._create_generic_chunk([line], filepath, 'function', start_line=i+1)
                chunks.append(chunk)
            elif re.search(class_pattern, line):
                chunk = self._create_generic_chunk([line], filepath, 'class', start_line=i+1)
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_generic(self, content: str, filepath: str, language: str, start_line: int = 1) -> List[CodeChunk]:
        """Generic chunking for any file type."""
        lines = content.split('\n')
        chunks = []
        
        # Split into chunks of max_chunk_size lines
        for i in range(0, len(lines), self.max_chunk_size):
            chunk_lines = lines[i:i + self.max_chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            chunk = CodeChunk(
                content=chunk_content,
                filepath=filepath,
                start_line=start_line + i,
                end_line=start_line + i + len(chunk_lines),
                chunk_type='code',
                language=language,
                imports=[],
                dependencies=[]
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_generic_chunk(self, lines: List[str], filepath: str, chunk_type: str, start_line: int = 1) -> CodeChunk:
        """Create a generic code chunk."""
        content = '\n'.join(lines)
        
        return CodeChunk(
            content=content,
            filepath=filepath,
            start_line=start_line,
            end_line=start_line + len(lines),
            chunk_type=chunk_type,
            language=self._detect_language(filepath),
            imports=[],
            dependencies=[]
        )

class EnhancedRAGSystem:
    """Enhanced RAG system with intelligent chunking and retrieval."""
    
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        self.embedding_model = embedding_model
        self.chunker = CodeChunker()
        self.chunk_embeddings: Dict[str, List[float]] = {}
        self.chunks: Dict[str, CodeChunk] = {}
        self.file_chunks: Dict[str, List[str]] = defaultdict(list)
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def index_file(self, filepath: str, content: str) -> bool:
        """Index a file with enhanced chunking."""
        try:
            # Create chunks
            chunks = self.chunker.chunk_file(filepath, content)
            
            # Store chunks
            chunk_ids = []
            for chunk in chunks:
                chunk_id = self._generate_chunk_id(filepath, chunk)
                self.chunks[chunk_id] = chunk
                chunk_ids.append(chunk_id)
                
                # Store metadata
                self.chunk_metadata[chunk_id] = {
                    'filepath': filepath,
                    'chunk_type': chunk.chunk_type,
                    'language': chunk.language,
                    'function_name': chunk.function_name,
                    'class_name': chunk.class_name,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'has_docstring': bool(chunk.docstring)
                }
            
            self.file_chunks[filepath] = chunk_ids
            logger.info(f"Indexed {len(chunks)} chunks from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing file {filepath}: {e}")
            return False
    
    def _generate_chunk_id(self, filepath: str, chunk: CodeChunk) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()[:16]
        return f"{filepath}:{chunk.start_line}:{chunk.end_line}:{content_hash}"
    
    async def search(self, query: str, top_k: int = 10, 
                    chunk_types: Optional[List[str]] = None,
                    languages: Optional[List[str]] = None,
                    min_similarity: float = 0.1) -> List[SearchResult]:
        """Enhanced search with filtering and ranking."""
        # This would integrate with the embedding system
        # For now, return a mock result structure
        return []
    
    def get_file_summary(self, filepath: str) -> str:
        """Generate a summary of a file based on its chunks."""
        if filepath not in self.file_chunks:
            return "File not indexed"
        
        chunk_ids = self.file_chunks[filepath]
        chunks = [self.chunks[chunk_id] for chunk_id in chunk_ids]
        
        # Count different types
        function_count = sum(1 for c in chunks if c.chunk_type == 'function')
        class_count = sum(1 for c in chunks if c.chunk_type == 'class')
        import_count = sum(1 for c in chunks if c.chunk_type == 'import')
        
        summary = f"File contains {function_count} functions, {class_count} classes, {import_count} import statements"
        
        # Add language info
        if chunks:
            summary += f" (Language: {chunks[0].language})"
        
        return summary
    
    def get_chunk_context(self, chunk_id: str, context_lines: int = 5) -> str:
        """Get context around a chunk."""
        if chunk_id not in self.chunks:
            return ""
        
        chunk = self.chunks[chunk_id]
        
        try:
            with open(chunk.filepath, 'r') as f:
                lines = f.readlines()
            
            start = max(0, chunk.start_line - context_lines - 1)
            end = min(len(lines), chunk.end_line + context_lines)
            
            context_lines = lines[start:end]
            return ''.join(context_lines)
            
        except Exception as e:
            logger.error(f"Error getting context for {chunk_id}: {e}")
            return chunk.content

# Example usage and testing
async def test_enhanced_rag():
    """Test the enhanced RAG system."""
    rag = EnhancedRAGSystem()
    
    # Test with a sample Python file
    sample_content = '''
import os
import json
from typing import List, Dict

class DataProcessor:
    """Process data from various sources."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data = []
    
    def process_file(self, filepath: str) -> List[Dict]:
        """Process a single file and return structured data."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return self._transform_data(data)
    
    def _transform_data(self, data: List[Dict]) -> List[Dict]:
        """Transform raw data into processed format."""
        return [item for item in data if self._is_valid(item)]
    
    def _is_valid(self, item: Dict) -> bool:
        """Check if an item is valid."""
        return bool(item and isinstance(item, dict))
'''
    
    # Index the sample content
    success = await rag.index_file("sample.py", sample_content)
    print(f"Indexing success: {success}")
    
    # Get file summary
    summary = rag.get_file_summary("sample.py")
    print(f"File summary: {summary}")
    
    # Show chunks
    for chunk_id, chunk in rag.chunks.items():
        print(f"\nChunk: {chunk_id}")
        print(f"Type: {chunk.chunk_type}")
        print(f"Function: {chunk.function_name}")
        print(f"Class: {chunk.class_name}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Content preview: {chunk.content[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_enhanced_rag())
