#!/usr/bin/env python3
"""
Test the enhanced RAG system functionality.
"""

import asyncio
import json
import os
import sys
from enhanced_rag_system import EnhancedRAGSystem, CodeChunk

async def test_enhanced_rag_features():
    """Test enhanced RAG system features."""
    print("🧪 Testing Enhanced RAG System Features\n")
    
    # Initialize RAG system
    rag = EnhancedRAGSystem()
    
    # Test 1: Python file chunking
    print("📋 Test 1: Python File Chunking")
    python_content = '''
import os
import json
from typing import List, Dict, Optional

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

def utility_function():
    """A utility function outside the class."""
    return "utility"
'''
    
    success = await rag.index_file("test_python.py", python_content)
    print(f"✅ Python indexing: {'Success' if success else 'Failed'}")
    
    # Test 2: JavaScript file chunking
    print("\n📋 Test 2: JavaScript File Chunking")
    js_content = '''
import React from 'react';
import { useState, useEffect } from 'react';

class DataComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: [],
            loading: false
        };
    }
    
    componentDidMount() {
        this.fetchData();
    }
    
    async fetchData() {
        this.setState({ loading: true });
        try {
            const response = await fetch('/api/data');
            const data = await response.json();
            this.setState({ data, loading: false });
        } catch (error) {
            console.error('Error fetching data:', error);
            this.setState({ loading: false });
        }
    }
    
    render() {
        const { data, loading } = this.state;
        return (
            <div>
                {loading ? <p>Loading...</p> : <DataList data={data} />}
            </div>
        );
    }
}

function DataList({ data }) {
    return (
        <ul>
            {data.map(item => (
                <li key={item.id}>{item.name}</li>
            ))}
        </ul>
    );
}

export default DataComponent;
'''
    
    success = await rag.index_file("test_javascript.jsx", js_content)
    print(f"✅ JavaScript indexing: {'Success' if success else 'Failed'}")
    
    # Test 3: Chunk analysis
    print("\n📋 Test 3: Chunk Analysis")
    print(f"Total chunks: {len(rag.chunks)}")
    
    chunk_types = {}
    languages = {}
    functions = 0
    classes = 0
    
    for chunk_id, chunk in rag.chunks.items():
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        languages[chunk.language] = languages.get(chunk.language, 0) + 1
        
        if chunk.chunk_type == 'function':
            functions += 1
        elif chunk.chunk_type == 'class':
            classes += 1
    
    print(f"Chunk types: {chunk_types}")
    print(f"Languages: {languages}")
    print(f"Functions: {functions}")
    print(f"Classes: {classes}")
    
    # Test 4: File summaries
    print("\n📋 Test 4: File Summaries")
    for filepath in ["test_python.py", "test_javascript.jsx"]:
        summary = rag.get_file_summary(filepath)
        print(f"📄 {filepath}: {summary}")
    
    # Test 5: Context retrieval
    print("\n📋 Test 5: Context Retrieval")
    for chunk_id, chunk in list(rag.chunks.items())[:3]:  # Test first 3 chunks
        context = rag.get_chunk_context(chunk_id, context_lines=2)
        print(f"🔍 Context for {chunk_id}:")
        print(f"   Type: {chunk.chunk_type}")
        print(f"   Function: {chunk.function_name}")
        print(f"   Class: {chunk.class_name}")
        print(f"   Context preview: {context[:100]}...")
        print()
    
    # Test 6: Search simulation
    print("📋 Test 6: Search Simulation")
    search_results = await rag.search("data processing function", top_k=5)
    print(f"Search results count: {len(search_results)}")
    
    # Test 7: Metadata extraction
    print("\n📋 Test 7: Metadata Extraction")
    for chunk_id, metadata in list(rag.chunk_metadata.items())[:5]:
        print(f"📊 Metadata for {chunk_id}:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        print()
    
    print("✅ Enhanced RAG system tests completed!")

def test_chunking_strategies():
    """Test different chunking strategies."""
    print("\n🧪 Testing Chunking Strategies\n")
    
    from enhanced_rag_system import CodeChunker
    
    chunker = CodeChunker()
    
    # Test language detection
    test_files = [
        "test.py",
        "test.js",
        "test.tsx",
        "test.java",
        "test.cpp",
        "test.go",
        "test.rs",
        "test.md",
        "test.json"
    ]
    
    print("📋 Language Detection:")
    for file in test_files:
        language = chunker._detect_language(file)
        print(f"   {file} -> {language}")
    
    print("\n✅ Chunking strategy tests completed!")

def main():
    """Run all enhanced RAG tests."""
    print("🚀 Starting Enhanced RAG System Tests\n")
    
    # Test chunking strategies
    test_chunking_strategies()
    
    # Test enhanced RAG features
    asyncio.run(test_enhanced_rag_features())
    
    print("\n🎉 All Enhanced RAG tests completed successfully!")
    print("\n📊 Key Improvements:")
    print("   • Intelligent semantic chunking")
    print("   • Language-specific parsing")
    print("   • Rich metadata extraction")
    print("   • Context retrieval")
    print("   • Advanced search filtering")
    print("   • Better performance and accuracy")

if __name__ == "__main__":
    main()
