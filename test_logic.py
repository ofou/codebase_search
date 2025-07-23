#!/usr/bin/env python3
"""
Test the core logic without external dependencies.
"""

import os
import sys
import json
from typing import List

# Mock the external dependencies for testing
class MockOllama:
    def embeddings(self, model, prompt):
        # Return a mock embedding
        return {"embedding": [0.1] * 384}  # Mock embedding vector
    
    def list(self):
        return {"models": [{"name": "nomic-embed-text"}]}

# Mock numpy for testing
class MockNumpy:
    def array(self, data):
        return MockArray(data)
    
    def dot(self, a, b):
        return 0.5  # Mock similarity
    
    def linalg(self):
        return MockLinalg()

class MockArray:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

class MockLinalg:
    def norm(self, arr):
        return 1.0

# Mock the dependencies
sys.modules['ollama'] = MockOllama()
sys.modules['numpy'] = MockNumpy()

# Now import our functions
from mcp_codebase_server import should_process_file, find_all_files

def test_file_filtering():
    """Test file filtering logic."""
    print("🧪 Testing file filtering...")
    
    test_files = [
        "test.py",
        "test.pyc", 
        "test.jpg",
        ".hidden_file",
        "node_modules/test.js",
        ".git/config",
        "test.txt",
        "large_file.bin"
    ]
    
    results = []
    for filepath in test_files:
        should_process = should_process_file(filepath)
        status = "✅" if should_process else "❌"
        print(f"{status} {filepath}: {'process' if should_process else 'skip'}")
        results.append(should_process)
    
    # Should process: test.py, test.txt
    # Should skip: test.pyc, test.jpg, .hidden_file, node_modules/test.js, .git/config
    expected = [True, False, False, False, False, False, True, False]
    
    if results == expected:
        print("✅ File filtering test passed!")
        return True
    else:
        print(f"❌ File filtering test failed. Expected: {expected}, Got: {results}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("🧪 Testing configuration...")
    
    # Test that configuration variables are defined
    try:
        from mcp_codebase_server import (
            EMBEDDING_MODEL,
            TOP_N_RESULTS,
            IGNORE_EXTENSIONS,
            MAX_FILE_SIZE_MB,
            CACHE_EMBEDDINGS,
            ENABLE_BACKGROUND_INDEXING,
            ENABLE_FILE_WATCHING
        )
        
        print(f"✅ EMBEDDING_MODEL: {EMBEDDING_MODEL}")
        print(f"✅ TOP_N_RESULTS: {TOP_N_RESULTS}")
        print(f"✅ IGNORE_EXTENSIONS: {len(IGNORE_EXTENSIONS)} extensions")
        print(f"✅ MAX_FILE_SIZE_MB: {MAX_FILE_SIZE_MB}")
        print(f"✅ CACHE_EMBEDDINGS: {CACHE_EMBEDDINGS}")
        print(f"✅ ENABLE_BACKGROUND_INDEXING: {ENABLE_BACKGROUND_INDEXING}")
        print(f"✅ ENABLE_FILE_WATCHING: {ENABLE_FILE_WATCHING}")
        
        return True
    except ImportError as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_file_discovery():
    """Test file discovery logic."""
    print("🧪 Testing file discovery...")
    
    try:
        # Create a test file
        with open("test_file.py", "w") as f:
            f.write("# Test file for discovery")
        
        # Test finding files in current directory
        files = find_all_files(["."])
        
        # Clean up
        os.remove("test_file.py")
        
        if isinstance(files, list):
            print(f"✅ File discovery found {len(files)} files")
            return True
        else:
            print(f"❌ File discovery returned wrong type: {type(files)}")
            return False
            
    except Exception as e:
        print(f"❌ File discovery test failed: {e}")
        return False

def test_docker_files():
    """Test that Docker files are properly configured."""
    print("🧪 Testing Docker configuration...")
    
    required_files = [
        "Dockerfile",
        "docker-compose.yml", 
        "docker-compose.dev.yml",
        ".dockerignore",
        "start_server.sh",
        "health_check.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - missing")
            missing_files.append(file)
    
    if not missing_files:
        print("✅ All Docker files present!")
        return True
    else:
        print(f"❌ Missing files: {missing_files}")
        return False

def test_mcp_configuration():
    """Test MCP configuration."""
    print("🧪 Testing MCP configuration...")
    
    try:
        with open(".kilocode/mcp.json", "r") as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = ["mcpServers", "codebase-search", "tools"]
        for field in required_fields:
            if field in config:
                print(f"✅ {field}")
            else:
                print(f"❌ {field} - missing")
                return False
        
        # Check tools
        tools = config["mcpServers"]["codebase-search"]["tools"]
        expected_tools = ["codebase_search", "get_index_status"]
        
        for tool in expected_tools:
            if tool in tools:
                print(f"✅ Tool: {tool}")
            else:
                print(f"❌ Tool: {tool} - missing")
                return False
        
        print("✅ MCP configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MCP configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting logic tests...\n")
    
    tests = [
        ("File Filtering", test_file_filtering),
        ("Configuration", test_configuration),
        ("File Discovery", test_file_discovery),
        ("Docker Files", test_docker_files),
        ("MCP Configuration", test_mcp_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"📋 {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 Test Results:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All logic tests passed! Implementation is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
