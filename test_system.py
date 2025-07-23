#!/usr/bin/env python3
"""
Test script for the codebase search system.
"""

import asyncio
import json
import sys
import time
from mcp_codebase_server import (
    get_embedding, 
    cosine_similarity, 
    should_process_file,
    find_all_files
)

async def test_embeddings():
    """Test embedding generation."""
    print("🧪 Testing embedding generation...")
    
    test_text = "This is a test query for codebase search"
    embedding = await get_embedding(test_text)
    
    if embedding:
        print(f"✅ Embedding generated successfully (length: {len(embedding)})")
        return True
    else:
        print("❌ Failed to generate embedding")
        return False

def test_file_filtering():
    """Test file filtering logic."""
    print("🧪 Testing file filtering...")
    
    test_files = [
        "test.py",
        "test.pyc",
        "test.jpg",
        ".hidden_file",
        "node_modules/test.js",
        ".git/config"
    ]
    
    for filepath in test_files:
        should_process = should_process_file(filepath)
        status = "✅" if should_process else "❌"
        print(f"{status} {filepath}: {'process' if should_process else 'skip'}")
    
    return True

async def test_file_discovery():
    """Test file discovery."""
    print("🧪 Testing file discovery...")
    
    try:
        files = await find_all_files(["."])
        print(f"✅ Found {len(files)} files in current directory")
        return True
    except Exception as e:
        print(f"❌ File discovery failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting system tests...\n")
    
    tests = [
        ("Embedding Generation", test_embeddings),
        ("File Filtering", lambda: test_file_filtering()),
        ("File Discovery", test_file_discovery),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"📋 {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
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
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
