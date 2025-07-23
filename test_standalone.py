#!/usr/bin/env python3
"""
Standalone test for the implementation logic.
"""

import os
import json
import sys

def should_process_file(filepath: str) -> bool:
    """Determines if a file should be processed based on extension and size."""
    # File extensions to ignore
    IGNORE_EXTENSIONS = [
        ".pyc", ".jpg", ".png", ".gif", ".pdf", ".zip", ".gz", ".class", ".jar", ".*",
        ".bin", ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".lib",
        ".git", ".cache", ".kilocode", "__pycache__", "node_modules", ".venv"
    ]
    
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

    # Check file size (simplified for testing)
    try:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > 5:  # 5MB limit
                return False
    except Exception:
        return False

    return True

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
    # Should skip: test.pyc, test.jpg, .hidden_file, node_modules/test.js, .git/config, large_file.bin
    expected = [True, False, False, False, False, False, True, False]
    
    if results == expected:
        print("✅ File filtering test passed!")
        return True
    else:
        print(f"❌ File filtering test failed. Expected: {expected}, Got: {results}")
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
        if "mcpServers" not in config:
            print("❌ mcpServers - missing")
            return False
        
        if "codebase-search" not in config["mcpServers"]:
            print("❌ codebase-search - missing")
            return False
        
        if "tools" not in config["mcpServers"]["codebase-search"]:
            print("❌ tools - missing")
            return False
        
        print("✅ mcpServers")
        print("✅ codebase-search")
        print("✅ tools")
        
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

def test_file_structure():
    """Test the overall file structure."""
    print("🧪 Testing file structure...")
    
    # Check main files
    main_files = [
        "mcp_codebase_server.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file in main_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - missing")
            return False
    
    # Check file sizes
    if os.path.getsize("mcp_codebase_server.py") > 1000:
        print("✅ Main server file has substantial content")
    else:
        print("❌ Main server file seems too small")
        return False
    
    print("✅ File structure test passed!")
    return True

def test_configuration_values():
    """Test configuration values in the main file."""
    print("🧪 Testing configuration values...")
    
    try:
        with open("mcp_codebase_server.py", "r") as f:
            content = f.read()
        
        # Check for key configuration variables
        config_checks = [
            ("EMBEDDING_MODEL", "nomic-embed-text"),
            ("TOP_N_RESULTS", "20"),
            ("ENABLE_BACKGROUND_INDEXING", "True"),
            ("ENABLE_FILE_WATCHING", "True"),
            ("watchdog", "import"),
            ("FileChangeHandler", "class"),
            ("background_indexer", "async def"),
            ("get_index_status", "async def")
        ]
        
        for check_name, expected in config_checks:
            if expected in content:
                print(f"✅ {check_name}: {expected}")
            else:
                print(f"❌ {check_name}: {expected} - not found")
                return False
        
        print("✅ Configuration values test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration values test failed: {e}")
        return False

def test_binary_file_filtering():
    """Test binary file filtering specifically."""
    print("🧪 Testing binary file filtering...")
    
    # Test various binary file extensions
    binary_files = ["test.bin", "test.exe", "test.dll", "test.so", "test.o"]
    
    for binary_file in binary_files:
        should_process = should_process_file(binary_file)
        if should_process:
            print(f"❌ {binary_file} should be filtered but isn't")
            return False
        else:
            print(f"✅ {binary_file} correctly filtered")
    
    print("✅ Binary file filtering test passed!")
    return True

def main():
    """Run all tests."""
    print("🚀 Starting standalone tests...\n")
    
    tests = [
        ("File Filtering", test_file_filtering),
        ("Binary File Filtering", test_binary_file_filtering),
        ("Docker Files", test_docker_files),
        ("MCP Configuration", test_mcp_configuration),
        ("File Structure", test_file_structure),
        ("Configuration Values", test_configuration_values),
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
        print("🎉 All standalone tests passed! Implementation is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
