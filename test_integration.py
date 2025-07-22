#!/usr/bin/env python3
"""
Integration test for the codebase search system.
"""

import os
import json
import subprocess
import sys
import time

def test_dockerfile_syntax():
    """Test Dockerfile syntax and structure."""
    print("🧪 Testing Dockerfile...")
    
    try:
        with open("Dockerfile", "r") as f:
            content = f.read()
        
        # Check for required Dockerfile components
        required_components = [
            "FROM python:3.11-slim",
            "WORKDIR /app",
            "COPY requirements.txt",
            "RUN pip install",
            "EXPOSE 8086",
            "CMD"
        ]
        
        for component in required_components:
            if component in content:
                print(f"✅ {component}")
            else:
                print(f"❌ {component} - missing")
                return False
        
        print("✅ Dockerfile test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dockerfile test failed: {e}")
        return False

def test_docker_compose():
    """Test Docker Compose configuration."""
    print("🧪 Testing Docker Compose...")
    
    try:
        with open("docker-compose.yml", "r") as f:
            content = f.read()
        
        # Check for required compose components
        required_components = [
            "version:",
            "services:",
            "codebase-search:",
            "build:",
            "ports:",
            "volumes:",
            "8086:8086"
        ]
        
        for component in required_components:
            if component in content:
                print(f"✅ {component}")
            else:
                print(f"❌ {component} - missing")
                return False
        
        print("✅ Docker Compose test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Docker Compose test failed: {e}")
        return False

def test_startup_script():
    """Test the startup script."""
    print("🧪 Testing startup script...")
    
    try:
        with open("start_server.sh", "r") as f:
            content = f.read()
        
        # Check for required script components
        required_components = [
            "#!/bin/bash",
            "ollama serve",
            "ollama pull nomic-embed-text",
            "python3 mcp_codebase_server.py"
        ]
        
        for component in required_components:
            if component in content:
                print(f"✅ {component}")
            else:
                print(f"❌ {component} - missing")
                return False
        
        # Check if script is executable
        if os.access("start_server.sh", os.X_OK):
            print("✅ Script is executable")
        else:
            print("❌ Script is not executable")
            return False
        
        print("✅ Startup script test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Startup script test failed: {e}")
        return False

def test_health_check():
    """Test the health check script."""
    print("🧪 Testing health check script...")
    
    try:
        with open("health_check.py", "r") as f:
            content = f.read()
        
        # Check for required health check components
        required_components = [
            "requests.get",
            "localhost:8086/health",
            "status_code",
            "sys.exit"
        ]
        
        for component in required_components:
            if component in content:
                print(f"✅ {component}")
            else:
                print(f"❌ {component} - missing")
                return False
        
        print("✅ Health check test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

def test_requirements():
    """Test requirements.txt."""
    print("🧪 Testing requirements.txt...")
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        # Check for required packages
        required_packages = [
            "mcp",
            "ollama",
            "numpy",
            "watchdog"
        ]
        
        for package in required_packages:
            if package in content:
                print(f"✅ {package}")
            else:
                print(f"❌ {package} - missing")
                return False
        
        print("✅ Requirements test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def test_makefile():
    """Test the Makefile."""
    print("🧪 Testing Makefile...")
    
    try:
        with open("Makefile", "r") as f:
            content = f.read()
        
        # Check for required make targets
        required_targets = [
            "build:",
            "run:",
            "stop:",
            "logs:",
            "test:",
            "clean:"
        ]
        
        for target in required_targets:
            if target in content:
                print(f"✅ {target}")
            else:
                print(f"❌ {target} - missing")
                return False
        
        print("✅ Makefile test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Makefile test failed: {e}")
        return False

def test_file_permissions():
    """Test file permissions."""
    print("🧪 Testing file permissions...")
    
    # Check if startup script is executable
    if os.access("start_server.sh", os.X_OK):
        print("✅ start_server.sh is executable")
    else:
        print("❌ start_server.sh is not executable")
        return False
    
    print("✅ File permissions test passed!")
    return True

def test_configuration_consistency():
    """Test configuration consistency across files."""
    print("🧪 Testing configuration consistency...")
    
    try:
        # Check that port 8086 is consistent across files
        files_to_check = [
            ("docker-compose.yml", "8086:8086"),
            ("docker-compose.dev.yml", "8086:8086"),
            (".kilocode/mcp.json", "8086"),
            ("Dockerfile", "8086")
        ]
        
        for filename, expected_content in files_to_check:
            with open(filename, "r") as f:
                content = f.read()
                if expected_content in content:
                    print(f"✅ {filename} has correct port")
                else:
                    print(f"❌ {filename} missing port configuration")
                    return False
        
        print("✅ Configuration consistency test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration consistency test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting integration tests...\n")
    
    tests = [
        ("Dockerfile Syntax", test_dockerfile_syntax),
        ("Docker Compose", test_docker_compose),
        ("Startup Script", test_startup_script),
        ("Health Check", test_health_check),
        ("Requirements", test_requirements),
        ("Makefile", test_makefile),
        ("File Permissions", test_file_permissions),
        ("Configuration Consistency", test_configuration_consistency),
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
    print("📊 Integration Test Results:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} integration tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! System is ready for deployment.")
        return 0
    else:
        print("⚠️  Some integration tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
