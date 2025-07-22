#!/usr/bin/env python3
"""
Health check script for the codebase search MCP server.
"""

import requests
import sys
import time

def check_health():
    """Check if the MCP server is healthy."""
    try:
        # Try to connect to the MCP server
        response = requests.get("http://localhost:8086/health", timeout=5)
        if response.status_code == 200:
            print("✅ MCP server is healthy")
            return True
        else:
            print(f"❌ MCP server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to MCP server: {e}")
        return False

if __name__ == "__main__":
    # Wait a bit for the server to start
    time.sleep(2)
    
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
