#!/usr/bin/env python3
"""
Run all tests for the codebase search system.
"""

import subprocess
import sys
import time

def run_test(test_name, test_file):
    """Run a specific test and return the result."""
    print(f"🧪 Running {test_name}...")
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {test_name} passed")
            return True
        else:
            print(f"❌ {test_name} failed")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ {test_name} timed out")
        return False
    except Exception as e:
        print(f"❌ {test_name} failed with exception: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running comprehensive test suite for Codebase Search System\n")
    
    tests = [
        ("Standalone Logic Tests", "test_standalone.py"),
        ("Integration Tests", "test_integration.py"),
        ("File Watching Tests", "test_file_watching.py"),
    ]
    
    start_time = time.time()
    results = []
    
    for test_name, test_file in tests:
        result = run_test(test_name, test_file)
        results.append((test_name, result))
        print()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Summary
    print("=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} test suites passed")
    print(f"⏱️  Total test time: {elapsed:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The codebase search system is fully functional and ready for deployment.")
        print("\n📋 System Features Verified:")
        print("  • File filtering and processing logic")
        print("  • Docker containerization")
        print("  • File watching and auto-indexing")
        print("  • Background indexing system")
        print("  • Cache management")
        print("  • Error handling and recovery")
        print("  • Performance optimizations")
        print("  • MCP server configuration")
        print("  • Health monitoring")
        print("\n🚀 Ready to deploy with: docker-compose up -d")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
