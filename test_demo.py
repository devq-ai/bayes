#!/usr/bin/env python3
"""
Simple Test Demo for Bayesian MCP Tool

This script tests basic functionality of the Bayesian MCP server
with a minimal A/B testing example.
"""

import requests
import time
import subprocess
import signal
import os
import sys
import json
from typing import Optional

class SimpleBayesianTest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.server_process = None
        
    def start_server(self, timeout: int = 15) -> bool:
        """Start the server and wait for it to be ready."""
        print("🚀 Starting Bayesian MCP Server...")
        
        try:
            # Start server in background
            self.server_process = subprocess.Popen(
                [sys.executable, "bayes_mcp.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Wait for server to be ready
            for i in range(timeout):
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Server ready at {self.base_url}")
                        return True
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
                if i < timeout - 1:
                    print(f"⏳ Waiting for server... ({i+1}/{timeout})")
                    
            print("❌ Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
            
    def stop_server(self):
        """Stop the server."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            print("🛑 Server stopped")
            
    def test_health(self) -> bool:
        """Test server health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
            
    def test_functions_endpoint(self) -> bool:
        """Test functions listing endpoint."""
        try:
            response = requests.get(f"{self.base_url}/functions")
            if response.status_code == 200:
                functions = response.json().get("available_functions", [])
                print(f"✅ Available functions: {functions}")
                return True
            else:
                print(f"❌ Functions endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Functions endpoint error: {e}")
            return False
            
    def test_simple_model(self) -> bool:
        """Test creating a simple Bayesian model."""
        print("\n📊 Testing Simple Bayesian Model Creation...")
        
        try:
            # Simple coin flip model
            request_data = {
                "function_name": "create_model",
                "parameters": {
                    "model_name": "simple_test",
                    "variables": {
                        "p": {
                            "distribution": "beta",
                            "params": {"alpha": 1, "beta": 1}
                        },
                        "obs": {
                            "distribution": "binomial",
                            "params": {"n": 10, "p": "p"},
                            "observed": 7
                        }
                    }
                }
            }
            
            response = requests.post(f"{self.base_url}/mcp", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    print("✅ Model creation successful")
                    print(f"   Message: {result.get('message', 'No message')}")
                    return True
                else:
                    print(f"❌ Model creation failed: {result.get('message', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Model creation request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Model creation error: {e}")
            return False
            
    def run_basic_tests(self) -> bool:
        """Run basic functionality tests."""
        print("🧪 Running Basic Functionality Tests")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health),
            ("Functions Endpoint", self.test_functions_endpoint), 
            ("Simple Model Creation", self.test_simple_model)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            try:
                result = test_func()
                results.append(result)
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                results.append(False)
                
        passed = sum(results)
        total = len(results)
        
        print(f"\n📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed. Check server logs for details.")
            return False

def main():
    """Main test function."""
    print("🧪 Bayesian MCP Tool - Basic Functionality Test")
    print("=" * 60)
    
    tester = SimpleBayesianTest()
    
    try:
        # Start server
        if not tester.start_server():
            print("❌ Cannot start server. Check your installation.")
            return 1
            
        # Run tests
        if tester.run_basic_tests():
            print("\n🎉 Basic functionality verified!")
            print("✅ Your Bayesian MCP tool is working correctly.")
            print("\n🚀 Next steps:")
            print("  1. Run full demos: python demos/run_all_demos.py")
            print("  2. Try examples: python examples/ab_test.py")
            print("  3. Read setup guide: SETUP_GUIDE.md")
            return 0
        else:
            print("\n❌ Some tests failed.")
            print("🔧 Check the installation and try again.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    finally:
        tester.stop_server()

if __name__ == "__main__":
    sys.exit(main())