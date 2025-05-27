#!/usr/bin/env python3
"""Debug script to test server startup."""

import sys
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports step by step."""
    try:
        print("1. Testing basic imports...")
        import pymc as pm
        import numpy as np
        import arviz as az
        print("   ✅ PyMC imports successful")
        
        print("2. Testing engine import...")
        from bayesian_mcp.bayesian_engine.engine import BayesianEngine
        print("   ✅ Engine import successful")
        
        print("3. Testing engine initialization...")
        engine = BayesianEngine()
        print("   ✅ Engine initialization successful")
        
        print("4. Testing FastAPI import...")
        from fastapi import FastAPI
        import uvicorn
        print("   ✅ FastAPI imports successful")
        
        print("5. Testing server module import...")
        from bayesian_mcp.mcp.server import app
        print("   ✅ Server module import successful")
        
        print("6. Testing handlers import...")
        from bayesian_mcp.mcp.handlers import handle_mcp_request
        print("   ✅ Handlers import successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_simple_model():
    """Test simple model creation."""
    try:
        print("7. Testing simple model creation...")
        from bayesian_mcp.bayesian_engine.engine import BayesianEngine
        
        engine = BayesianEngine()
        
        # Simple coin flip model
        variables = {
            "p": {
                "distribution": "beta",
                "params": {"alpha": 1, "beta": 1}
            },
            "flips": {
                "distribution": "binomial",
                "params": {"n": 10, "p": "p"},
                "observed": 7
            }
        }
        
        engine.create_model("test_model", variables)
        print("   ✅ Model creation successful")
        
        # Test belief updating
        print("8. Testing belief updating...")
        result = engine.update_beliefs("test_model", {}, {
            "draws": 100,
            "tune": 100,
            "chains": 1,
            "progressbar": False
        })
        
        if "p" in result:
            p_mean = result["p"]["mean"]
            print(f"   ✅ Belief updating successful - estimated p: {p_mean:.3f}")
        else:
            print("   ❌ No 'p' in result")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        traceback.print_exc()
        return False

def test_server_creation():
    """Test server app creation."""
    try:
        print("9. Testing server app creation...")
        from bayesian_mcp.mcp.server import app
        
        # Test that app has expected routes
        routes = [route.path for route in app.routes]
        print(f"   Available routes: {routes}")
        
        if "/health" in routes:
            print("   ✅ Health endpoint found")
        else:
            print("   ❌ Health endpoint missing")
            
        if "/mcp" in routes:
            print("   ✅ MCP endpoint found")
        else:
            print("   ❌ MCP endpoint missing")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Server app test failed: {e}")
        traceback.print_exc()
        return False

def test_manual_server_start():
    """Test manual server startup."""
    try:
        print("10. Testing manual server startup...")
        import uvicorn
        from bayesian_mcp.mcp.server import app
        
        # Try to start server manually (this will block)
        print("    Starting server on port 8002...")
        uvicorn.run(app, host="127.0.0.1", port=8002, log_level="debug")
        
    except Exception as e:
        print(f"   ❌ Manual server start failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔍 Bayesian MCP Server Debug Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_simple_model,
        test_server_creation
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
            print("\n❌ Stopping due to test failure")
            break
        print()
    
    if all_passed:
        print("🎉 All tests passed! Trying manual server start...")
        test_manual_server_start()
    else:
        print("❌ Some tests failed. Fix issues before starting server.")
        sys.exit(1)

if __name__ == "__main__":
    main()