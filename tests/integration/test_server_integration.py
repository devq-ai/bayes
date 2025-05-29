import pytest
import asyncio
import time
import subprocess
import sys
import requests
from pathlib import Path
import json
from typing import Dict, Any


class TestServerIntegration:
    """Integration tests for MCP server lifecycle."""

    def test_server_startup_and_health(self, test_server):
        """Test that server starts up correctly and responds to health checks."""
        response = requests.get(f"{test_server['base_url']}/health", timeout=5)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"

    def test_server_root_endpoint(self, test_server):
        """Test the root endpoint."""
        response = requests.get(test_server["base_url"], timeout=5)
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "Bayesian MCP Server" in data["message"]

    def test_schema_endpoint(self, test_server):
        """Test the schema endpoint."""
        response = requests.get(f"{test_server['base_url']}/schema", timeout=5)
        assert response.status_code == 200
        
        data = response.json()
        assert "create_model" in data
        assert "update_beliefs" in data
        assert "predict" in data

    def test_functions_endpoint(self, test_server):
        """Test the functions listing endpoint."""
        response = requests.get(f"{test_server['base_url']}/functions", timeout=5)
        assert response.status_code == 200
        
        data = response.json()
        assert "available_functions" in data
        assert isinstance(data["available_functions"], list)

    def test_mcp_endpoint_structure(self, test_server):
        """Test that MCP endpoint exists and handles requests."""
        # Test with invalid request (missing required fields)
        response = requests.post(
            f"{test_server['base_url']}/mcp",
            json={"invalid": "request"},
            timeout=5
        )
        assert response.status_code in [400, 500]  # Server may return either

    def test_create_model_integration(self, test_server):
        """Test model creation through MCP endpoint."""
        request_data = {
            "function_name": "create_model",
            "parameters": {
                "model_name": "integration_test_model",
                "variables": {
                    "p": {
                        "distribution": "beta",
                        "params": {"alpha": 1, "beta": 1}
                    },
                    "observations": {
                        "distribution": "binomial",
                        "params": {"n": 10, "p": "p"},
                        "observed": 7
                    }
                }
            }
        }
        
        response = requests.post(
            f"{test_server['base_url']}/mcp",
            json=request_data,
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == "integration_test_model"

    def test_update_beliefs_integration(self, test_server):
        """Test belief updating through MCP endpoint."""
        # First create a model
        create_request = {
            "function_name": "create_model",
            "parameters": {
                "model_name": "belief_test_model",
                "variables": {
                    "p": {
                        "distribution": "beta",
                        "params": {"alpha": 1, "beta": 1}
                    },
                    "observations": {
                        "distribution": "binomial",
                        "params": {"n": 10, "p": "p"},
                        "observed": 6
                    }
                }
            }
        }
        
        create_response = requests.post(
            f"{test_server['base_url']}/mcp",
            json=create_request,
            timeout=10
        )
        assert create_response.status_code == 200
        
        # Now update beliefs
        update_request = {
            "function_name": "update_beliefs",
            "parameters": {
                "model_name": "belief_test_model",
                "evidence": {},
                "sample_kwargs": {
                    "draws": 100,
                    "tune": 100,
                    "chains": 1,
                    "progressbar": False
                }
            }
        }
        
        update_response = requests.post(
            f"{test_server['base_url']}/mcp",
            json=update_request,
            timeout=30
        )
        
        assert update_response.status_code == 200
        data = update_response.json()
        assert data["success"] is True
        assert "posterior" in data
        assert "p" in data["posterior"]

    def test_error_handling_integration(self, test_server):
        """Test error handling in integration scenarios."""
        # Test with nonexistent model
        request_data = {
            "function_name": "update_beliefs",
            "parameters": {
                "model_name": "nonexistent_model",
                "evidence": {},
                "sample_kwargs": {"draws": 100}
            }
        }
        
        response = requests.post(
            f"{test_server['base_url']}/mcp",
            json=request_data,
            timeout=10
        )
        
        assert response.status_code == 200  # Server responds, but with error
        data = response.json()
        assert data["success"] is False

    def test_concurrent_requests_integration(self, test_server):
        """Test handling of concurrent requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = requests.get(f"{test_server['base_url']}/health", timeout=5)
                results.put(response.status_code)
            except Exception as e:
                results.put(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        while not results.empty():
            result = results.get()
            assert result == 200

    def test_server_shutdown_graceful(self, test_server):
        """Test that server can be shut down gracefully."""
        # This test verifies the server is running
        response = requests.get(f"{test_server['base_url']}/health", timeout=5)
        assert response.status_code == 200
        
        # The test_server fixture handles shutdown automatically
        # We just verify it's currently working

    def test_large_request_handling(self, test_server):
        """Test server handling of large requests."""
        # Create a large model configuration
        large_variables = {}
        for i in range(20):  # Reasonable size for testing
            large_variables[f"var_{i}"] = {
                "distribution": "normal",
                "params": {"mu": 0, "sigma": 1}
            }
        
        request_data = {
            "function_name": "create_model",
            "parameters": {
                "model_name": "large_test_model",
                "variables": large_variables
            }
        }
        
        response = requests.post(
            f"{test_server['base_url']}/mcp",
            json=request_data,
            timeout=15
        )
        
        # Should handle large requests gracefully
        assert response.status_code in [200, 413, 500]  # Various acceptable responses

    def test_malformed_json_handling(self, test_server):
        """Test server handling of malformed JSON."""
        response = requests.post(
            f"{test_server['base_url']}/mcp",
            data="invalid json content",
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        # Server should handle malformed JSON gracefully
        assert response.status_code in [400, 422, 500]

    def test_server_persistence(self, test_server):
        """Test that models persist across requests."""
        # Create a model
        create_request = {
            "function_name": "create_model",
            "parameters": {
                "model_name": "persistence_test",
                "variables": {
                    "p": {
                        "distribution": "beta",
                        "params": {"alpha": 2, "beta": 2}
                    }
                }
            }
        }
        
        create_response = requests.post(
            f"{test_server['base_url']}/mcp",
            json=create_request,
            timeout=10
        )
        assert create_response.status_code == 200
        
        # Wait a moment
        time.sleep(1)
        
        # Try to use the model in another request
        use_request = {
            "function_name": "update_beliefs",
            "parameters": {
                "model_name": "persistence_test",
                "evidence": {},
                "sample_kwargs": {"draws": 50, "tune": 50, "chains": 1, "progressbar": False}
            }
        }
        
        use_response = requests.post(
            f"{test_server['base_url']}/mcp",
            json=use_request,
            timeout=15
        )
        
        # Model should still exist
        assert use_response.status_code == 200
        data = use_response.json()
        assert data["success"] is True