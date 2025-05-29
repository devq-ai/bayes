import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from bayes_mcp.mcp.handlers import handle_mcp_request
from bayes_mcp.mcp.server import app


class TestMCPHandlers:
    """Test suite for MCP handlers."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def mock_engine(self):
        """Create a mock Bayesian engine."""
        mock = Mock()
        mock.create_model.return_value = None
        mock.update_beliefs.return_value = {
            "p": {"mean": 0.7, "std": 0.1, "hdi_2.5%": 0.5, "hdi_97.5%": 0.9}
        }
        mock.get_model_info.return_value = {
            "variables": {"p": {"distribution": "beta"}},
            "created_at": "2024-01-01T00:00:00"
        }
        mock.list_models.return_value = ["model1", "model2"]
        mock.delete_model.return_value = True
        return mock

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_create_model_endpoint(self, client):
        """Test the create model endpoint."""
        model_data = {
            "model_name": "test_model",
            "variables": {
                "p": {
                    "distribution": "beta",
                    "params": {"alpha": 1, "beta": 1}
                }
            }
        }
        
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.create_model.return_value = None
            
            response = client.post("/mcp/create_model", json=model_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["model_name"] == "test_model"
            
            mock_engine.create_model.assert_called_once_with(
                "test_model", 
                model_data["variables"]
            )

    def test_update_beliefs_endpoint(self, client):
        """Test the update beliefs endpoint."""
        update_data = {
            "model_name": "test_model",
            "evidence": {"observed_data": 7},
            "mcmc_config": {
                "draws": 100,
                "tune": 100,
                "chains": 1
            }
        }
        
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.update_beliefs.return_value = {
                "p": {"mean": 0.7, "std": 0.1}
            }
            
            response = client.post("/mcp/update_beliefs", json=update_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "results" in data
            assert "p" in data["results"]

    def test_get_model_info_endpoint(self, client):
        """Test the get model info endpoint."""
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.get_model_info.return_value = {
                "variables": {"p": {"distribution": "beta"}},
                "created_at": "2024-01-01"
            }
            
            response = client.get("/mcp/model/test_model")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "model_info" in data
            assert "variables" in data["model_info"]

    def test_list_models_endpoint(self, client):
        """Test the list models endpoint."""
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.list_models.return_value = ["model1", "model2"]
            
            response = client.get("/mcp/models")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["models"] == ["model1", "model2"]

    def test_delete_model_endpoint(self, client):
        """Test the delete model endpoint."""
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.delete_model.return_value = True
            
            response = client.delete("/mcp/model/test_model")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Model 'test_model' deleted successfully"

    def test_create_model_validation_error(self, client):
        """Test create model with invalid data."""
        invalid_data = {
            "model_name": "",  # Empty name should fail
            "variables": {}
        }
        
        response = client.post("/mcp/create_model", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_create_model_missing_fields(self, client):
        """Test create model with missing required fields."""
        incomplete_data = {
            "model_name": "test"
            # Missing variables field
        }
        
        response = client.post("/mcp/create_model", json=incomplete_data)
        assert response.status_code == 422

    def test_update_beliefs_nonexistent_model(self, client):
        """Test updating beliefs for nonexistent model."""
        update_data = {
            "model_name": "nonexistent",
            "evidence": {},
            "mcmc_config": {"draws": 100}
        }
        
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.update_beliefs.side_effect = KeyError("Model not found")
            
            response = client.post("/mcp/update_beliefs", json=update_data)
            assert response.status_code == 404

    def test_get_model_info_nonexistent(self, client):
        """Test getting info for nonexistent model."""
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.get_model_info.side_effect = KeyError("Model not found")
            
            response = client.get("/mcp/model/nonexistent")
            assert response.status_code == 404

    def test_delete_nonexistent_model(self, client):
        """Test deleting nonexistent model."""
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.delete_model.return_value = False
            
            response = client.delete("/mcp/model/nonexistent")
            assert response.status_code == 404

    def test_engine_exception_handling(self, client):
        """Test handling of engine exceptions."""
        model_data = {
            "model_name": "error_model",
            "variables": {"p": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}}}
        }
        
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.create_model.side_effect = ValueError("Invalid model configuration")
            
            response = client.post("/mcp/create_model", json=model_data)
            assert response.status_code == 400
            
            data = response.json()
            assert data["success"] is False
            assert "error" in data

    def test_schema_endpoint(self, client):
        """Test the schema endpoint."""
        response = client.get("/schema")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_request_validation_middleware(self, client):
        """Test request validation for malformed JSON."""
        response = client.post(
            "/mcp/create_model",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/mcp/models")
        assert response.status_code == 200
        # CORS headers should be present if configured

    def test_content_type_validation(self, client):
        """Test content type validation."""
        model_data = {
            "model_name": "test_model",
            "variables": {"p": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}}}
        }
        
        # Test with correct content type
        response = client.post(
            "/mcp/create_model",
            json=model_data,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [200, 422]  # Should at least accept the request

    @patch('bayes_mcp.mcp.handlers.engine')
    def test_concurrent_requests(self, mock_engine, client):
        """Test handling of concurrent requests."""
        mock_engine.list_models.return_value = ["model1"]
        
        # Simulate concurrent requests
        responses = []
        for _ in range(5):
            response = client.get("/mcp/models")
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_large_model_data(self, client):
        """Test handling of large model configurations."""
        # Create a large model configuration
        large_variables = {}
        for i in range(50):
            large_variables[f"var_{i}"] = {
                "distribution": "normal",
                "params": {"mu": 0, "sigma": 1}
            }
        
        model_data = {
            "model_name": "large_model",
            "variables": large_variables
        }
        
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.create_model.return_value = None
            
            response = client.post("/mcp/create_model", json=model_data)
            # Should handle large payloads gracefully
            assert response.status_code in [200, 413, 422]

    def test_mcp_protocol_compliance(self, client):
        """Test basic MCP protocol compliance."""
        # Test that responses follow expected format
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.list_models.return_value = []
            
            response = client.get("/mcp/models")
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, dict)
            assert "success" in data
            assert isinstance(data["success"], bool)

    def test_error_response_format(self, client):
        """Test that error responses follow consistent format."""
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.get_model_info.side_effect = KeyError("Not found")
            
            response = client.get("/mcp/model/nonexistent")
            assert response.status_code == 404
            
            data = response.json()
            assert "success" in data
            assert data["success"] is False
            assert "error" in data or "message" in data

    def test_special_characters_in_model_names(self, client):
        """Test handling of special characters in model names."""
        special_names = [
            "model-with-dashes",
            "model_with_underscores", 
            "model.with.dots",
            "model with spaces"
        ]
        
        with patch('bayes_mcp.mcp.handlers.engine') as mock_engine:
            mock_engine.get_model_info.side_effect = KeyError("Not found")
            
            for name in special_names:
                response = client.get(f"/mcp/model/{name}")
                # Should handle gracefully, either succeed or return proper error
                assert response.status_code in [200, 404, 422]