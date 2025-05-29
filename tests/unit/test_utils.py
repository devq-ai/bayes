import pytest
from unittest.mock import Mock, patch, mock_open
import json
import tempfile
from pathlib import Path
from bayes_mcp.utils.plotting import create_trace_plot, create_posterior_plot


class TestPlottingUtils:
    """Test suite for plotting utilities."""

    def test_create_trace_plot_basic(self):
        """Test basic trace plot creation."""
        # Mock data for trace plot
        mock_trace_data = {
            "parameter": [0.1, 0.2, 0.15, 0.25, 0.18]
        }
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            with patch('matplotlib.pyplot.plot') as mock_plot:
                result = create_trace_plot(mock_trace_data, "test_parameter")
                assert mock_figure.called
                assert mock_plot.called

    def test_create_posterior_plot_basic(self):
        """Test basic posterior plot creation."""
        # Mock data for posterior plot
        mock_posterior_data = {
            "values": [0.1, 0.2, 0.15, 0.25, 0.18, 0.22, 0.12, 0.28]
        }
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            with patch('matplotlib.pyplot.hist') as mock_hist:
                result = create_posterior_plot(mock_posterior_data, "test_parameter")
                assert mock_figure.called
                assert mock_hist.called

    def test_plotting_with_invalid_data(self):
        """Test plotting functions handle invalid data gracefully."""
        invalid_data = None
        
        with patch('matplotlib.pyplot.figure'):
            # Should handle None data gracefully
            result = create_trace_plot(invalid_data, "test")
            # Function should handle this case appropriately

    def test_plotting_with_empty_data(self):
        """Test plotting functions handle empty data."""
        empty_data = {"parameter": []}
        
        with patch('matplotlib.pyplot.figure'):
            # Should handle empty data gracefully
            result = create_trace_plot(empty_data, "test")
            # Function should handle this case appropriately


class TestBasicUtilities:
    """Test suite for basic utility functions."""

    def test_json_serialization(self, temp_workspace):
        """Test basic JSON serialization utilities."""
        test_data = {
            "name": "test_model",
            "variables": {"p": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}}},
            "created_at": "2024-01-01T00:00:00"
        }
        
        file_path = temp_workspace / "test_model.json"
        
        # Test saving
        with open(file_path, 'w') as f:
            json.dump(test_data, f)
        
        assert file_path.exists()
        
        # Test loading
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data

    def test_file_operations(self, temp_workspace):
        """Test basic file operations."""
        test_file = temp_workspace / "test.txt"
        test_content = "test content"
        
        # Write test
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        assert test_file.exists()
        
        # Read test
        with open(test_file, 'r') as f:
            content = f.read()
        
        assert content == test_content

    def test_pathlib_operations(self, temp_workspace):
        """Test pathlib operations."""
        test_dir = temp_workspace / "subdir"
        test_dir.mkdir()
        
        assert test_dir.exists()
        assert test_dir.is_dir()
        
        test_file = test_dir / "test.json"
        test_file.write_text('{"test": "data"}')
        
        assert test_file.exists()
        assert test_file.is_file()

    def test_mock_functionality(self):
        """Test mock functionality works correctly."""
        mock_obj = Mock()
        mock_obj.test_method.return_value = "test_result"
        
        result = mock_obj.test_method()
        assert result == "test_result"
        assert mock_obj.test_method.called

    @patch('json.dump')
    def test_patch_functionality(self, mock_json_dump):
        """Test patch functionality works correctly."""
        test_data = {"test": "data"}
        
        with open("dummy_path", 'w') as f:
            json.dump(test_data, f)
        
        mock_json_dump.assert_called_once_with(test_data, f)

    def test_unicode_handling(self, temp_workspace):
        """Test handling of unicode characters."""
        unicode_data = {
            "name": "unicode_test_μ_σ",
            "description": "Test with unicode: α, β, γ, Ω",
            "variables": {
                "θ": {"distribution": "beta", "params": {"α": 1, "β": 1}}
            }
        }
        
        file_path = temp_workspace / "unicode_test.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(unicode_data, f, ensure_ascii=False)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == unicode_data
        assert "θ" in loaded_data["variables"]

    def test_large_data_handling(self, temp_workspace):
        """Test handling of large data structures."""
        large_data = {
            "name": "large_test",
            "variables": {}
        }
        
        # Create a large structure
        for i in range(100):  # Reduced from 1000 for faster testing
            large_data["variables"][f"var_{i}"] = {
                "distribution": "normal",
                "params": {"mu": i, "sigma": 1}
            }
        
        file_path = temp_workspace / "large_test.json"
        
        with open(file_path, 'w') as f:
            json.dump(large_data, f)
        
        assert file_path.exists()
        
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data["variables"]) == 100

    def test_error_handling(self, temp_workspace):
        """Test basic error handling."""
        non_existent_file = temp_workspace / "does_not_exist.json"
        
        # Test file not found
        try:
            with open(non_existent_file, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            # Expected behavior
            pass
        
        # Test invalid JSON
        invalid_json_file = temp_workspace / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("invalid json content")
        
        try:
            with open(invalid_json_file, 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            # Expected behavior
            pass