import pytest
import asyncio
import subprocess
import time
import requests
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bayes_mcp.bayesian_engine.engine import BayesianEngine


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def bayesian_engine():
    """Create a fresh Bayesian engine instance for each test."""
    return BayesianEngine()


@pytest.fixture
def sample_model_config():
    """Provide a sample model configuration for testing."""
    return {
        "coin_flip": {
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
    }


@pytest.fixture
def ab_test_config():
    """Provide A/B test configuration for testing."""
    return {
        "conversion_rate_a": {
            "distribution": "beta",
            "params": {"alpha": 1, "beta": 1}
        },
        "conversion_rate_b": {
            "distribution": "beta", 
            "params": {"alpha": 1, "beta": 1}
        },
        "conversions_a": {
            "distribution": "binomial",
            "params": {"n": 100, "p": "conversion_rate_a"},
            "observed": 12
        },
        "conversions_b": {
            "distribution": "binomial",
            "params": {"n": 100, "p": "conversion_rate_b"},
            "observed": 18
        }
    }


@pytest.fixture
def medical_diagnosis_config():
    """Provide medical diagnosis configuration for testing."""
    return {
        "disease_prevalence": {
            "distribution": "beta",
            "params": {"alpha": 1, "beta": 99}
        },
        "test_sensitivity": {
            "distribution": "beta",
            "params": {"alpha": 95, "beta": 5}
        },
        "test_specificity": {
            "distribution": "beta",
            "params": {"alpha": 99, "beta": 1}
        }
    }


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def free_port():
    """Find a free port for test server."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture(scope="session")
def test_server(free_port):
    """Start the MCP server for integration tests."""
    server_process = None
    try:
        # Start server in background
        server_process = subprocess.Popen(
            [sys.executable, "bayes_mcp.py", "--host", "127.0.0.1", "--port", str(free_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent
        )
        
        # Wait for server to be ready
        base_url = f"http://127.0.0.1:{free_port}"
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError(f"Server failed to start after {max_attempts} seconds")
            
        yield {
            "base_url": base_url,
            "port": free_port,
            "process": server_process
        }
        
    finally:
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()


@pytest.fixture
def mcp_request_headers():
    """Standard headers for MCP requests."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


@pytest.fixture
def sample_mcmc_config():
    """Sample MCMC configuration for testing."""
    return {
        "draws": 100,
        "tune": 100,
        "chains": 1,
        "progressbar": False,
        "return_inferencedata": True
    }


@pytest.fixture
def fast_mcmc_config():
    """Fast MCMC configuration for quick tests."""
    return {
        "draws": 50,
        "tune": 50,
        "chains": 1,
        "progressbar": False,
        "return_inferencedata": True
    }