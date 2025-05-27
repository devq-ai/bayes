# Bayesian MCP Tool - Complete Setup & Deployment Guide

## 🎯 Production Status: Ready for Deployment

This guide provides complete instructions for setting up, deploying, and using the Bayesian MCP tool in production environments. The tool has been fully tested and validated across multiple real-world scenarios.

## 📋 Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Internet access for package installation

### Development Tools
- Git for version control
- Virtual environment support (venv, conda, or virtualenv)
- curl or equivalent for API testing

## 🚀 Quick Start (5 Minutes)

### 1. Clone and Install
```bash
# Clone repository
git clone <repository-url>
cd bayes

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package and dependencies
pip install -e .
```

### 2. Start Server
```bash
# Start on default port 8002
python bayesian_mcp.py --port 8002

# Or with custom configuration
python bayesian_mcp.py --host 0.0.0.0 --port 8080 --log-level info
```

### 3. Verify Installation
```bash
# Check server health
curl http://localhost:8002/health
# Expected: {"status":"healthy"}

# List available functions
curl http://localhost:8002/functions
# Expected: {"available_functions":["create_model","update_beliefs","predict","compare_models","create_visualization"]}
```

### 4. Run Demonstrations
```bash
# Comprehensive demo (all use cases)
python demos/master_demo.py

# Individual demos
python demos/ab_testing_demo.py      # Business A/B testing
python demos/medical_diagnosis_demo.py  # Healthcare applications
python demos/financial_risk_demo.py     # Risk management
```

## 📦 Detailed Installation Guide

### Environment Setup

#### Option 1: Virtual Environment (Recommended)
```bash
# Create isolated environment
python3 -m venv bayesian_mcp_env
source bayesian_mcp_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package
cd bayes
pip install -e .
```

#### Option 2: Conda Environment
```bash
# Create conda environment
conda create -n bayesian_mcp python=3.9
conda activate bayesian_mcp

# Install package
cd bayes
pip install -e .
```

#### Option 3: System Installation (Not Recommended)
```bash
# Install system-wide (use with caution)
cd bayes
pip install -e . --user
```

### Dependency Management

#### Core Dependencies (Automatically Installed)
```toml
# From pyproject.toml
dependencies = [
    "pymc>=5.0.0",        # Bayesian inference engine
    "arviz>=0.14.0",      # Posterior analysis
    "numpy>=1.20.0",      # Numerical computing
    "scipy>=1.7.0",       # Scientific computing
    "pydantic>=2.0.0",    # Data validation
    "fastapi>=0.100.0",   # Web framework
    "uvicorn>=0.20.0",    # ASGI server
    "matplotlib>=3.5.0",  # Plotting
    "seaborn>=0.11.0",    # Statistical plotting
    "requests>=2.25.0",   # HTTP client
    "scikit-learn>=1.0.0" # Machine learning
]
```

#### Development Dependencies (Optional)
```bash
# Install development tools
pip install -e ".[dev]"

# Includes: pytest, black, isort, mypy, jupyter
```

### Verification Steps

#### 1. Package Import Test
```python
# Test basic imports
python -c "
import bayesian_mcp
from bayesian_mcp.bayesian_engine.engine import BayesianEngine
from bayesian_mcp.mcp.server import app
print('✅ All imports successful')
"
```

#### 2. Engine Functionality Test
```python
# Test core engine
python -c "
from bayesian_mcp.bayesian_engine.engine import BayesianEngine
engine = BayesianEngine()
print('✅ Engine initialization successful')
"
```

#### 3. Server Startup Test
```bash
# Start server in background
python bayesian_mcp.py --port 8002 &
SERVER_PID=$!

# Wait for startup
sleep 3

# Test health endpoint
curl -f http://localhost:8002/health || echo "❌ Health check failed"

# Stop server
kill $SERVER_PID
```

## 🏗️ Configuration Options

### Server Configuration

#### Command Line Arguments
```bash
python bayesian_mcp.py [OPTIONS]

Options:
  --host TEXT        Host to bind server (default: 127.0.0.1)
  --port INTEGER     Port to bind server (default: 8000)
  --log-level TEXT   Logging level (default: info)
                     Choices: debug, info, warning, error, critical
  --help            Show help message
```

#### Environment Variables
```bash
# Set environment variables
export BAYESIAN_MCP_HOST=0.0.0.0
export BAYESIAN_MCP_PORT=8002
export BAYESIAN_MCP_LOG_LEVEL=info

# Start server with environment config
python bayesian_mcp.py
```

#### Configuration File (.env)
```bash
# Create .env file
cat > .env << EOF
BAYESIAN_MCP_HOST=127.0.0.1
BAYESIAN_MCP_PORT=8002
BAYESIAN_MCP_LOG_LEVEL=info
BAYESIAN_MCP_WORKERS=1
EOF
```

### MCMC Sampling Configuration

#### Default Parameters
```python
default_sample_kwargs = {
    "draws": 1000,      # Samples per chain
    "tune": 1000,       # Tuning samples
    "chains": 2,        # Number of chains
    "cores": 1,         # CPU cores to use
    "progressbar": False # Disable progress bar
}
```

#### Performance Tuning
```python
# Fast sampling (development)
fast_kwargs = {
    "draws": 500,
    "tune": 500,
    "chains": 1
}

# Production sampling (high accuracy)
production_kwargs = {
    "draws": 3000,
    "tune": 1500,
    "chains": 4,
    "target_accept": 0.9
}

# Memory-constrained sampling
constrained_kwargs = {
    "draws": 1000,
    "tune": 1000,
    "chains": 2,
    "cores": 1
}
```

## 🌐 Production Deployment

### Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Start server
CMD ["python", "bayesian_mcp.py", "--host", "0.0.0.0", "--port", "8002"]
```

#### Build and Run
```bash
# Build Docker image
docker build -t bayesian-mcp:latest .

# Run container
docker run -d \
    --name bayesian-mcp \
    -p 8002:8002 \
    --restart unless-stopped \
    bayesian-mcp:latest

# Check health
curl http://localhost:8002/health
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  bayesian-mcp:
    build: .
    ports:
      - "8002:8002"
    environment:
      - BAYESIAN_MCP_HOST=0.0.0.0
      - BAYESIAN_MCP_PORT=8002
      - BAYESIAN_MCP_LOG_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - bayesian-mcp
    restart: unless-stopped
```

### Kubernetes Deployment

#### Deployment Manifest
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bayesian-mcp
  labels:
    app: bayesian-mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bayesian-mcp
  template:
    metadata:
      labels:
        app: bayesian-mcp
    spec:
      containers:
      - name: bayesian-mcp
        image: bayesian-mcp:latest
        ports:
        - containerPort: 8002
        env:
        - name: BAYESIAN_MCP_HOST
          value: "0.0.0.0"
        - name: BAYESIAN_MCP_PORT
          value: "8002"
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: bayesian-mcp-service
spec:
  selector:
    app: bayesian-mcp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8002
  type: LoadBalancer
```

#### Deploy to Kubernetes
```bash
# Apply deployment
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=bayesian-mcp
kubectl get service bayesian-mcp-service

# Test service
kubectl port-forward service/bayesian-mcp-service 8002:80
curl http://localhost:8002/health
```

### Load Balancing

#### Nginx Configuration
```nginx
# nginx.conf
upstream bayesian_mcp {
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    server 127.0.0.1:8004;
}

server {
    listen 80;
    server_name bayesian-mcp.company.com;

    location / {
        proxy_pass http://bayesian_mcp;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_timeout 300s;
    }

    location /health {
        proxy_pass http://bayesian_mcp/health;
        access_log off;
    }
}
```

#### HAProxy Configuration
```cfg
# haproxy.cfg
global
    daemon

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend bayesian_mcp_frontend
    bind *:80
    default_backend bayesian_mcp_backend

backend bayesian_mcp_backend
    balance roundrobin
    option httpchk GET /health
    server app1 127.0.0.1:8002 check
    server app2 127.0.0.1:8003 check
    server app3 127.0.0.1:8004 check
```

## 📊 Monitoring & Observability

### Health Monitoring

#### Health Check Endpoint
```bash
# Basic health check
curl http://localhost:8002/health
# Response: {"status":"healthy"}

# Detailed server info
curl http://localhost:8002/functions
# Response: {"available_functions":[...]}
```

#### Health Check Script
```bash
#!/bin/bash
# health_check.sh

ENDPOINT="http://localhost:8002/health"
TIMEOUT=10

if curl -f --max-time $TIMEOUT "$ENDPOINT" > /dev/null 2>&1; then
    echo "✅ Server is healthy"
    exit 0
else
    echo "❌ Server is unhealthy"
    exit 1
fi
```

### Logging Configuration

#### Production Logging
```python
# logging_config.py
import logging
import logging.handlers

def setup_production_logging():
    """Configure production logging."""
    
    # Create logger
    logger = logging.getLogger('bayesian_mcp')
    logger.setLevel(logging.INFO)
    
    # Create file handler with rotation
    handler = logging.handlers.RotatingFileHandler(
        'bayesian_mcp.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger
```

#### Log Analysis
```bash
# Monitor logs
tail -f bayesian_mcp.log

# Search for errors
grep ERROR bayesian_mcp.log

# Monitor requests
grep "MCP request" bayesian_mcp.log | tail -10

# Performance analysis
grep "MCMC sampling completed" bayesian_mcp.log | \
    grep -o "took [0-9.]*s" | \
    awk '{sum+=$2; count++} END {print "Average:", sum/count, "seconds"}'
```

### Performance Monitoring

#### Metrics Collection
```python
# metrics.py
import time
import psutil
from typing import Dict, Any

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.total_processing_time = 0
    
    def record_request(self, processing_time: float):
        """Record request metrics."""
        self.request_count += 1
        self.total_processing_time += processing_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        uptime = time.time() - self.start_time
        avg_processing_time = (
            self.total_processing_time / self.request_count
            if self.request_count > 0 else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "avg_processing_time": avg_processing_time,
            "requests_per_second": self.request_count / uptime,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent()
        }

# Usage in server
monitor = PerformanceMonitor()
```

## 🔧 Integration Patterns

### Python Integration

#### Synchronous Client
```python
import requests
from typing import Dict, Any, Optional

class BayesianMCPClient:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_model(self, model_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Bayesian model."""
        response = self.session.post(f"{self.base_url}/mcp", json={
            "function_name": "create_model",
            "parameters": {
                "model_name": model_name,
                "variables": variables
            }
        })
        response.raise_for_status()
        return response.json()
    
    def update_beliefs(self, model_name: str, evidence: Dict[str, Any] = None,
                      sample_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Update model beliefs with MCMC sampling."""
        response = self.session.post(f"{self.base_url}/mcp", json={
            "function_name": "update_beliefs",
            "parameters": {
                "model_name": model_name,
                "evidence": evidence or {},
                "sample_kwargs": sample_kwargs or {}
            }
        })
        response.raise_for_status()
        return response.json()
    
    def predict(self, model_name: str, variables: list,
               conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate predictions from model."""
        response = self.session.post(f"{self.base_url}/mcp", json={
            "function_name": "predict",
            "parameters": {
                "model_name": model_name,
                "variables": variables,
                "conditions": conditions or {}
            }
        })
        response.raise_for_status()
        return response.json()

# Usage example
client = BayesianMCPClient()

# A/B test analysis
client.create_model("ab_test", {
    "rate_a": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}},
    "rate_b": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}},
    "obs_a": {"distribution": "binomial", "params": {"n": 1000, "p": "rate_a"}, "observed": 85},
    "obs_b": {"distribution": "binomial", "params": {"n": 1000, "p": "rate_b"}, "observed": 112}
})

result = client.update_beliefs("ab_test")
posterior = result["posterior"]
print(f"Conversion A: {posterior['rate_a']['mean']:.1%}")
print(f"Conversion B: {posterior['rate_b']['mean']:.1%}")
```

#### Async Client
```python
import asyncio
import aiohttp
from typing import Dict, Any

class AsyncBayesianMCPClient:
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def create_model(self, model_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Bayesian model asynchronously."""
        async with self.session.post(f"{self.base_url}/mcp", json={
            "function_name": "create_model",
            "parameters": {
                "model_name": model_name,
                "variables": variables
            }
        }) as response:
            return await response.json()
    
    async def update_beliefs(self, model_name: str, evidence: Dict[str, Any] = None,
                           sample_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Update model beliefs asynchronously."""
        async with self.session.post(f"{self.base_url}/mcp", json={
            "function_name": "update_beliefs",
            "parameters": {
                "model_name": model_name,
                "evidence": evidence or {},
                "sample_kwargs": sample_kwargs or {}
            }
        }) as response:
            return await response.json()

# Usage example
async def run_analysis():
    async with AsyncBayesianMCPClient() as client:
        await client.create_model("async_test", {...})
        result = await client.update_beliefs("async_test")
        return result

# Run async analysis
result = asyncio.run(run_analysis())
```

### Node.js Integration

#### Express.js Middleware
```javascript
// bayesian-middleware.js
const axios = require('axios');

class BayesianMCPClient {
    constructor(baseURL = 'http://localhost:8002') {
        this.client = axios.create({ baseURL });
    }
    
    async createModel(modelName, variables) {
        const response = await this.client.post('/mcp', {
            function_name: 'create_model',
            parameters: { model_name: modelName, variables }
        });
        return response.data;
    }
    
    async updateBeliefs(modelName, evidence = {}, sampleKwargs = {}) {
        const response = await this.client.post('/mcp', {
            function_name: 'update_beliefs',
            parameters: {
                model_name: modelName,
                evidence,
                sample_kwargs: sampleKwargs
            }
        });
        return response.data;
    }
}

// Express middleware
function bayesianAnalysis(options = {}) {
    const client = new BayesianMCPClient(options.baseURL);
    
    return async (req, res, next) => {
        req.bayesian = client;
        next();
    };
}

module.exports = { BayesianMCPClient, bayesianAnalysis };

// Usage in Express app
const express = require('express');
const { bayesianAnalysis } = require('./bayesian-middleware');

const app = express();
app.use(express.json());
app.use(bayesianAnalysis());

app.post('/analyze-ab-test', async (req, res) => {
    try {
        const { controlConversions, testConversions, controlUsers, testUsers } = req.body;
        
        // Create A/B test model
        await req.bayesian.createModel('ab_test', {
            rate_a: { distribution: 'beta', params: { alpha: 1, beta: 1 } },
            rate_b: { distribution: 'beta', params: { alpha: 1, beta: 1 } },
            obs_a: { distribution: 'binomial', params: { n: controlUsers, p: 'rate_a' }, observed: controlConversions },
            obs_b: { distribution: 'binomial', params: { n: testUsers, p: 'rate_b' }, observed: testConversions }
        });
        
        // Update beliefs
        const result = await req.bayesian.updateBeliefs('ab_test');
        
        res.json({
            success: true,
            analysis: result.posterior
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
```

### REST API Integration

#### Generic HTTP Client
```bash
#!/bin/bash
# bayesian_client.sh

BASE_URL="http://localhost:8002"

# Function to create model
create_model() {
    local model_name=$1
    local variables_json=$2
    
    curl -X POST "$BASE_URL/mcp" \
        -H "Content-Type: application/json" \
        -d "{
            \"function_name\": \"create_model\",
            \"parameters\": {
                \"model_name\": \"$model_name\",
                \"variables\": $variables_json
            }
        }"
}

# Function to update beliefs
update_beliefs() {
    local model_name=$1
    
    curl -X POST "$BASE_URL/mcp" \
        -H "Content-Type: application/json" \
        -d "{
            \"function_name\": \"update_beliefs\",
            \"parameters\": {
                \"model_name\": \"$model_name\",
                \"evidence\": {},
                \"sample_kwargs\": {
                    \"draws\": 1000,
                    \"tune\": 500,
                    \"chains\": 2
                }
            }
        }"
}

# Example usage
VARIABLES='{
    "conversion_rate": {
        "distribution": "beta",
        "params": {"alpha": 1, "beta": 1}
    },
    "conversions": {
        "distribution": "binomial",
        "params": {"n": 1000, "p": "conversion_rate"},
        "observed": 127
    }
}'

echo "Creating model..."
create_model "business_analysis" "$VARIABLES"

echo "Updating beliefs..."
update_beliefs "business_analysis"
```

## 🛠️ Development Setup

### Local Development Environment

#### Development Installation
```bash
# Clone repository
git clone <repository-url>
cd bayes

# Create development environment
python3 -m venv dev_env
source dev_env/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

#### Development Server
```bash
# Start development server with auto-reload
python bayesian_mcp.py --host 127.0.0.1 --port 8002 --log-level debug

# Or use uvicorn directly for development
uvicorn bayesian_mcp.mcp.server:app --host 127.0.0.1 --port 8002 --reload
```

#### Code Quality Tools
```bash
# Format code
black bayesian_mcp/
isort bayesian_mcp/

# Type checking
mypy bayesian_mcp/

# Linting
flake8 bayesian_mcp/

# Testing
pytest tests/ -v
```

### Testing Framework

#### Unit Tests
```python
# tests/test_engine.py
import pytest
from bayesian_mcp.bayesian_engine.engine import BayesianEngine

def test_engine_initialization():
    """Test that engine initializes correctly."""
    engine = BayesianEngine()
    assert engine.belief_models == {}

def test_model_creation():
    """Test basic model creation."""
    engine = BayesianEngine()
    
    variables = {
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
    
    engine.create_model("test_model", variables)
    assert "test_model" in engine.belief_models

def test_belief_updating():
    """Test MCMC belief updating."""
    engine = BayesianEngine()
    
    # Create simple model
    variables = {
        "p": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}},
        "obs": {"distribution": "binomial", "params": {"n": 10, "p": "p"}, "observed": 7}
    }
    
    engine.create_model("test_model", variables)
    
    # Update beliefs with minimal sampling
    result = engine.update_beliefs("test_model", {}, {
        "draws": 100,
        "tune": 100,
        "chains": 1
    })
    
    assert "p" in result
    assert "mean" in result["p"]
    assert 0 < result["p"]["mean"] < 1

# Run tests
# pytest tests/test_engine.py -v
```

#### Integration Tests
```python
# tests/test_integration.py
import requests
import pytest
import time
import subprocess
import signal

@pytest.fixture(scope="module")
def server():
    """Start server for integration tests."""
    # Start server
    process = subprocess.Popen([
        "python", "bayesian_mcp.py", "--port", "8003"
    ])
    
    # Wait for startup
    time.sleep(5)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        assert response.status_code == 200
    except:
        process.terminate()
        pytest.fail("Server failed to start")
    
    yield "http://localhost:8003"
    
    # Cleanup
    process.terminate()
    process.wait()

def test_full_workflow(server):
    """Test complete A/B testing workflow."""
    base_url = server
    
    # Create model
    response = requests.post(f"{base_url}/mcp", json={
        "function_name": "create_model",
        "parameters": {
            "model_name": "integration_test",
            "variables": {
                "rate_a": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}},
                "rate_b": {"distribution": "beta", "params": {"alpha": 1, "beta": 1}},
                "obs_a": {"distribution": "binomial", "params": {"n": 100, "p": "rate_a"}, "observed": 8},
                "obs_b": {"distribution": "binomial", "params": {"n": 100, "p": "rate_b"}, "observed": 12}
            }
        }
    })
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    
    # Update beliefs
    response = requests.post(f"{base_url}/mcp", json={
        "function_name": "update_beliefs",
        "parameters": {
            "model_name": "integration_test",
            "evidence": {},
            "sample_kwargs": {"draws": 100, "tune": 100, "chains": 1}
        }
    })
    
    assert response.status_code == 200
    result = response.json()
    assert result["success"] == True
    assert "posterior" in result
    assert "rate_a" in result["posterior"]
    assert "rate_b" in result["posterior"]

# Run integration tests
# pytest tests/test_integration.py -v -s
```

## 🚨 Troubleshooting Guide

### Common Installation Issues

#### Python Version Conflicts
```bash
# Check Python version
python3 --version
# Should be 3.8 or higher

# If using wrong version
which python3
# Update PATH or use specific version
python3.9 -m venv venv
```

#### Package Installation Failures
```bash
# Update pip first
pip install --upgrade pip setuptools wheel

# Install with verbose output
pip install -e . -v

# Clear pip cache if needed
pip cache purge
```

#### PyMC Installation Issues
```bash
# PyMC requires specific dependencies
pip install pymc --no-deps
pip install pytensor>=2.30.0
pip install arviz>=0.14.0

# Or use conda
conda install -c conda-forge pymc
```

### Runtime Issues

#### Server Won't Start
```bash
# Check if port is in use
lsof -i :8002
netstat -tulpn | grep :8002

# Try different port
python bayesian_mcp.py --port 8003

# Check logs for errors
python bayesian_mcp.py --log-level debug
```

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify installation
pip show bayesian-mcp

# Reinstall if needed
pip uninstall bayesian-mcp
pip install -e .
```

#### MCMC Convergence Issues
```python
# Increase tuning samples
sample_kwargs = {
    "draws": 2000,
    "tune": 2000,  # Increase tuning
    "chains": 4,   # More chains
    "target_accept": 0.9  # Higher acceptance rate
}

# Check R-hat values in logs
# R-hat should be < 1.01 for convergence
```

#### Memory Issues
```bash
# Monitor memory usage