"""
MCP server implementation for the Bayesian Engine.

This module implements a FastAPI-based MCP server that exposes the Bayesian
Engine's functionality through a standardized API.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

from .handlers import handle_mcp_request

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Logfire if available
logfire_initialized = False
try:
    import logfire
    
    # Initialize Logfire if token is available
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if logfire_token:
        logfire.configure(
            token=logfire_token,
            service_name=os.getenv("LOGFIRE_SERVICE_NAME", "bayes-mcp-server"),
            environment=os.getenv("LOGFIRE_ENVIRONMENT", "development")
        )
        logfire_initialized = True
        logger.info("Logfire initialized successfully")
    else:
        logger.info("Logfire token not found, running without Logfire")
except ImportError:
    logger.info("Logfire not installed, running without observability")
    logfire = None

# Create FastAPI app
app = FastAPI(
    title="Bayesian MCP Server",
    description="Model Calling Protocol server for Bayesian inference",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with Logfire if available
if logfire and logfire_token:
    logfire.instrument_fastapi(app)
    logger.info("FastAPI instrumented with Logfire")


@app.get("/")
async def root():
    """Root endpoint for the API."""
    if logfire:
        logfire.info("Root endpoint accessed")
    return {"message": "Bayesian MCP Server is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """
    MCP endpoint for handling model calls.
    
    This endpoint receives MCP standard requests and routes them to the
    appropriate handler functions based on the function name.
    """
    try:
        # Parse the request body
        body = await request.json()
        
        # Validate structure
        if "function_name" not in body or "parameters" not in body:
            if logfire:
                logfire.warning("Invalid MCP request format", 
                               missing_fields=["function_name" if "function_name" not in body else None,
                                             "parameters" if "parameters" not in body else None])
            raise HTTPException(
                status_code=400, 
                detail="Invalid request format. Must include 'function_name' and 'parameters'."
            )
            
        # Extract function name and parameters
        function_name = body["function_name"]
        parameters = body["parameters"]
        
        # Log the request
        logger.info(f"MCP request: {function_name}")
        if logfire:
            with logfire.span("mcp_request", function_name=function_name):
                logfire.info(f"Processing MCP request", 
                           function_name=function_name,
                           parameter_count=len(parameters))
                
                # Handle the request
                result = handle_mcp_request(function_name, parameters)
                
                # Log successful completion
                logfire.info(f"MCP request completed successfully",
                           function_name=function_name,
                           success=result.get("success", True))
        else:
            # Handle the request without Logfire
            result = handle_mcp_request(function_name, parameters)
        
        # Return the response
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error handling MCP request: {str(e)}")
        if logfire:
            logfire.exception(f"MCP request error",
                            error_type=type(e).__name__,
                            error_message=str(e),
                            function_name=body.get("function_name", "unknown"))
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error processing request: {str(e)}",
                "data": None
            }
        )


@app.get("/schema")
async def get_schema():
    """Get the API schema for the MCP server."""
    from ..schemas.mcp_schemas import CreateModelRequest, UpdateBeliefRequest, PredictRequest
    
    schema = {
        "create_model": CreateModelRequest.schema(),
        "update_beliefs": UpdateBeliefRequest.schema(),
        "predict": PredictRequest.schema(),
        # Add other schemas as needed
    }
    
    return JSONResponse(content=schema)


@app.get("/functions")
async def list_functions():
    """List all available functions in the MCP server."""
    from .handlers import FUNCTION_MAP
    
    functions = list(FUNCTION_MAP.keys())
    return {"available_functions": functions}


def start_server(host: str = "127.0.0.1", port: int = 8000, log_level: str = "info"):
    """
    Start the MCP server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        log_level: Logging level
    """
    uvicorn.run(
        "bayes_mcp.mcp.server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=True
    )