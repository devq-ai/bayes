#!/usr/bin/env python3
"""
Command-line interface for the Bayesian MCP server.

This script provides a simple entry point for starting the MCP server.
"""

import sys
from src.main import main

if __name__ == "__main__":
    sys.exit(main())