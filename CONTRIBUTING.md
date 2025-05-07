# Contributing to Bayesian MCP

Thank you for your interest in contributing to Bayesian MCP! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up a development environment
4. Create a feature branch
5. Make your changes
6. Submit a pull request

## Development Environment

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/bayesian-mcp.git
cd bayesian-mcp

# Install development dependencies
pip install -e ".[dev]"
```

## Code Style

We follow PEP 8 style guidelines for Python code. Please ensure your code adheres to these standards.

## Testing

Before submitting a pull request, please run the tests:

```bash
pytest
```

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the examples if necessary
3. The PR should work for Python 3.9 and above
4. Make sure all tests pass
5. Update documentation as needed

## Feature Requests

We welcome feature requests! Please submit them as issues with the tag [FEATURE REQUEST].

## Bug Reports

When reporting bugs, please include:

- A clear and descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Actual behavior
- Any error messages or logs
- Your environment (Python version, OS, etc.)

## License

By contributing, you agree that your contributions will be licensed under the project's MIT license.