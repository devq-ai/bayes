#!/usr/bin/env python3
"""
Example client for the Bayesian MCP server demonstrating A/B testing.

This example shows how to use the Bayesian MCP server to analyze 
the results of an A/B test using Bayesian inference.
"""

import argparse
import json
import sys
import time
from typing import Dict, Any, List

import numpy as np
import requests
import matplotlib.pyplot as plt
from IPython.display import Image, display
import base64
from io import BytesIO


def main():
    """Run the A/B test example client."""
    parser = argparse.ArgumentParser(description="Bayesian A/B Test Example")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="URL of the Bayesian MCP server"
    )
    args = parser.parse_args()
    
    base_url = args.url
    
    # Check if the server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"Server is not healthy. Status code: {response.status_code}")
            return 1
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the server is running and accessible.")
        return 1
        
    print("Connected to Bayesian MCP server successfully.")
    
    # Generate synthetic data for A/B test
    print("Generating synthetic data for A/B test...")
    np.random.seed(42)
    
    # True conversion rates
    true_rate_a = 0.10  # 10% conversion for variant A
    true_rate_b = 0.12  # 12% conversion for variant B
    
    # Sample sizes
    n_a = 1000  # users in variant A
    n_b = 1000  # users in variant B
    
    # Generate conversion data
    conversions_a = np.random.binomial(1, true_rate_a, n_a)
    conversions_b = np.random.binomial(1, true_rate_b, n_b)
    
    # Summary statistics
    total_a = len(conversions_a)
    total_b = len(conversions_b)
    successes_a = np.sum(conversions_a)
    successes_b = np.sum(conversions_b)
    
    print(f"Variant A: {successes_a} conversions out of {total_a} ({successes_a/total_a*100:.2f}%)")
    print(f"Variant B: {successes_b} conversions out of {total_b} ({successes_b/total_b*100:.2f}%)")
    
    # Create a Bayesian model for A/B test
    print("\nCreating Bayesian A/B test model...")
    model_name = "ab_test_example"
    
    # Define the model variables
    variables = {
        "rate_a": {
            "distribution": "beta",
            "params": {"alpha": 1, "beta": 1}
        },
        "rate_b": {
            "distribution": "beta",
            "params": {"alpha": 1, "beta": 1}
        },
        "likelihood_a": {
            "distribution": "bernoulli",
            "params": {"p": "rate_a"},
            "observed": conversions_a.tolist()
        },
        "likelihood_b": {
            "distribution": "bernoulli",
            "params": {"p": "rate_b"},
            "observed": conversions_b.tolist()
        },
        "difference": {
            "distribution": "deterministic",
            "params": {"expr": "rate_b - rate_a"}
        },
        "relative_improvement": {
            "distribution": "deterministic",
            "params": {"expr": "(rate_b - rate_a) / rate_a"}
        }
    }
    
    # Create the model
    create_model_request = {
        "function_name": "create_model",
        "parameters": {
            "model_name": model_name,
            "variables": variables
        }
    }
    
    response = requests.post(
        f"{base_url}/mcp",
        json=create_model_request
    )
    
    if response.status_code != 200 or not response.json().get("success", False):
        print(f"Error creating model: {response.json().get('message', 'Unknown error')}")
        return 1
        
    print(f"Model '{model_name}' created successfully.")
    
    # Update beliefs
    print("\nUpdating model with data...")
    update_beliefs_request = {
        "function_name": "update_beliefs",
        "parameters": {
            "model_name": model_name,
            "evidence": {},
            "sample_kwargs": {
                "draws": 2000,
                "tune": 1000,
                "chains": 2,
                "cores": 1
            }
        }
    }
    
    response = requests.post(
        f"{base_url}/mcp",
        json=update_beliefs_request
    )
    
    if response.status_code != 200 or not response.json().get("success", False):
        print(f"Error updating beliefs: {response.json().get('message', 'Unknown error')}")
        return 1
        
    posterior = response.json().get("posterior", {})
    
    print("\nPosterior distribution summary:")
    for param, stats in posterior.items():
        if isinstance(stats, dict) and "mean" in stats:
            print(f"  {param}: mean = {stats['mean']:.4f}, std = {stats['std']:.4f}")
    
    # Calculate probability of improvement
    diff = posterior.get("difference", {})
    if "samples" in diff:
        samples = diff["samples"]
        prob_improvement = sum(s > 0 for s in samples) / len(samples)
        print(f"\nProbability that B is better than A: {prob_improvement:.2%}")
    
    # Create a visualization
    print("\nCreating visualization of the posterior distributions...")
    visualization_request = {
        "function_name": "create_visualization",
        "parameters": {
            "model_name": model_name,
            "plot_type": "posterior",
            "variables": ["rate_a", "rate_b", "difference", "relative_improvement"]
        }
    }
    
    response = requests.post(
        f"{base_url}/mcp",
        json=visualization_request
    )
    
    if response.status_code != 200 or not response.json().get("success", False):
        print(f"Error creating visualization: {response.json().get('message', 'Unknown error')}")
        return 1
        
    # Save the visualization to a file
    image_data = response.json().get("image_data")
    if image_data:
        with open("ab_test_posterior.png", "wb") as f:
            f.write(base64.b64decode(image_data))
        print("Visualization saved to 'ab_test_posterior.png'")
    
    print("\nA/B test example complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())