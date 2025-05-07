#!/usr/bin/env python3
"""
Example client for the Bayesian MCP server.

This example demonstrates how to use the Bayesian MCP server to create
and update a simple linear regression model.
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
    """Run the example client."""
    parser = argparse.ArgumentParser(description="Bayesian MCP Client Example")
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
    
    # Generate synthetic data for linear regression
    print("Generating synthetic data for linear regression...")
    np.random.seed(42)
    n_samples = 50
    x = np.linspace(0, 10, n_samples)
    true_intercept = 2.0
    true_slope = 0.5
    sigma = 1.0
    
    # y = mx + b + noise
    y = true_slope * x + true_intercept + np.random.normal(0, sigma, n_samples)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.plot(x, true_slope * x + true_intercept, 'r', label=f'True: y = {true_slope}x + {true_intercept}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Synthetic Linear Regression Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot to display later
    data_plot_buffer = BytesIO()
    plt.savefig(data_plot_buffer, format='png')
    data_plot_buffer.seek(0)
    data_plot_b64 = base64.b64encode(data_plot_buffer.read()).decode('utf-8')
    plt.close()
    
    # Display the data
    print("Data plot saved. Let's create a Bayesian model for this data.")
    
    # Create a linear regression model
    print("\nCreating linear regression model...")
    model_name = "linear_regression_example"
    
    # Define the model variables
    variables = {
        "intercept": {
            "distribution": "normal",
            "params": {"mu": 0, "sigma": 10}
        },
        "slope": {
            "distribution": "normal",
            "params": {"mu": 0, "sigma": 10}
        },
        "sigma": {
            "distribution": "halfnormal",
            "params": {"sigma": 5}
        },
        "likelihood": {
            "distribution": "normal",
            "params": {
                "mu": "intercept + slope * x_data",
                "sigma": "sigma"
            },
            "observed": y.tolist()
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
    
    # Update beliefs with the data
    print("\nUpdating model with data...")
    update_beliefs_request = {
        "function_name": "update_beliefs",
        "parameters": {
            "model_name": model_name,
            "evidence": {
                "x_data": x.tolist()
            },
            "sample_kwargs": {
                "draws": 1000,
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
    
    # Create a visualization
    print("\nCreating visualization of the posterior distribution...")
    visualization_request = {
        "function_name": "create_visualization",
        "parameters": {
            "model_name": model_name,
            "plot_type": "trace",
            "variables": ["intercept", "slope", "sigma"]
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
        with open("posterior_trace.png", "wb") as f:
            f.write(base64.b64decode(image_data))
        print("Visualization saved to 'posterior_trace.png'")
    
    # Make predictions
    print("\nMaking predictions with the model...")
    predict_request = {
        "function_name": "predict",
        "parameters": {
            "model_name": model_name,
            "variables": ["intercept", "slope"],
            "conditions": {}
        }
    }
    
    response = requests.post(
        f"{base_url}/mcp",
        json=predict_request
    )
    
    if response.status_code != 200 or not response.json().get("success", False):
        print(f"Error making predictions: {response.json().get('message', 'Unknown error')}")
        return 1
        
    predictions = response.json().get("predictions", {})
    
    # Get the mean values for intercept and slope
    intercept_mean = predictions.get("intercept", {}).get("mean", 0)
    slope_mean = predictions.get("slope", {}).get("mean", 0)
    
    print(f"\nPredicted relationship: y = {slope_mean:.4f}x + {intercept_mean:.4f}")
    print(f"True relationship:     y = {true_slope}x + {true_intercept}")
    
    print("\nExample complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())