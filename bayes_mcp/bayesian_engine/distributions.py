"""Distribution utilities for the Bayesian engine."""

import numpy as np
import pymc as pm
import arviz as az
from typing import Dict, Any, Optional


def extract_posterior_stats(trace: az.InferenceData, var_name: str) -> Dict[str, Any]:
    """Extract summary statistics for a posterior variable.
    
    Args:
        trace: ArviZ InferenceData object containing posterior samples
        var_name: Name of the variable to extract
    
    Returns:
        Dictionary with posterior statistics
    """
    if var_name not in trace.posterior.data_vars:
        return {}
    
    posterior = trace.posterior[var_name]
    samples = posterior.values.flatten()
    
    # Basic statistics
    result = {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "median": float(np.median(samples)),
        "q025": float(np.percentile(samples, 2.5)),
        "q975": float(np.percentile(samples, 97.5)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples))
    }
    
    # Add convergence diagnostics if available
    try:
        ess = az.ess(trace, var_names=[var_name])
        rhat = az.rhat(trace, var_names=[var_name])
        
        if var_name in ess:
            result["ess"] = float(ess[var_name].values)
        if var_name in rhat:
            result["r_hat"] = float(rhat[var_name].values)
            
    except Exception:
        # Skip diagnostics if they fail
        pass
    
    return result


def validate_distribution_params(dist_type: str, params: Dict[str, Any]) -> bool:
    """Validate parameters for a given distribution type.
    
    Args:
        dist_type: Type of distribution
        params: Parameters to validate
        
    Returns:
        True if parameters are valid
        
    Raises:
        ValueError: If parameters are invalid
    """
    required_params = {
        'normal': ['mu', 'sigma'],
        'beta': ['alpha', 'beta'],
        'gamma': ['alpha', 'beta'],
        'uniform': ['lower', 'upper'],
        'exponential': ['lam'],
        'poisson': ['mu'],
        'bernoulli': ['p'],
        'binomial': ['n', 'p'],
        'halfnormal': ['sigma'],
        'lognormal': ['mu', 'sigma']
    }
    
    if dist_type not in required_params:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    required = required_params[dist_type]
    missing = [p for p in required if p not in params]
    
    if missing:
        raise ValueError(f"Missing required parameters for {dist_type}: {missing}")
    
    return True


def get_supported_distributions() -> Dict[str, Dict[str, Any]]:
    """Get information about supported distributions.
    
    Returns:
        Dictionary with distribution information
    """
    return {
        'normal': {
            'params': ['mu', 'sigma'],
            'description': 'Normal (Gaussian) distribution'
        },
        'beta': {
            'params': ['alpha', 'beta'],
            'description': 'Beta distribution (for probabilities)'
        },
        'gamma': {
            'params': ['alpha', 'beta'],
            'description': 'Gamma distribution'
        },
        'uniform': {
            'params': ['lower', 'upper'],
            'description': 'Uniform distribution'
        },
        'exponential': {
            'params': ['lam'],
            'description': 'Exponential distribution'
        },
        'poisson': {
            'params': ['mu'],
            'description': 'Poisson distribution'
        },
        'bernoulli': {
            'params': ['p'],
            'description': 'Bernoulli distribution'
        },
        'binomial': {
            'params': ['n', 'p'],
            'description': 'Binomial distribution'
        },
        'halfnormal': {
            'params': ['sigma'],
            'description': 'Half-normal distribution'
        },
        'deterministic': {
            'params': ['expr'],
            'description': 'Deterministic transformation'
        }
    }