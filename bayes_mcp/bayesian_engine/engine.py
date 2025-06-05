"""Core Bayesian engine implementation."""

import time
import logging
import os
from typing import Dict, List, Any, Tuple, Optional

import pymc as pm
import numpy as np
import arviz as az

logger = logging.getLogger(__name__)

# Configure Logfire if available
try:
    import logfire
    
    # Only initialize if token is available
    if os.getenv("LOGFIRE_TOKEN"):
        logfire.configure(
            token=os.getenv("LOGFIRE_TOKEN"),
            project_name=os.getenv("LOGFIRE_PROJECT_NAME", "bayes-mcp"),
            service_name=os.getenv("LOGFIRE_SERVICE_NAME", "bayes-engine"),
            environment=os.getenv("LOGFIRE_ENVIRONMENT", "development")
        )
        logger.info("Logfire initialized for Bayesian Engine")
    else:
        logfire = None
except ImportError:
    logfire = None

class BayesianEngine:
    """Core engine for probabilistic reasoning using PyMC.
    
    This engine manages Bayesian models, performs belief updating with
    new evidence, and provides interfaces for model comparison and
    predictive inference.
    """
    
    def __init__(self):
        """Initialize the Bayesian engine."""
        self.belief_models = {}
        logger.info("Bayesian Engine initialized")
        if logfire:
            logfire.info("Bayesian Engine initialized", model_count=0)
    
    def create_model(self, model_name: str, variables: Dict[str, Dict]) -> None:
        """Create a Bayesian model with specified variables.
        
        Args:
            model_name: Unique identifier for the model
            variables: Dictionary defining model variables and their distributions
                Format: {
                    'var_name': {
                        'distribution': 'normal'|'beta'|'binomial'|etc,
                        'params': {'param1': value1, 'param2': value2},
                        'observed': optional_observed_data
                    }
                }
        """
        try:
            if logfire:
                with logfire.span("create_model", model_name=model_name, variable_count=len(variables)):
                    logfire.info(f"Creating Bayesian model: {model_name}", 
                               variable_names=list(variables.keys()))
            with pm.Model() as model:
                model_vars = {}
                observed_vars = {}
                
                # Sort variables: create priors first, then likelihoods
                prior_vars = {k: v for k, v in variables.items() if v.get('observed') is None}
                likelihood_vars = {k: v for k, v in variables.items() if v.get('observed') is not None}
                
                # Create prior distributions first
                for var_name, var_config in prior_vars.items():
                    try:
                        dist_type = var_config['distribution']
                        params = var_config['params']
                        
                        if dist_type == 'normal':
                            model_vars[var_name] = pm.Normal(var_name, **params)
                        elif dist_type == 'beta':
                            model_vars[var_name] = pm.Beta(var_name, **params)
                        elif dist_type == 'gamma':
                            model_vars[var_name] = pm.Gamma(var_name, **params)
                        elif dist_type == 'uniform':
                            model_vars[var_name] = pm.Uniform(var_name, **params)
                        elif dist_type == 'halfnormal':
                            model_vars[var_name] = pm.HalfNormal(var_name, **params)
                        elif dist_type == 'exponential':
                            model_vars[var_name] = pm.Exponential(var_name, **params)
                        elif dist_type == 'deterministic':
                            # Handle deterministic transformations
                            expr_str = params.get('expr', '')
                            if expr_str:
                                # Simple expression parsing for common cases
                                expr = self._parse_expression(expr_str, model_vars)
                                model_vars[var_name] = pm.Deterministic(var_name, expr)
                        else:
                            logger.warning(f"Unsupported prior distribution: {dist_type}")
                            
                    except Exception as e:
                        logger.error(f"Error creating variable {var_name}: {e}")
                        raise
                
                # Create likelihood distributions
                for var_name, var_config in likelihood_vars.items():
                    try:
                        dist_type = var_config['distribution']
                        params = var_config['params'].copy()
                        observed = var_config['observed']
                        
                        # Resolve parameter references to other variables
                        for param_name, param_value in params.items():
                            if isinstance(param_value, str) and param_value in model_vars:
                                params[param_name] = model_vars[param_value]
                        
                        if dist_type == 'normal':
                            observed_vars[var_name] = pm.Normal(var_name, observed=observed, **params)
                        elif dist_type == 'binomial':
                            observed_vars[var_name] = pm.Binomial(var_name, observed=observed, **params)
                        elif dist_type == 'bernoulli':
                            observed_vars[var_name] = pm.Bernoulli(var_name, observed=observed, **params)
                        elif dist_type == 'poisson':
                            observed_vars[var_name] = pm.Poisson(var_name, observed=observed, **params)
                        else:
                            logger.warning(f"Unsupported likelihood distribution: {dist_type}")
                            
                    except Exception as e:
                        logger.error(f"Error creating likelihood {var_name}: {e}")
                        raise
                
                # Store the model
                self.belief_models[model_name] = {
                    'model': model,
                    'vars': model_vars,
                    'observed': observed_vars,
                    'created_at': time.time()
                }
                
                logger.info(f"Created model: {model_name} with {len(model_vars)} variables")
                
                if logfire:
                    logfire.info(f"Model created successfully",
                               model_name=model_name,
                               prior_count=len(model_vars),
                               likelihood_count=len(observed_vars))
                    logfire.metric("models.created", 1, unit="count", model_name=model_name)
                    logfire.metric("model.variables", len(model_vars) + len(observed_vars), 
                                 unit="count", model_name=model_name)
                
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            if logfire:
                logfire.exception(f"Model creation failed",
                                error_type=type(e).__name__,
                                model_name=model_name)
            raise
    
    def _parse_expression(self, expr_str: str, variables: Dict[str, Any]) -> Any:
        """Parse simple mathematical expressions for deterministic variables."""
        # Handle common expressions like "rate_b - rate_a" or "(rate_b - rate_a) / rate_a"
        expr = expr_str
        
        # Replace variable names with their PyMC objects
        for var_name, var_obj in variables.items():
            expr = expr.replace(var_name, f"variables['{var_name}']")
        
        try:
            # Evaluate the expression safely
            return eval(expr, {"__builtins__": {}}, {"variables": variables, "pm": pm})
        except Exception as e:
            logger.error(f"Failed to parse expression '{expr_str}': {e}")
            raise ValueError(f"Invalid expression: {expr_str}")
    
    def update_beliefs(self, model_name: str, 
                       evidence: Dict[str, Any],
                       sample_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update beliefs given new evidence using MCMC sampling.
        
        Args:
            model_name: Name of the model to update
            evidence: Additional evidence (usually empty for models with observed data)
            sample_kwargs: Optional parameters for MCMC sampling
            
        Returns:
            Dictionary with posterior summaries for all variables
        """
        if model_name not in self.belief_models:
            raise ValueError(f"Model '{model_name}' not found")
            
        model_data = self.belief_models[model_name]
        
        # Default sampling parameters
        default_kwargs = {
            'draws': 1000,
            'tune': 1000,
            'chains': 2,
            'cores': 1,
            'return_inferencedata': True,
            'progressbar': False
        }
        
        if sample_kwargs:
            default_kwargs.update(sample_kwargs)
        
        try:
            if logfire:
                with logfire.span("update_beliefs", 
                                model_name=model_name,
                                draws=default_kwargs['draws'],
                                chains=default_kwargs['chains']):
                    logfire.info(f"Starting MCMC sampling",
                               model_name=model_name,
                               sampling_params=default_kwargs)
                    
                    start_time = time.time()
                    
                    with model_data['model']:
                        # Run MCMC sampling
                        logger.info(f"Starting MCMC sampling for model '{model_name}'")
                        trace = pm.sample(**default_kwargs)
                        
                        # Store trace for later use
                        model_data['trace'] = trace
                        
                        # Extract posterior statistics
                        posterior_stats = {}
                        for var_name in trace.posterior.data_vars:
                            samples = trace.posterior[var_name].values
                            
                            # Flatten samples from all chains
                            flat_samples = samples.flatten()
                            
                            posterior_stats[var_name] = {
                                "mean": float(np.mean(flat_samples)),
                                "std": float(np.std(flat_samples)),
                                "median": float(np.median(flat_samples)),
                                "q025": float(np.percentile(flat_samples, 2.5)),
                                "q975": float(np.percentile(flat_samples, 97.5)),
                                "samples": flat_samples[:100].tolist()  # Limit samples for JSON response
                            }
                        
                        sampling_time = time.time() - start_time
                        
                        logger.info(f"MCMC sampling completed for model '{model_name}'")
                        
                        logfire.info(f"MCMC sampling completed successfully",
                                   model_name=model_name,
                                   sampling_time_ms=sampling_time * 1000,
                                   variables_sampled=list(posterior_stats.keys()))
                        logfire.metric("mcmc.sampling_time", sampling_time * 1000, 
                                     unit="ms", model_name=model_name)
                        logfire.metric("mcmc.total_samples", 
                                     default_kwargs['draws'] * default_kwargs['chains'],
                                     unit="count", model_name=model_name)
                        
                        return posterior_stats
            else:
                # Original code without Logfire
                with model_data['model']:
                    # Run MCMC sampling
                    logger.info(f"Starting MCMC sampling for model '{model_name}'")
                    trace = pm.sample(**default_kwargs)
                    
                    # Store trace for later use
                    model_data['trace'] = trace
                    
                    # Extract posterior statistics
                    posterior_stats = {}
                    for var_name in trace.posterior.data_vars:
                        samples = trace.posterior[var_name].values
                        
                        # Flatten samples from all chains
                        flat_samples = samples.flatten()
                        
                        posterior_stats[var_name] = {
                            "mean": float(np.mean(flat_samples)),
                            "std": float(np.std(flat_samples)),
                            "median": float(np.median(flat_samples)),
                            "q025": float(np.percentile(flat_samples, 2.5)),
                            "q975": float(np.percentile(flat_samples, 97.5)),
                            "samples": flat_samples[:100].tolist()  # Limit samples for JSON response
                        }
                    
                    logger.info(f"MCMC sampling completed for model '{model_name}'")
                    return posterior_stats
                
        except Exception as e:
            logger.error(f"MCMC sampling failed for model '{model_name}': {e}")
            if logfire:
                logfire.exception(f"MCMC sampling failed",
                                error_type=type(e).__name__,
                                model_name=model_name)
            raise
    
    def predict(self, model_name: str, 
                variables: List[str],
                conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate predictions using the posterior distribution.
        
        Args:
            model_name: Name of the model to use for prediction
            variables: List of variables to predict
            conditions: Optional conditions for prediction
            
        Returns:
            Dictionary with prediction summaries
        """
        if model_name not in self.belief_models:
            raise ValueError(f"Model '{model_name}' not found")
            
        model_data = self.belief_models[model_name]
        
        if 'trace' not in model_data:
            raise ValueError(f"Model '{model_name}' has not been updated with beliefs yet")
        
        try:
            trace = model_data['trace']
            predictions = {}
            
            # For now, return posterior statistics for requested variables
            for var_name in variables:
                if var_name in trace.posterior.data_vars:
                    samples = trace.posterior[var_name].values.flatten()
                    predictions[var_name] = {
                        "mean": float(np.mean(samples)),
                        "std": float(np.std(samples)),
                        "median": float(np.median(samples)),
                        "q025": float(np.percentile(samples, 2.5)),
                        "q975": float(np.percentile(samples, 97.5))
                    }
                else:
                    logger.warning(f"Variable '{var_name}' not found in posterior")
                    
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed for model '{model_name}': {e}")
            raise
    
    def compare_models(self, model_names: List[str], 
                       metric: str = "waic") -> Dict[str, Any]:
        """Compare multiple models using information criteria.
        
        Args:
            model_names: List of model names to compare
            metric: Comparison metric ('waic' or 'loo')
            
        Returns:
            Dictionary with comparison results
        """
        # Validate models
        for name in model_names:
            if name not in self.belief_models:
                raise ValueError(f"Model '{name}' not found")
            if 'trace' not in self.belief_models[name]:
                raise ValueError(f"Model '{name}' has not been updated with beliefs yet")
        
        try:
            results = {}
            
            for model_name in model_names:
                trace = self.belief_models[model_name]['trace']
                
                if metric == "waic":
                    waic_result = az.waic(trace)
                    results[model_name] = {
                        "waic": float(waic_result.waic),
                        "waic_se": float(waic_result.waic_se),
                        "p_waic": float(waic_result.p_waic)
                    }
                elif metric == "loo":
                    loo_result = az.loo(trace)
                    results[model_name] = {
                        "loo": float(loo_result.loo),
                        "loo_se": float(loo_result.loo_se),
                        "p_loo": float(loo_result.p_loo)
                    }
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            
            return {"metric": metric, "results": results}
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise
    
    def get_model_names(self) -> List[str]:
        """Get names of all available models."""
        return list(self.belief_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_name not in self.belief_models:
            raise ValueError(f"Model '{model_name}' not found")
            
        model_data = self.belief_models[model_name]
        return {
            "name": model_name,
            "variables": list(model_data['vars'].keys()),
            "observed": list(model_data['observed'].keys()),
            "created_at": model_data['created_at'],
            "has_trace": 'trace' in model_data
        }