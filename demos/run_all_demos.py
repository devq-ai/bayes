#!/usr/bin/env python3
"""
Comprehensive Demo Runner for Bayesian MCP Tool

This script demonstrates the capabilities of the Bayesian MCP tool through
various real-world scenarios including A/B testing, medical diagnosis,
financial risk assessment, and machine learning parameter estimation.
"""

import argparse
import json
import sys
import time
import asyncio
from typing import Dict, Any, List
import subprocess
import signal
import os

import numpy as np
import requests
import matplotlib.pyplot as plt
from IPython.display import Image, display
import base64
from io import BytesIO
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BayesianMCPDemo:
    """Demo runner for Bayesian MCP tool capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.server_process = None
        
    def start_server(self, timeout: int = 30):
        """Start the Bayesian MCP server."""
        print("🚀 Starting Bayesian MCP Server...")
        
        # Start server in background
        self.server_process = subprocess.Popen(
            ["python", "bayesian_mcp.py", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready
        for i in range(timeout):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"✅ Server ready at {self.base_url}")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
            print(f"⏳ Waiting for server... ({i+1}/{timeout})")
            
        print("❌ Server failed to start")
        return False
        
    def stop_server(self):
        """Stop the Bayesian MCP server."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("🛑 Server stopped")
            
    def make_request(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the MCP server."""
        request_data = {
            "function_name": function_name,
            "parameters": parameters
        }
        
        response = requests.post(f"{self.base_url}/mcp", json=request_data)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.text}")
            
        result = response.json()
        if not result.get("success", False):
            raise Exception(f"MCP request failed: {result.get('message', 'Unknown error')}")
            
        return result
        
    def demo_ab_testing(self):
        """Demo 1: Advanced A/B Testing with Bayesian Analysis."""
        print("\n" + "="*60)
        print("📊 DEMO 1: Bayesian A/B Testing")
        print("="*60)
        
        # Generate realistic A/B test data
        np.random.seed(42)
        
        # Scenario: Testing two website designs
        print("Scenario: Testing two website landing page designs")
        print("- Variant A: Current design")
        print("- Variant B: New design with better CTA button")
        
        # True conversion rates (unknown in real scenario)
        true_rate_a = 0.085  # 8.5% conversion
        true_rate_b = 0.112  # 11.2% conversion (31% relative improvement)
        
        # Sample sizes (realistic for a week-long test)
        n_a = 2500
        n_b = 2450
        
        # Generate conversion data
        conversions_a = np.random.binomial(1, true_rate_a, n_a)
        conversions_b = np.random.binomial(1, true_rate_b, n_b)
        
        successes_a = np.sum(conversions_a)
        successes_b = np.sum(conversions_b)
        
        observed_rate_a = successes_a / n_a
        observed_rate_b = successes_b / n_b
        
        print(f"\n📈 Test Results:")
        print(f"Variant A: {successes_a}/{n_a} conversions ({observed_rate_a:.2%})")
        print(f"Variant B: {successes_b}/{n_b} conversions ({observed_rate_b:.2%})")
        print(f"Observed lift: {(observed_rate_b - observed_rate_a)/observed_rate_a:.1%}")
        
        # Create Bayesian model
        print("\n🧠 Creating Bayesian A/B test model...")
        
        model_name = "advanced_ab_test"
        variables = {
            "rate_a": {
                "distribution": "beta",
                "params": {"alpha": 1, "beta": 1}  # Uninformative prior
            },
            "rate_b": {
                "distribution": "beta", 
                "params": {"alpha": 1, "beta": 1}
            },
            "likelihood_a": {
                "distribution": "binomial",
                "params": {"n": n_a, "p": "rate_a"},
                "observed": successes_a
            },
            "likelihood_b": {
                "distribution": "binomial",
                "params": {"n": n_b, "p": "rate_b"},
                "observed": successes_b
            },
            "difference": {
                "distribution": "deterministic",
                "params": {"expr": "rate_b - rate_a"}
            },
            "relative_lift": {
                "distribution": "deterministic", 
                "params": {"expr": "(rate_b - rate_a) / rate_a"}
            }
        }
        
        # Create model
        result = self.make_request("create_model", {
            "model_name": model_name,
            "variables": variables
        })
        print(f"✅ {result['message']}")
        
        # Update beliefs
        print("🔄 Running Bayesian inference...")
        result = self.make_request("update_beliefs", {
            "model_name": model_name,
            "evidence": {},
            "sample_kwargs": {
                "draws": 3000,
                "tune": 1500,
                "chains": 4,
                "cores": 1
            }
        })
        
        posterior = result["posterior"]
        
        # Analyze results
        print("\n📊 Bayesian Analysis Results:")
        for param in ["rate_a", "rate_b", "difference", "relative_lift"]:
            if param in posterior:
                stats = posterior[param]
                if param == "relative_lift":
                    print(f"  {param}: {stats['mean']:.1%} ± {stats['std']:.1%}")
                else:
                    print(f"  {param}: {stats['mean']:.4f} ± {stats['std']:.4f}")
                    
        # Calculate key business metrics
        if "difference" in posterior and "samples" in posterior["difference"]:
            diff_samples = posterior["difference"]["samples"]
            prob_b_better = sum(s > 0 for s in diff_samples) / len(diff_samples)
            
            # Expected loss calculations
            expected_loss_a = np.mean([max(0, s) for s in diff_samples])
            expected_loss_b = np.mean([max(0, -s) for s in diff_samples])
            
            print(f"\n🎯 Decision Metrics:")
            print(f"  Probability B > A: {prob_b_better:.1%}")
            print(f"  Expected loss if choosing A: {expected_loss_a:.4f}")
            print(f"  Expected loss if choosing B: {expected_loss_b:.4f}")
            
            if prob_b_better > 0.95:
                print("  📈 RECOMMENDATION: Deploy Variant B (High confidence)")
            elif prob_b_better > 0.80:
                print("  ⚠️  RECOMMENDATION: Deploy Variant B (Moderate confidence)")
            elif prob_b_better < 0.20:
                print("  📉 RECOMMENDATION: Keep Variant A")
            else:
                print("  🔄 RECOMMENDATION: Continue testing (Inconclusive)")
        
        # Create visualization
        print("\n📈 Creating posterior visualization...")
        result = self.make_request("create_visualization", {
            "model_name": model_name,
            "plot_type": "posterior",
            "variables": ["rate_a", "rate_b", "difference", "relative_lift"]
        })
        
        if result.get("image_data"):
            with open("demos/ab_test_results.png", "wb") as f:
                f.write(base64.b64decode(result["image_data"]))
            print("✅ Visualization saved: demos/ab_test_results.png")
            
    def demo_medical_diagnosis(self):
        """Demo 2: Medical Diagnosis with Bayesian Inference."""
        print("\n" + "="*60)
        print("🏥 DEMO 2: Medical Diagnosis Bayesian Inference")
        print("="*60)
        
        print("Scenario: COVID-19 rapid test interpretation")
        print("Using Bayesian inference to update disease probability given test results")
        
        # Test characteristics (realistic values)
        sensitivity = 0.85  # True positive rate
        specificity = 0.97  # True negative rate
        
        # Prior probability scenarios
        scenarios = [
            {"name": "Low prevalence community", "prior": 0.02},
            {"name": "High prevalence area", "prior": 0.15},
            {"name": "Symptomatic patient", "prior": 0.30},
            {"name": "Close contact exposure", "prior": 0.50}
        ]
        
        print(f"\n🧪 Test Characteristics:")
        print(f"  Sensitivity: {sensitivity:.0%} (detects positive cases)")
        print(f"  Specificity: {specificity:.0%} (correctly identifies negatives)")
        
        results = []
        
        for i, scenario in enumerate(scenarios):
            model_name = f"covid_diagnosis_{i}"
            prior_prob = scenario["prior"]
            
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"  Prior probability of disease: {prior_prob:.1%}")
            
            # Test both positive and negative results
            for test_result in ["positive", "negative"]:
                print(f"\n  🧪 Test Result: {test_result.upper()}")
                
                # Create Bayesian model
                variables = {
                    "has_disease": {
                        "distribution": "bernoulli",
                        "params": {"p": prior_prob}
                    },
                    "test_positive": {
                        "distribution": "bernoulli", 
                        "params": {
                            "p": "has_disease * sensitivity + (1 - has_disease) * (1 - specificity)"
                        },
                        "observed": 1 if test_result == "positive" else 0
                    }
                }
                
                # Manually calculate using Bayes' theorem for comparison
                if test_result == "positive":
                    likelihood = sensitivity * prior_prob + (1 - specificity) * (1 - prior_prob)
                    posterior_prob = (sensitivity * prior_prob) / likelihood
                else:
                    likelihood = (1 - sensitivity) * prior_prob + specificity * (1 - prior_prob)
                    posterior_prob = ((1 - sensitivity) * prior_prob) / likelihood
                
                print(f"    Bayesian calculation: {posterior_prob:.1%}")
                
                # Interpretation
                if test_result == "positive":
                    if posterior_prob > 0.95:
                        interpretation = "Very high probability - confirmatory test recommended"
                    elif posterior_prob > 0.80:
                        interpretation = "High probability - likely positive"
                    elif posterior_prob > 0.50:
                        interpretation = "Moderate probability - consider retesting"
                    else:
                        interpretation = "Low probability despite positive test"
                else:
                    if posterior_prob < 0.05:
                        interpretation = "Very low probability - likely negative"
                    elif posterior_prob < 0.20:
                        interpretation = "Low probability - probably negative"
                    else:
                        interpretation = "Moderate probability - consider retesting"
                        
                print(f"    Clinical interpretation: {interpretation}")
                
                results.append({
                    "scenario": scenario["name"],
                    "prior": prior_prob,
                    "test_result": test_result,
                    "posterior": posterior_prob,
                    "interpretation": interpretation
                })
        
        print(f"\n💡 Key Insights:")
        print("  • Prior probability strongly affects posterior probability")
        print("  • Positive tests in low-prevalence settings often false positives")
        print("  • Negative tests are more reliable due to high specificity")
        print("  • Clinical context is crucial for test interpretation")
        
    def demo_financial_risk(self):
        """Demo 3: Financial Risk Assessment."""
        print("\n" + "="*60)
        print("💰 DEMO 3: Financial Portfolio Risk Assessment")
        print("="*60)
        
        print("Scenario: Bayesian estimation of portfolio Value at Risk (VaR)")
        
        # Generate synthetic portfolio returns
        np.random.seed(123)
        n_days = 252  # Trading days in a year
        
        # Simulate returns for 3 assets
        assets = ["Tech Stock", "Bond Fund", "Real Estate"]
        true_means = [0.0008, 0.0003, 0.0005]  # Daily returns
        true_stds = [0.025, 0.008, 0.015]
        
        returns_data = {}
        for i, asset in enumerate(assets):
            returns_data[asset] = np.random.normal(true_means[i], true_stds[i], n_days)
            
        # Portfolio weights
        weights = [0.6, 0.3, 0.1]
        portfolio_returns = sum(w * returns_data[asset] for w, asset in zip(weights, assets))
        
        print(f"\n📊 Portfolio Composition:")
        for asset, weight in zip(assets, weights):
            print(f"  {asset}: {weight:.0%}")
            
        print(f"\n📈 Historical Performance (252 days):")
        for asset in assets:
            annual_return = np.mean(returns_data[asset]) * 252
            annual_vol = np.std(returns_data[asset]) * np.sqrt(252)
            print(f"  {asset}: {annual_return:.1%} return, {annual_vol:.1%} volatility")
            
        portfolio_annual_return = np.mean(portfolio_returns) * 252
        portfolio_annual_vol = np.std(portfolio_returns) * np.sqrt(252)
        print(f"  Portfolio: {portfolio_annual_return:.1%} return, {portfolio_annual_vol:.1%} volatility")
        
        # Bayesian VaR estimation
        model_name = "portfolio_var"
        
        print(f"\n🧠 Bayesian VaR Estimation...")
        
        # Create model for portfolio returns
        variables = {
            "mu": {
                "distribution": "normal",
                "params": {"mu": 0, "sigma": 0.01}  # Prior on mean return
            },
            "sigma": {
                "distribution": "halfnormal",
                "params": {"sigma": 0.05}  # Prior on volatility
            },
            "returns": {
                "distribution": "normal",
                "params": {"mu": "mu", "sigma": "sigma"},
                "observed": portfolio_returns.tolist()
            }
        }
        
        # Create and run model
        result = self.make_request("create_model", {
            "model_name": model_name,
            "variables": variables
        })
        
        result = self.make_request("update_beliefs", {
            "model_name": model_name,
            "evidence": {},
            "sample_kwargs": {
                "draws": 2000,
                "tune": 1000,
                "chains": 2
            }
        })
        
        posterior = result["posterior"]
        
        # Calculate VaR from posterior samples
        if "mu" in posterior and "sigma" in posterior:
            mu_samples = posterior["mu"]["samples"]
            sigma_samples = posterior["sigma"]["samples"]
            
            # Calculate VaR at different confidence levels
            confidence_levels = [0.95, 0.99]
            portfolio_value = 1000000  # $1M portfolio
            
            print(f"\n📊 Value at Risk (VaR) for ${portfolio_value:,} portfolio:")
            
            for conf_level in confidence_levels:
                var_samples = []
                for mu, sigma in zip(mu_samples, sigma_samples):
                    # VaR calculation: percentile of loss distribution
                    var_percentile = np.percentile(
                        np.random.normal(mu, sigma, 1000), 
                        (1 - conf_level) * 100
                    )
                    var_samples.append(-var_percentile * portfolio_value)
                
                var_mean = np.mean(var_samples)
                var_std = np.std(var_samples)
                
                print(f"  {conf_level:.0%} VaR: ${var_mean:,.0f} ± ${var_std:,.0f}")
                print(f"    Interpretation: {conf_level:.0%} confidence that daily loss won't exceed ${var_mean:,.0f}")
                
        print(f"\n💡 Risk Management Insights:")
        print("  • Bayesian approach provides uncertainty quantification")
        print("  • VaR estimates include parameter uncertainty")
        print("  • Can be updated daily with new market data")
        print("  • Helps set appropriate position sizes and stop-losses")
        
    def demo_ml_parameter_estimation(self):
        """Demo 4: Machine Learning Parameter Estimation."""
        print("\n" + "="*60)
        print("🤖 DEMO 4: Bayesian Machine Learning Parameter Estimation")
        print("="*60)
        
        print("Scenario: Bayesian Linear Regression with Uncertainty Quantification")
        
        # Generate synthetic ML dataset
        np.random.seed(456)
        n_samples = 100
        
        # True relationship: y = 2.5x + 1.2 + noise
        true_slope = 2.5
        true_intercept = 1.2
        noise_std = 0.8
        
        x = np.linspace(0, 10, n_samples)
        y = true_slope * x + true_intercept + np.random.normal(0, noise_std, n_samples)
        
        # Add some outliers to make it realistic
        outlier_indices = np.random.choice(n_samples, 5, replace=False)
        y[outlier_indices] += np.random.normal(0, 3, 5)
        
        print(f"\n📊 Dataset:")
        print(f"  Samples: {n_samples}")
        print(f"  True relationship: y = {true_slope}x + {true_intercept}")
        print(f"  Noise level: σ = {noise_std}")
        print(f"  Added {len(outlier_indices)} outliers")
        
        # Traditional least squares for comparison
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y)
        
        print(f"\n📈 Traditional Least Squares:")
        print(f"  Slope: {lr.coef_[0]:.3f}")
        print(f"  Intercept: {lr.intercept_:.3f}")
        print("  ⚠️  No uncertainty quantification!")
        
        # Bayesian approach
        print(f"\n🧠 Bayesian Linear Regression:")
        
        model_name = "bayesian_regression"
        variables = {
            "intercept": {
                "distribution": "normal",
                "params": {"mu": 0, "sigma": 10}  # Weakly informative prior
            },
            "slope": {
                "distribution": "normal", 
                "params": {"mu": 0, "sigma": 10}
            },
            "sigma": {
                "distribution": "halfnormal",
                "params": {"sigma": 5}  # Prior on noise
            },
            "likelihood": {
                "distribution": "normal",
                "params": {"mu": "intercept + slope * x_data", "sigma": "sigma"},
                "observed": y.tolist()
            }
        }
        
        # Create and run model
        result = self.make_request("create_model", {
            "model_name": model_name,
            "variables": variables
        })
        
        result = self.make_request("update_beliefs", {
            "model_name": model_name,
            "evidence": {"x_data": x.tolist()},
            "sample_kwargs": {
                "draws": 2000,
                "tune": 1000,
                "chains": 3
            }
        })
        
        posterior = result["posterior"]
        
        print(f"  Parameter Estimates:")
        for param in ["intercept", "slope", "sigma"]:
            if param in posterior:
                stats = posterior[param]
                credible_interval = [
                    stats["mean"] - 1.96 * stats["std"],
                    stats["mean"] + 1.96 * stats["std"]
                ]
                print(f"    {param}: {stats['mean']:.3f} ± {stats['std']:.3f}")
                print(f"      95% CI: [{credible_interval[0]:.3f}, {credible_interval[1]:.3f}]")
                
        # Predictive inference
        print(f"\n🔮 Predictive Inference:")
        x_new = [3.0, 7.5, 12.0]  # New x values for prediction
        
        if all(param in posterior for param in ["intercept", "slope", "sigma"]):
            intercept_samples = posterior["intercept"]["samples"]
            slope_samples = posterior["slope"]["samples"]
            sigma_samples = posterior["sigma"]["samples"]
            
            for x_pred in x_new:
                # Generate predictions accounting for all uncertainty
                predictions = []
                for i, s, sig in zip(intercept_samples, slope_samples, sigma_samples):
                    pred_mean = i + s * x_pred
                    # Add noise for predictive distribution
                    pred_sample = np.random.normal(pred_mean, sig)
                    predictions.append(pred_sample)
                    
                pred_mean = np.mean(predictions)
                pred_std = np.std(predictions)
                pred_ci = [pred_mean - 1.96 * pred_std, pred_mean + 1.96 * pred_std]
                
                print(f"    x = {x_pred}: y = {pred_mean:.2f} ± {pred_std:.2f}")
                print(f"      95% Prediction interval: [{pred_ci[0]:.2f}, {pred_ci[1]:.2f}]")
        
        print(f"\n💡 Advantages of Bayesian Approach:")
        print("  • Quantifies uncertainty in all parameters")
        print("  • Provides prediction intervals, not just point estimates")
        print("  • Can incorporate prior knowledge")
        print("  • Naturally handles model comparison")
        print("  • Robust to outliers with appropriate priors")
        
    def demo_sequential_learning(self):
        """Demo 5: Sequential Bayesian Learning."""
        print("\n" + "="*60)
        print("🔄 DEMO 5: Sequential Bayesian Learning")
        print("="*60)
        
        print("Scenario: Online conversion rate estimation with streaming data")
        print("Demonstrates how beliefs update as new data arrives")
        
        # Simulate streaming data
        np.random.seed(789)
        true_conversion_rate = 0.15
        
        # Simulate data arrival in batches
        batch_sizes = [50, 100, 200, 300, 500]
        
        print(f"\n📊 True conversion rate: {true_conversion_rate:.1%}")
        print("Simulating data arrival in batches...")
        
        # Prior beliefs
        prior_alpha = 1  # Prior successes
        prior_beta = 1   # Prior failures
        
        current_alpha = prior_alpha
        current_beta = prior_beta
        
        results = []
        
        for batch_num, batch_size in enumerate(batch_sizes):
            # Generate new data batch
            new_conversions = np.random.binomial(1, true_conversion_rate, batch_size)
            new_successes = np.sum(new_conversions)
            new_failures = batch_size - new_successes
            
            # Update beliefs (conjugate prior)
            current_alpha += new_successes
            current_beta += new_failures
            
            # Calculate posterior statistics
            posterior_mean = current_alpha / (current_alpha + current_beta)
            posterior_var = (current_alpha * current_beta) / \
                          ((current_alpha + current_beta)**2 * (current_alpha + current_beta + 1))
            posterior_std = np.sqrt(posterior_var)
            
            # Credible interval
            from scipy.stats import beta
            ci_lower = beta.ppf(0.025, current_alpha, current_beta)
            ci_upper = beta.ppf(0.975, current_alpha, current_beta)
            
            total_observations = current_alpha + current_beta - 2  # Subtract prior
            
            print(f"\n  📈 Batch {batch_num + 1}: {batch_size} new observations")
            print(f"    New conversions: {new_successes}/{batch_size} ({new_successes/batch_size:.1%})")
            print(f"    Total observations: {total_observations}")
            print(f"    Posterior estimate: {posterior_mean:.3f} ± {posterior_std:.3f}")
            print(f"    95% Credible interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
            
            # Decision making
            ci_width = ci_upper - ci_lower
            if ci_width < 0.02:  # High precision threshold
                decision = "🎯 PRECISE: Ready for decision making"
            elif ci_width < 0.05:
                decision = "⚠️  MODERATE: Consider more data"
            else:
                decision = "🔄 UNCERTAIN: Continue collecting data"
                
            print(f"    Decision status: {decision}")
            
            results.append({
                "batch": batch_num + 1,
                "total_obs": total_observations,
                "estimate": posterior_mean,
                "ci_width": ci_width,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            })
            
        print(f"\n📊 Learning Progression:")
        print("  Batch | Total Obs | Estimate | CI Width | Status")
        print("  ------|-----------|----------|----------|--------")
        for r in results:
            status = "✓" if r["ci_width"] < 0.02 else "○"
            print(f"  {r['batch']:5d} | {r['total_obs']:9d} | {r['estimate']:8.3f} | {r['ci_width']:8.3f} | {status}")
            
        print(f"\n💡 Sequential Learning Benefits:")
        print("  • Continuously updated beliefs as data arrives")
        print("  • Optimal stopping: know when enough data is collected")
        print("  • Early detection of changes in underlying rate")
        print("  • Efficient resource allocation")
        
    def run_all_demos(self):
        """Run all demonstrations."""
        print("🚀 Bayesian MCP Tool - Comprehensive Demonstration")
        print("="*60)
        print("This demo showcases the power of Bayesian reasoning for AI applications")
        print("Areas covered:")
        print("  1. 📊 A/B Testing with business decision making")
        print("  2. 🏥 Medical diagnosis with test interpretation") 
        print("  3. 💰 Financial risk assessment and VaR")
        print("  4. 🤖 Machine learning parameter estimation")
        print("  5. 🔄 Sequential learning with streaming data")
        print("="*60)
        
        try:
            # Start server
            if not self.start_server():
                return False
                
            # Run demos
            self.demo_ab_testing()
            self.demo_medical_diagnosis()
            self.demo_financial_risk()
            self.demo_ml_parameter_estimation()
            self.demo_sequential_learning()
            
            print("\n" + "="*60)
            print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\n💡 Key Takeaways:")
            print("  • Bayesian methods provide uncertainty quantification")
            print("  • Prior knowledge can be incorporated systematically")
            print("  • Sequential updating enables online learning")
            print("  • Business decisions benefit from probabilistic reasoning")
            print("  • The MCP tool makes Bayesian analysis accessible to AI systems")
            
            print(f"\n📁 Generated files:")
            print("  • demos/ab_test_results.png - A/B test visualization")
            print("  • Check server logs for detailed analysis")
            
            return True
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            return False
            
        finally:
            self.stop_server()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Bayesian MCP Tool Demo")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="URL of the Bayesian MCP server"
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=["all", "ab", "medical", "financial", "ml", "sequential"],
        default="all",
        help="Which demo to run"
    )
    
    args = parser.parse_args()
    
    demo = BayesianMCPDemo(args.url)
    
    if args.demo == "all":
        success = demo.run_all_demos()
    else:
        # Start server first
        if not demo.start_server():
            return 1
            
        try:
            if args.demo == "ab":
                demo.demo_ab_testing()
            elif args.demo == "medical":
                demo.demo_medical_diagnosis()
            elif args.demo == "financial":
                demo.demo_financial_risk()
            elif args.demo == "ml":
                demo.demo_ml_parameter_estimation()
            elif args.demo == "sequential":
                demo.demo_sequential_learning()
            success = True
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            success = False
        finally:
            demo.stop_server()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())