#!/usr/bin/env python3
"""
A/B Testing Demo for Bayesian MCP Tool

This demo shows how to use Bayesian analysis for A/B testing with proper
uncertainty quantification and business decision making.
"""

import requests
import numpy as np
import json
import time

class ABTestingDemo:
    def __init__(self, base_url="http://localhost:8002"):
        self.base_url = base_url
        
    def run_demo(self):
        """Run the complete A/B testing demonstration."""
        print("🧪 Bayesian A/B Testing Demo")
        print("="*60)
        
        # Scenario setup
        print("📊 Scenario: E-commerce Checkout Optimization")
        print("Testing two checkout processes:")
        print("  • Variant A: Current 3-step checkout (control)")
        print("  • Variant B: New 1-click checkout (treatment)")
        print("  • Metric: Conversion rate")
        
        # Generate realistic test data
        np.random.seed(42)
        
        # True conversion rates (unknown in real scenario)
        true_rate_a = 0.085  # 8.5%
        true_rate_b = 0.112  # 11.2% (31% relative improvement)
        
        # Sample sizes
        n_a = 2847  # Users in variant A
        n_b = 2791  # Users in variant B
        
        # Generate conversions
        conversions_a = np.random.binomial(1, true_rate_a, n_a)
        conversions_b = np.random.binomial(1, true_rate_b, n_b)
        
        successes_a = np.sum(conversions_a)
        successes_b = np.sum(conversions_b)
        
        observed_rate_a = successes_a / n_a
        observed_rate_b = successes_b / n_b
        
        print(f"\n📈 Observed Results:")
        print(f"  Variant A: {successes_a:,}/{n_a:,} conversions ({observed_rate_a:.3%})")
        print(f"  Variant B: {successes_b:,}/{n_b:,} conversions ({observed_rate_b:.3%})")
        print(f"  Observed lift: {(observed_rate_b - observed_rate_a)/observed_rate_a:.1%}")
        
        # Traditional statistical test (for comparison)
        from scipy import stats
        _, p_value = stats.chi2_contingency([
            [successes_a, n_a - successes_a],
            [successes_b, n_b - successes_b]
        ])[:2]
        
        print(f"\n📊 Traditional Chi-square Test:")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  Result: Statistically significant ✓")
        else:
            print("  Result: Not statistically significant ✗")
        print("  ⚠️  But this doesn't tell us the probability B is better!")
        
        # Bayesian analysis
        print(f"\n🧠 Bayesian Analysis:")
        print("Creating Bayesian A/B test model...")
        
        # Create Bayesian model
        model_name = "ab_test_checkout"
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
                "params": {"n": int(n_a), "p": "rate_a"},
                "observed": int(successes_a)
            },
            "likelihood_b": {
                "distribution": "binomial", 
                "params": {"n": int(n_b), "p": "rate_b"},
                "observed": int(successes_b)
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
        create_response = requests.post(f"{self.base_url}/mcp", json={
            "function_name": "create_model",
            "parameters": {
                "model_name": model_name,
                "variables": variables
            }
        })
        
        if not create_response.json().get("success"):
            print(f"❌ Model creation failed: {create_response.json()}")
            return
            
        print("✅ Model created successfully")
        
        # Update beliefs with MCMC
        print("⚙️  Running MCMC sampling...")
        
        update_response = requests.post(f"{self.base_url}/mcp", json={
            "function_name": "update_beliefs",
            "parameters": {
                "model_name": model_name,
                "evidence": {},
                "sample_kwargs": {
                    "draws": 3000,
                    "tune": 1500,
                    "chains": 2,
                    "cores": 1
                }
            }
        })
        
        if not update_response.json().get("success"):
            print(f"❌ Belief update failed: {update_response.json()}")
            return
            
        posterior = update_response.json()["posterior"]
        print("✅ MCMC sampling completed")
        
        # Analyze results
        print(f"\n📊 Posterior Results:")
        
        rate_a_stats = posterior["rate_a"]
        rate_b_stats = posterior["rate_b"]
        
        print(f"  Conversion Rate A: {rate_a_stats['mean']:.3%} ± {rate_a_stats['std']:.3%}")
        print(f"    95% CI: [{rate_a_stats['q025']:.3%}, {rate_a_stats['q975']:.3%}]")
        
        print(f"  Conversion Rate B: {rate_b_stats['mean']:.3%} ± {rate_b_stats['std']:.3%}")
        print(f"    95% CI: [{rate_b_stats['q025']:.3%}, {rate_b_stats['q975']:.3%}]")
        
        if "difference" in posterior:
            diff_stats = posterior["difference"]
            print(f"  Absolute Difference: {diff_stats['mean']:.3%} ± {diff_stats['std']:.3%}")
            print(f"    95% CI: [{diff_stats['q025']:.3%}, {diff_stats['q975']:.3%}]")
            
        if "relative_lift" in posterior:
            lift_stats = posterior["relative_lift"]
            print(f"  Relative Lift: {lift_stats['mean']:.1%} ± {lift_stats['std']:.1%}")
            print(f"    95% CI: [{lift_stats['q025']:.1%}, {lift_stats['q975']:.1%}]")
        
        # Business decision metrics
        print(f"\n🎯 Business Decision Metrics:")
        
        if "difference" in posterior and "samples" in posterior["difference"]:
            diff_samples = posterior["difference"]["samples"]
            prob_b_better = sum(1 for s in diff_samples if s > 0) / len(diff_samples)
            
            # Expected loss if choosing wrong variant
            expected_loss_choose_a = np.mean([max(0, s) for s in diff_samples])
            expected_loss_choose_b = np.mean([max(0, -s) for s in diff_samples])
            
            print(f"  Probability B > A: {prob_b_better:.1%}")
            print(f"  Expected loss if choosing A: {expected_loss_choose_a:.4f}")
            print(f"  Expected loss if choosing B: {expected_loss_choose_b:.4f}")
            
            # Risk assessment
            prob_significant_lift = sum(1 for s in diff_samples if s > 0.01) / len(diff_samples)
            prob_material_loss = sum(1 for s in diff_samples if s < -0.005) / len(diff_samples)
            
            print(f"  Probability of >1% absolute lift: {prob_significant_lift:.1%}")
            print(f"  Probability of >0.5% loss: {prob_material_loss:.1%}")
            
            # Business recommendation
            print(f"\n💼 Business Recommendation:")
            if prob_b_better > 0.95 and expected_loss_choose_b < 0.002:
                print("  🚀 STRONG RECOMMENDATION: Deploy Variant B")
                print("     High confidence with low downside risk")
            elif prob_b_better > 0.85 and expected_loss_choose_b < 0.005:
                print("  📈 MODERATE RECOMMENDATION: Deploy Variant B")
                print("     Good evidence with acceptable risk")
            elif prob_b_better > 0.60:
                print("  ⚠️  WEAK RECOMMENDATION: Consider longer test")
                print("     Promising but needs more evidence")
            elif prob_b_better < 0.20:
                print("  📉 RECOMMENDATION: Keep Variant A")
                print("     Strong evidence against Variant B")
            else:
                print("  🔄 RECOMMENDATION: Continue testing")
                print("     Inconclusive - collect more data")
        
        # Revenue impact estimate
        print(f"\n💰 Revenue Impact Estimate:")
        
        if "relative_lift" in posterior:
            lift_samples = posterior["relative_lift"]["samples"]
            
            # Assume baseline metrics
            monthly_visitors = 50000
            baseline_conversion = rate_a_stats["mean"]
            avg_order_value = 85.50
            
            monthly_revenue_baseline = monthly_visitors * baseline_conversion * avg_order_value
            
            revenue_lifts = []
            for lift in lift_samples[:100]:  # Sample subset for calculation
                new_conversion = baseline_conversion * (1 + lift)
                new_revenue = monthly_visitors * new_conversion * avg_order_value
                revenue_lift = new_revenue - monthly_revenue_baseline
                revenue_lifts.append(revenue_lift)
            
            mean_monthly_lift = np.mean(revenue_lifts)
            std_monthly_lift = np.std(revenue_lifts)
            
            print(f"  Baseline monthly revenue: ${monthly_revenue_baseline:,.0f}")
            print(f"  Expected monthly lift: ${mean_monthly_lift:,.0f} ± ${std_monthly_lift:,.0f}")
            print(f"  Expected annual lift: ${mean_monthly_lift * 12:,.0f}")
            
            # Confidence intervals
            ci_lower = np.percentile(revenue_lifts, 2.5)
            ci_upper = np.percentile(revenue_lifts, 97.5)
            print(f"  95% CI monthly lift: [${ci_lower:,.0f}, ${ci_upper:,.0f}]")
        
        print(f"\n✨ Key Advantages of Bayesian A/B Testing:")
        print("  • Provides probability that B is better than A")
        print("  • Quantifies expected loss for each decision")
        print("  • Incorporates uncertainty in all estimates")
        print("  • Enables early stopping with confidence")
        print("  • Directly supports business decision making")
        
        print(f"\n🎉 Demo completed successfully!")
        
        return {
            "observed_rate_a": observed_rate_a,
            "observed_rate_b": observed_rate_b,
            "posterior": posterior,
            "prob_b_better": prob_b_better if "difference" in posterior else None
        }

def main():
    """Run the A/B testing demo."""
    demo = ABTestingDemo()
    
    try:
        # Check if server is running
        response = requests.get(f"{demo.base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not running. Start it with: python bayes_mcp.py --port 8002")
            return 1
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Start it with: python bayes_mcp.py --port 8002")
        return 1
    
    demo.run_demo()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())