#!/usr/bin/env python3
"""
Financial Risk Assessment Demo for Bayesian MCP Tool

This demo shows how to use Bayesian inference for portfolio Value at Risk (VaR)
estimation with proper uncertainty quantification.
"""

import requests
import numpy as np
import json

class FinancialRiskDemo:
    def __init__(self, base_url="http://localhost:8002"):
        self.base_url = base_url
        
    def run_demo(self):
        """Run the complete financial risk assessment demonstration."""
        print("💰 Bayesian Financial Risk Assessment Demo")
        print("="*60)
        
        print("📊 Scenario: Portfolio Value at Risk (VaR) Estimation")
        print("Demonstrating Bayesian estimation of portfolio risk with")
        print("uncertainty quantification for regulatory capital requirements.")
        
        # Portfolio composition
        assets = ["Tech Stocks", "Government Bonds", "Corporate Bonds", "REITs"]
        weights = [0.40, 0.30, 0.20, 0.10]
        
        print(f"\n📈 Portfolio Composition:")
        for asset, weight in zip(assets, weights):
            print(f"  {asset}: {weight:.0%}")
        
        # Generate synthetic portfolio returns (252 trading days)
        np.random.seed(123)
        n_days = 252
        
        # True parameters (unknown in real scenario)
        true_daily_return = 0.0008  # 0.08% daily return
        true_volatility = 0.018     # 1.8% daily volatility
        
        # Generate returns with some fat tails (student-t distribution)
        from scipy import stats
        portfolio_returns = stats.t.rvs(df=5, loc=true_daily_return, 
                                      scale=true_volatility, size=n_days)
        
        # Calculate portfolio statistics
        annual_return = np.mean(portfolio_returns) * 252
        annual_volatility = np.std(portfolio_returns) * np.sqrt(252)
        max_drawdown = np.min(np.cumsum(portfolio_returns))
        
        print(f"\n📊 Historical Performance (252 trading days):")
        print(f"  Annualized return: {annual_return:.1%}")
        print(f"  Annualized volatility: {annual_volatility:.1%}")
        print(f"  Maximum drawdown: {max_drawdown:.1%}")
        print(f"  Sharpe ratio: {annual_return/annual_volatility:.2f}")
        
        # Classical VaR calculation (for comparison)
        confidence_levels = [0.95, 0.99]
        
        print(f"\n📊 Classical VaR Estimates:")
        for conf in confidence_levels:
            classical_var = -np.percentile(portfolio_returns, (1-conf)*100)
            print(f"  {conf:.0%} VaR (classical): {classical_var:.2%}")
        
        print("  ⚠️  Classical VaR doesn't account for parameter uncertainty!")
        
        # Bayesian VaR estimation
        print(f"\n🧠 Bayesian VaR Estimation:")
        print("Creating Bayesian model for portfolio returns...")
        
        model_name = "portfolio_var"
        
        # Create Bayesian model for returns
        variables = {
            "mu": {
                "distribution": "normal",
                "params": {"mu": 0.0, "sigma": 0.005}  # Prior on daily return
            },
            "sigma": {
                "distribution": "halfnormal", 
                "params": {"sigma": 0.03}  # Prior on volatility
            },
            "returns": {
                "distribution": "normal",
                "params": {"mu": "mu", "sigma": "sigma"},
                "observed": portfolio_returns.tolist()
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
                    "chains": 2
                }
            }
        })
        
        if not update_response.json().get("success"):
            print(f"❌ Belief update failed: {update_response.json()}")
            return
            
        posterior = update_response.json()["posterior"]
        print("✅ MCMC sampling completed")
        
        # Extract posterior parameters
        mu_samples = posterior["mu"]["samples"]
        sigma_samples = posterior["sigma"]["samples"]
        
        print(f"\n📊 Posterior Parameter Estimates:")
        print(f"  Daily return (μ): {posterior['mu']['mean']:.4f} ± {posterior['mu']['std']:.4f}")
        print(f"  Daily volatility (σ): {posterior['sigma']['mean']:.4f} ± {posterior['sigma']['std']:.4f}")
        
        annualized_return_post = posterior['mu']['mean'] * 252
        annualized_vol_post = posterior['sigma']['mean'] * np.sqrt(252)
        
        print(f"  Annualized return: {annualized_return_post:.1%}")
        print(f"  Annualized volatility: {annualized_vol_post:.1%}")
        
        # Bayesian VaR calculation
        print(f"\n📊 Bayesian VaR Estimates:")
        
        portfolio_value = 10000000  # $10M portfolio
        
        for conf_level in confidence_levels:
            var_samples = []
            
            # For each posterior sample, calculate VaR
            for mu, sigma in zip(mu_samples[:1000], sigma_samples[:1000]):
                # Generate return distribution from posterior parameters
                sim_returns = np.random.normal(mu, sigma, 1000)
                var_percentile = np.percentile(sim_returns, (1-conf_level)*100)
                var_dollar = -var_percentile * portfolio_value
                var_samples.append(var_dollar)
            
            var_mean = np.mean(var_samples)
            var_std = np.std(var_samples)
            var_ci_lower = np.percentile(var_samples, 2.5)
            var_ci_upper = np.percentile(var_samples, 97.5)
            
            print(f"\n  {conf_level:.0%} VaR Analysis:")
            print(f"    Mean VaR: ${var_mean:,.0f}")
            print(f"    VaR uncertainty: ± ${var_std:,.0f}")
            print(f"    95% CI: [${var_ci_lower:,.0f}, ${var_ci_upper:,.0f}]")
            print(f"    As % of portfolio: {var_mean/portfolio_value:.2%}")
            
            # Risk interpretation
            if conf_level == 0.95:
                print(f"    Interpretation: 95% confidence that daily loss won't exceed ${var_mean:,.0f}")
            else:
                print(f"    Interpretation: 99% confidence that daily loss won't exceed ${var_mean:,.0f}")
        
        # Expected Shortfall (Conditional VaR)
        print(f"\n📊 Expected Shortfall Analysis:")
        
        for conf_level in confidence_levels:
            es_samples = []
            
            for mu, sigma in zip(mu_samples[:1000], sigma_samples[:1000]):
                sim_returns = np.random.normal(mu, sigma, 10000)
                var_threshold = np.percentile(sim_returns, (1-conf_level)*100)
                tail_losses = sim_returns[sim_returns <= var_threshold]
                if len(tail_losses) > 0:
                    expected_shortfall = -np.mean(tail_losses) * portfolio_value
                    es_samples.append(expected_shortfall)
            
            if es_samples:
                es_mean = np.mean(es_samples)
                es_std = np.std(es_samples)
                
                print(f"  {conf_level:.0%} Expected Shortfall: ${es_mean:,.0f} ± ${es_std:,.0f}")
                print(f"    (Average loss given VaR is exceeded)")
        
        # Regulatory capital calculation
        print(f"\n🏛️  Regulatory Capital Requirements:")
        
        # Basel III market risk requirements
        var_99 = np.mean([var_samples[i] for i, (mu, sigma) in enumerate(zip(mu_samples[:1000], sigma_samples[:1000])) 
                         if i < len(var_samples)])
        
        # Stressed VaR (assume 50% increase in volatility)
        stressed_var_samples = []
        for mu, sigma in zip(mu_samples[:500], sigma_samples[:500]):
            stressed_sigma = sigma * 1.5
            sim_returns = np.random.normal(mu, stressed_sigma, 1000)
            var_percentile = np.percentile(sim_returns, 1)  # 99% VaR
            stressed_var = -var_percentile * portfolio_value
            stressed_var_samples.append(stressed_var)
        
        stressed_var_mean = np.mean(stressed_var_samples)
        
        # Market risk capital (simplified Basel III calculation)
        market_risk_capital = max(var_99, stressed_var_mean) * 3  # 3x multiplier
        
        print(f"  99% VaR: ${var_99:,.0f}")
        print(f"  Stressed VaR: ${stressed_var_mean:,.0f}")
        print(f"  Market Risk Capital: ${market_risk_capital:,.0f}")
        print(f"  Capital as % of portfolio: {market_risk_capital/portfolio_value:.1%}")
        
        # Risk management insights
        print(f"\n🎯 Risk Management Insights:")
        
        # Tail risk analysis
        tail_risk_prob = len([r for r in portfolio_returns if r < -0.05]) / len(portfolio_returns)
        print(f"  Historical probability of >5% daily loss: {tail_risk_prob:.1%}")
        
        # Drawdown analysis
        cumulative_returns = np.cumsum(portfolio_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak)
        max_dd_duration = 0
        current_dd_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        print(f"  Maximum drawdown duration: {max_dd_duration} days")
        print(f"  Current drawdown: {drawdown[-1]:.2%}")
        
        # Portfolio recommendations
        print(f"\n💡 Portfolio Optimization Recommendations:")
        
        current_sharpe = annualized_return_post / annualized_vol_post
        target_vol = 0.12  # 12% target volatility
        
        if annualized_vol_post > target_vol:
            vol_reduction_needed = (annualized_vol_post - target_vol) / annualized_vol_post
            print(f"  Current volatility ({annualized_vol_post:.1%}) exceeds target ({target_vol:.1%})")
            print(f"  Consider reducing risk by {vol_reduction_needed:.1%}")
            print(f"  Suggested actions:")
            print(f"    • Increase bond allocation")
            print(f"    • Add defensive assets")
            print(f"    • Implement volatility targeting")
        
        # Stress testing scenarios
        print(f"\n🚨 Stress Testing Scenarios:")
        
        stress_scenarios = [
            {"name": "Market Crash", "return_shock": -0.30, "vol_shock": 2.0},
            {"name": "Interest Rate Spike", "return_shock": -0.15, "vol_shock": 1.5},
            {"name": "Credit Crisis", "return_shock": -0.25, "vol_shock": 1.8}
        ]
        
        for scenario in stress_scenarios:
            stressed_annual_return = annualized_return_post + scenario["return_shock"]
            stressed_annual_vol = annualized_vol_post * scenario["vol_shock"]
            stressed_daily_vol = stressed_annual_vol / np.sqrt(252)
            
            # Estimate portfolio loss under stress
            stress_var_95 = 1.645 * stressed_daily_vol * portfolio_value  # 95% VaR
            
            print(f"  {scenario['name']}:")
            print(f"    Expected annual return: {stressed_annual_return:.1%}")
            print(f"    Expected volatility: {stressed_annual_vol:.1%}")
            print(f"    Estimated 95% daily VaR: ${stress_var_95:,.0f}")
        
        print(f"\n✨ Advantages of Bayesian Risk Management:")
        print("  • Quantifies uncertainty in risk estimates")
        print("  • Incorporates parameter uncertainty in VaR")
        print("  • Provides confidence intervals for risk metrics")
        print("  • Enables probabilistic stress testing")
        print("  • Supports dynamic risk budgeting")
        print("  • Improves regulatory capital allocation")
        
        print(f"\n🎉 Financial risk assessment demo completed!")
        
        return {
            "var_estimates": var_samples if 'var_samples' in locals() else [],
            "portfolio_value": portfolio_value,
            "posterior": posterior
        }

def main():
    """Run the financial risk assessment demo."""
    demo = FinancialRiskDemo()
    
    try:
        # Check if server is running
        response = requests.get(f"{demo.base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not running. Start it with: python bayesian_mcp.py --port 8002")
            return 1
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Start it with: python bayesian_mcp.py --port 8002")
        return 1
    
    demo.run_demo()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())