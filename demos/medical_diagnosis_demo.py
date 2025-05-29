#!/usr/bin/env python3
"""
Medical Diagnosis Demo for Bayesian MCP Tool

This demo shows how to use Bayesian inference for medical test interpretation,
demonstrating how prior probabilities affect posterior diagnosis probabilities.
"""

import requests
import numpy as np
import json

class MedicalDiagnosisDemo:
    def __init__(self, base_url="http://localhost:8002"):
        self.base_url = base_url
        
    def run_demo(self):
        """Run the complete medical diagnosis demonstration."""
        print("🏥 Bayesian Medical Diagnosis Demo")
        print("="*60)
        
        print("📋 Scenario: COVID-19 Rapid Test Interpretation")
        print("Demonstrating how Bayesian inference helps interpret medical tests")
        print("by accounting for disease prevalence and test characteristics.")
        
        # Test characteristics (realistic values for COVID-19 rapid tests)
        sensitivity = 0.85  # True positive rate - probability test is positive given disease
        specificity = 0.97  # True negative rate - probability test is negative given no disease
        
        print(f"\n🧪 Test Characteristics:")
        print(f"  Sensitivity: {sensitivity:.0%} (detects {sensitivity:.0%} of actual cases)")
        print(f"  Specificity: {specificity:.0%} (correctly identifies {specificity:.0%} of non-cases)")
        print(f"  False positive rate: {1-specificity:.0%}")
        print(f"  False negative rate: {1-sensitivity:.0%}")
        
        # Different clinical scenarios with varying prior probabilities
        scenarios = [
            {
                "name": "General population screening",
                "prior": 0.02,
                "description": "Low-prevalence community screening"
            },
            {
                "name": "High-prevalence area",
                "prior": 0.15,
                "description": "Testing in outbreak area"
            },
            {
                "name": "Symptomatic patient",
                "prior": 0.30,
                "description": "Patient with COVID-like symptoms"
            },
            {
                "name": "Close contact exposure", 
                "prior": 0.50,
                "description": "Known exposure to confirmed case"
            },
            {
                "name": "Healthcare worker",
                "prior": 0.25,
                "description": "Symptomatic healthcare worker"
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            print(f"\n" + "="*50)
            print(f"📋 Scenario: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"Prior probability of COVID-19: {scenario['prior']:.1%}")
            
            # Test both positive and negative results
            for test_result in ["positive", "negative"]:
                print(f"\n🧪 Test Result: {test_result.upper()}")
                
                # Manual Bayes' theorem calculation for comparison
                prior_prob = scenario['prior']
                
                if test_result == "positive":
                    # P(Disease|+) = P(+|Disease) * P(Disease) / P(+)
                    # P(+) = P(+|Disease) * P(Disease) + P(+|No Disease) * P(No Disease)
                    likelihood_pos = sensitivity * prior_prob + (1 - specificity) * (1 - prior_prob)
                    posterior_manual = (sensitivity * prior_prob) / likelihood_pos
                else:
                    # P(Disease|-) = P(-|Disease) * P(Disease) / P(-)
                    # P(-) = P(-|Disease) * P(Disease) + P(-|No Disease) * P(No Disease)
                    likelihood_neg = (1 - sensitivity) * prior_prob + specificity * (1 - prior_prob)
                    posterior_manual = ((1 - sensitivity) * prior_prob) / likelihood_neg
                
                print(f"  Manual Bayes calculation: {posterior_manual:.1%}")
                
                # Now create Bayesian model to verify
                model_name = f"covid_diagnosis_{len(results)}"
                
                variables = {
                    "has_disease": {
                        "distribution": "bernoulli",
                        "params": {"p": prior_prob}
                    },
                    "test_result": {
                        "distribution": "bernoulli",
                        "params": {"p": f"has_disease * {sensitivity} + (1 - has_disease) * {1 - specificity}"},
                        "observed": 1 if test_result == "positive" else 0
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
                
                if create_response.json().get("success"):
                    # Update beliefs
                    update_response = requests.post(f"{self.base_url}/mcp", json={
                        "function_name": "update_beliefs",
                        "parameters": {
                            "model_name": model_name,
                            "evidence": {},
                            "sample_kwargs": {
                                "draws": 2000,
                                "tune": 1000,
                                "chains": 2
                            }
                        }
                    })
                    
                    if update_response.json().get("success"):
                        posterior = update_response.json()["posterior"]
                        if "has_disease" in posterior:
                            bayesian_prob = posterior["has_disease"]["mean"]
                            uncertainty = posterior["has_disease"]["std"]
                            ci_lower = posterior["has_disease"]["q025"]
                            ci_upper = posterior["has_disease"]["q975"]
                            
                            print(f"  Bayesian MCP result: {bayesian_prob:.1%} ± {uncertainty:.1%}")
                            print(f"  95% Credible interval: [{ci_lower:.1%}, {ci_upper:.1%}]")
                            
                            # Verify manual vs Bayesian calculation match
                            diff = abs(posterior_manual - bayesian_prob)
                            if diff < 0.01:
                                print(f"  ✅ Manual and Bayesian calculations match!")
                            else:
                                print(f"  ⚠️  Difference: {diff:.3f}")
                
                # Clinical interpretation
                prob = posterior_manual
                if test_result == "positive":
                    if prob > 0.90:
                        interpretation = "Very high probability - confirmatory test recommended"
                        action = "Isolate and treat presumptively"
                    elif prob > 0.70:
                        interpretation = "High probability - likely positive"
                        action = "Isolate and consider treatment"
                    elif prob > 0.30:
                        interpretation = "Moderate probability - consider retesting"
                        action = "Quarantine and retest with PCR"
                    else:
                        interpretation = "Low probability despite positive test"
                        action = "Retest with PCR - likely false positive"
                else:
                    if prob < 0.05:
                        interpretation = "Very low probability - likely negative"
                        action = "No isolation needed"
                    elif prob < 0.15:
                        interpretation = "Low probability - probably negative"
                        action = "Monitor symptoms"
                    else:
                        interpretation = "Moderate probability despite negative test"
                        action = "Consider PCR test if symptomatic"
                
                print(f"  📝 Clinical interpretation: {interpretation}")
                print(f"  🎯 Recommended action: {action}")
                
                results.append({
                    "scenario": scenario["name"],
                    "prior": prior_prob,
                    "test_result": test_result,
                    "posterior": prob,
                    "interpretation": interpretation,
                    "action": action
                })
        
        # Summary analysis
        print(f"\n" + "="*60)
        print("📊 Summary Analysis")
        print("="*60)
        
        print("\n💡 Key Insights from Bayesian Diagnosis:")
        
        # Find interesting comparisons
        low_prev_pos = [r for r in results if r["prior"] == 0.02 and r["test_result"] == "positive"][0]
        high_prev_pos = [r for r in results if r["prior"] == 0.50 and r["test_result"] == "positive"][0]
        
        print(f"\n🔍 Positive Test Results:")
        print(f"  Low prevalence ({low_prev_pos['prior']:.1%}): {low_prev_pos['posterior']:.1%} probability")
        print(f"  High prevalence ({high_prev_pos['prior']:.1%}): {high_prev_pos['posterior']:.1%} probability")
        print(f"  Same test, different contexts = different meanings!")
        
        # False positive analysis
        low_prev_false_pos_rate = (1 - low_prev_pos['posterior'])
        high_prev_false_pos_rate = (1 - high_prev_pos['posterior'])
        
        print(f"\n⚠️  False Positive Analysis:")
        print(f"  In low prevalence setting: {low_prev_false_pos_rate:.1%} of positive tests are false")
        print(f"  In high prevalence setting: {high_prev_false_pos_rate:.1%} of positive tests are false")
        
        # Practical implications
        print(f"\n🏥 Practical Clinical Implications:")
        print("  • Test performance depends critically on disease prevalence")
        print("  • Same test result means different things in different contexts")
        print("  • Positive tests in low-prevalence populations need confirmation")
        print("  • Negative tests are generally more reliable due to high specificity")
        print("  • Clinical judgment must incorporate pretest probability")
        
        # Population screening analysis
        population_size = 100000
        low_prev = 0.02
        
        true_cases = int(population_size * low_prev)
        true_negatives = population_size - true_cases
        
        detected_cases = int(true_cases * sensitivity)
        false_negatives = true_cases - detected_cases
        
        false_positives = int(true_negatives * (1 - specificity))
        true_negative_tests = true_negatives - false_positives
        
        total_positive_tests = detected_cases + false_positives
        ppv = detected_cases / total_positive_tests if total_positive_tests > 0 else 0
        
        print(f"\n📈 Population Screening Example (N={population_size:,}, prevalence={low_prev:.1%}):")
        print(f"  True cases in population: {true_cases:,}")
        print(f"  Total positive tests: {total_positive_tests:,}")
        print(f"  True positives: {detected_cases:,}")
        print(f"  False positives: {false_positives:,}")
        print(f"  Positive predictive value: {ppv:.1%}")
        print(f"  → {(1-ppv):.1%} of positive tests would be false positives!")
        
        print(f"\n🎓 Educational Takeaways:")
        print("  • Bayes' theorem quantifies how prior knowledge affects diagnosis")
        print("  • Test interpretation requires understanding disease prevalence")
        print("  • Bayesian reasoning prevents misinterpretation of test results")
        print("  • AI systems should incorporate epidemiological context")
        print("  • Uncertainty quantification is crucial for medical decisions")
        
        print(f"\n🎉 Medical diagnosis demo completed!")
        
        return results

def main():
    """Run the medical diagnosis demo."""
    demo = MedicalDiagnosisDemo()
    
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