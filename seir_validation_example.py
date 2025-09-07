#!/usr/bin/env python3
"""
Example script showing how to validate SEIR DAD designs
Run this after training your SEIR model
"""

import torch
import numpy as np
from seir_dad_model import SEIR_DAD_Model, create_seir_design_network
from actual_ig_validation import ActualIGValidator, run_validation_study

def validate_seir_dad():
    """Validate SEIR DAD against baseline methods"""
    
    # 1. Load your trained SEIR model
    device = torch.device("cpu")
    
    # Create design network (replace with your trained version)
    design_net = create_seir_design_network().to(device)
    # If you have a saved model: design_net.load_state_dict(torch.load('seir_dad_model.pth'))
    
    # Create SEIR model
    seir_model = SEIR_DAD_Model(
        design_net=design_net,
        T=4,
        population_size=500.0,
        theta_prior_loc=torch.tensor([0.8, 0.3, 0.2]).log(),  # Much higher rates
        theta_prior_scale=torch.eye(3) * 0.3 ** 2  # Same as robust test
    )

    # And update the baseline designs to use longer time horizons:
    design_methods = {
        'DAD_learned': lambda: extract_dad_design(design_net, seir_model),
        'uniform_spacing': torch.tensor([5.0, 15.0, 25.0, 35.0]),  # Longer times
        'early_heavy': torch.tensor([2.0, 6.0, 10.0, 14.0]),
        'expert_design': torch.tensor([5.0, 12.0, 24.0, 40.0]),
        'random_design': lambda: torch.rand(4) * 30 + 5,
    }
    
    # 3. Run validation study
    validator = ActualIGValidator(
        seir_model, 
        n_validation_scenarios=20,  # Start small for testing
        n_posterior_samples=200
    )
    
    print("Running validation (this may take several minutes)...")
    
    results, scenarios = validator.validate_design_methods(design_methods)
    stats = validator.compute_statistics(results)
    comparisons = validator.statistical_comparison(results)
    
    # 4. Print results
    print_results(stats, comparisons)
    
    return results, stats, comparisons

def extract_dad_design(design_net, seir_model):
    """Extract design from trained DAD network"""
    design_net.eval()
    
    with torch.no_grad():
        # Simulate the sequential design process
        designs = []
        observations = []
        
        for t in range(4):
            if t == 0:
                # First design uses empty_value
                raw_design = design_net.empty_value
            else:
                # Subsequent designs use encoder + emitter
                # Encode previous (design, observation) pairs
                encodings = []
                for xi, y in zip(designs, observations):
                    encoding = design_net.encoder(
                        torch.tensor([xi]), 
                        torch.tensor([y])
                    )
                    encodings.append(encoding)
                
                # Sum encodings and emit next design
                if encodings:
                    total_encoding = sum(encodings)
                    raw_design = design_net.emitter(total_encoding)
                else:
                    raw_design = design_net.empty_value
            
            # Apply softplus transformation (same as in model)
            design_time = torch.nn.functional.softplus(raw_design).clamp(min=1.0, max=100.0)
            design_val = design_time.item() if hasattr(design_time, 'item') else float(design_time)
            
            designs.append(design_val)
            observations.append(2.0)  # Dummy observation for next step
    
    return torch.tensor(designs)

def print_results(stats, comparisons):
    """Print validation results in readable format"""
    
    print("\n" + "="*60)
    print("SEIR DAD VALIDATION RESULTS")
    print("="*60)
    
    # Sort methods by mean IG
    sorted_methods = sorted(stats.items(), key=lambda x: x[1]['mean_ig'], reverse=True)
    
    print(f"\n{'Method':<15} {'Mean IG':<10} {'Std IG':<10} {'Success %':<10}")
    print("-" * 50)
    
    for method, stat in sorted_methods:
        print(f"{method:<15} {stat['mean_ig']:<10.4f} {stat['std_ig']:<10.4f} {stat['success_rate']:<10.1%}")
    
    # Highlight best performer
    best_method = sorted_methods[0][0]
    print(f"\n🏆 Best performer: {best_method}")
    
    # Show significant comparisons
    print(f"\n📊 Statistical Comparisons (p < 0.05):")
    for comp_name, comp_data in comparisons.items():
        if comp_data['t_test']['p_value'] < 0.05:
            method1, method2 = comp_name.split('_vs_')
            mean_diff = comp_data['mean_diff']
            p_val = comp_data['t_test']['p_value']
            
            if mean_diff > 0:
                print(f"   {method1} > {method2}: diff={mean_diff:.4f}, p={p_val:.4f}")
            else:
                print(f"   {method2} > {method1}: diff={-mean_diff:.4f}, p={p_val:.4f}")

def quick_test():
    """Quick test with minimal computation"""
    print("Running quick validation test...")
    
    device = torch.device("cpu")
    design_net = create_seir_design_network().to(device)
    seir_model = SEIR_DAD_Model(design_net=design_net, T=4)
    
    # Test single scenario
    validator = ActualIGValidator(seir_model, n_validation_scenarios=3, n_posterior_samples=50)
    
    design_methods = {
        'uniform': torch.tensor([0.1, 0.3, 0.5, 0.7]),
        'early': torch.tensor([0.05, 0.15, 0.25, 0.35]),
    }
    
    results, _ = validator.validate_design_methods(design_methods)
    stats = validator.compute_statistics(results)
    
    print("Quick test results:")
    for method, stat in stats.items():
        print(f"{method}: Mean IG = {stat['mean_ig']:.4f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        validate_seir_dad()
