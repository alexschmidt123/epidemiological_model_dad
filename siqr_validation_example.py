#!/usr/bin/env python3
"""
Example script showing how to validate SIQR DAD designs
"""

import torch
import numpy as np
from siqr_dad_model import SIQR_DAD_Model, create_siqr_design_network
from actual_ig_validation import ActualIGValidator

class SIQRValidator(ActualIGValidator):
    """SIQR-specific validator that handles different design types"""
    
    def __init__(self, siqr_model, design_type="time", **kwargs):
        super().__init__(siqr_model, **kwargs)
        self.design_type = design_type
    
    def generate_observations(self, design, true_theta):
        """Generate observations for SIQR model"""
        observations = []
        
        for design_val in design:
            if self.design_type == "time":
                # Time-based design
                outcomes = self.seir_model.solve_siqr_sde(true_theta.unsqueeze(0), design_val)
                # Use infected count (same as SEIR for consistency)
                infected_count = outcomes['infected'].item()
            elif self.design_type == "intervention":
                # Intervention-based design
                outcomes = self.seir_model.solve_siqr_sde(true_theta.unsqueeze(0), design_val)
                infected_count = outcomes['infected'].item()
            
            # Generate observation (consistent with SEIR)
            obs = torch.poisson(torch.tensor(infected_count + 1e-6)).item()
            observations.append(obs)
            
        return observations

def validate_siqr_dad(design_type="time"):
    """Validate SIQR DAD against baseline methods"""
    
    device = torch.device("cpu")
    
    # Create SIQR model
    design_net = create_siqr_design_network(design_type=design_type).to(device)
    siqr_model = SIQR_DAD_Model(
        design_net=design_net,
        T=4,
        design_type=design_type
    )
    
    print(f"Validating SIQR DAD with {design_type} designs...")
    
    # Define appropriate baseline designs based on type
    if design_type == "time":
        design_methods = {
            'DAD_learned': lambda: extract_siqr_dad_design(design_net, siqr_model),
            'uniform_spacing': torch.tensor([0.1, 0.3, 0.5, 0.7]),
            'early_heavy': torch.tensor([0.05, 0.15, 0.25, 0.35]),
            'expert_design': torch.tensor([0.1, 0.25, 0.5, 0.8]),
        }
    elif design_type == "intervention":
        design_methods = {
            'DAD_learned': lambda: extract_siqr_dad_design(design_net, siqr_model),
            'uniform_intervention': torch.tensor([0.2, 0.5, 0.8, 1.1]),  # Intervention strengths
            'escalating': torch.tensor([0.1, 0.4, 0.7, 1.0]),
            'high_early': torch.tensor([0.8, 1.0, 0.6, 0.4]),  # Strong early intervention
        }
    
    # Run validation
    validator = SIQRValidator(
        siqr_model, 
        design_type=design_type,
        n_validation_scenarios=20,
        n_posterior_samples=200
    )
    
    results, scenarios = validator.validate_design_methods(design_methods)
    stats = validator.compute_statistics(results)
    comparisons = validator.statistical_comparison(results)
    
    # Print results
    print_siqr_results(stats, comparisons, design_type)
    
    return results, stats, comparisons

def extract_siqr_dad_design(design_net, siqr_model):
    """Extract design from trained SIQR DAD network"""
    design_net.eval()
    
    with torch.no_grad():
        designs = []
        observations = []
        
        for t in range(4):
            if t == 0:
                raw_design = design_net.empty_value
            else:
                # Encode history and emit next design
                encodings = []
                for xi, y in zip(designs, observations):
                    encoding = design_net.encoder(
                        torch.tensor([xi]), 
                        torch.tensor([y])
                    )
                    encodings.append(encoding)
                
                if encodings:
                    total_encoding = sum(encodings)
                    raw_design = design_net.emitter(total_encoding)
                else:
                    raw_design = design_net.empty_value
            
            # Apply appropriate transformation based on design type
            if siqr_model.design_type == "time":
                design_val = torch.nn.functional.softplus(raw_design).clamp(min=1.0, max=50.0)
            elif siqr_model.design_type == "intervention":
                design_val = 2.0 * torch.nn.functional.sigmoid(raw_design)  # [0, 2]
            
            design_val = design_val.item() if hasattr(design_val, 'item') else float(design_val)
            designs.append(design_val)
            observations.append(2.0)  # Dummy observation
    
    return torch.tensor(designs)

def print_siqr_results(stats, comparisons, design_type):
    """Print SIQR-specific results"""
    
    print(f"\n{'='*60}")
    print(f"SIQR DAD VALIDATION RESULTS ({design_type.upper()} DESIGNS)")
    print("="*60)
    
    sorted_methods = sorted(stats.items(), key=lambda x: x[1]['mean_ig'], reverse=True)
    
    print(f"\n{'Method':<15} {'Mean IG':<10} {'Std IG':<10} {'Success %':<10}")
    print("-" * 50)
    
    for method, stat in sorted_methods:
        print(f"{method:<15} {stat['mean_ig']:<10.4f} {stat['std_ig']:<10.4f} {stat['success_rate']:<10.1%}")
    
    best_method = sorted_methods[0][0]
    print(f"\nBest performer: {best_method}")

if __name__ == "__main__":
    import sys
    
    design_type = "time"
    if len(sys.argv) > 1:
        design_type = sys.argv[1]  # "time" or "intervention"
    
    validate_siqr_dad(design_type)
