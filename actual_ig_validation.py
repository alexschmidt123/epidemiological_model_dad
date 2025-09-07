import torch
import numpy as np
from scipy.stats import entropy
import pyro
import pyro.distributions as dist
from tqdm import tqdm


class ActualIGValidator:
    """Validate design methods by measuring actual information gain"""
    
    def __init__(self, seir_model, n_validation_scenarios=100, n_posterior_samples=1000):
        self.seir_model = seir_model
        self.n_scenarios = n_validation_scenarios
        self.n_posterior_samples = n_posterior_samples
        
    def _get_prior_distribution(self):
        """Get prior distribution from model parameters"""
        import pyro.distributions as dist
        return dist.MultivariateNormal(
            self.seir_model.theta_prior_loc, 
            self.seir_model.theta_prior_scale
        )
        
    def compute_actual_information_gain(self, design, true_theta, return_details=False):
        """
        Compute actual IG by measuring posterior variance reduction
        
        IG = H(θ) - H(θ|Y,ξ) = Var_prior(θ) - Var_posterior(θ|Y,ξ)
        """
        
        # 1. Generate observations using true parameters
        observations = self.generate_observations(design, true_theta)
        
        # 2. Compute prior entropy (baseline uncertainty)
        prior_samples = self.seir_model.theta_prior.sample([self.n_posterior_samples])
        prior_entropy = self.compute_entropy(prior_samples)
        
        # 3. Compute posterior entropy (uncertainty after seeing data)
        posterior_samples = self.fit_posterior_samples(design, observations)
        posterior_entropy = self.compute_entropy(posterior_samples)
        
        # 4. Information gain = reduction in entropy
        actual_ig = prior_entropy - posterior_entropy
        
        if return_details:
            return {
                'actual_ig': actual_ig,
                'prior_entropy': prior_entropy,
                'posterior_entropy': posterior_entropy,
                'observations': observations,
                'posterior_samples': posterior_samples
            }
        
        return actual_ig
    
    def generate_observations(self, design, true_theta):
        """Generate observations using true parameters"""
        observations = []
        
        for t in design:
            # Solve SEIR SDE with true parameters
            infected_count = self.seir_model.solve_seir_sde(
                true_theta.unsqueeze(0), t
            )
            
            # Generate noisy observation
            if infected_count.dim() > 0:
                infected_count = infected_count.item()
            
            # Sample from observation model
            obs = torch.poisson(torch.tensor(infected_count + 1e-6)).item()
            observations.append(obs)
            
        return observations
    
    def fit_posterior_samples(self, design, observations):
        """Fit posterior using MCMC or variational inference"""
        
        # Simple rejection sampling approach
        n_accepted = 0
        posterior_samples = []
        max_attempts = self.n_posterior_samples * 100
        
        for _ in range(max_attempts):
            if n_accepted >= self.n_posterior_samples:
                break
                
            # Sample from prior
            theta_candidate = self.seir_model.theta_prior.sample()
            
            # Compute likelihood of observations
            log_likelihood = 0
            try:
                for i, (t, obs) in enumerate(zip(design, observations)):
                    predicted_infected = self.seir_model.solve_seir_sde(
                        theta_candidate.unsqueeze(0), t
                    )
                    if predicted_infected.dim() > 0:
                        predicted_infected = predicted_infected.item()
                    
                    # Ensure non-negative rate
                    predicted_infected = max(0.0, predicted_infected)
                    
                    # Poisson likelihood
                    log_likelihood += torch.poisson(
                        torch.tensor(predicted_infected + 1e-6)
                    ).log_prob(torch.tensor(float(obs))).item()
                
                # Accept/reject based on likelihood (simple thresholding)
                if log_likelihood > -10:  # Reasonable likelihood threshold
                    posterior_samples.append(theta_candidate)
                    n_accepted += 1
                    
            except Exception as e:
                # Skip this candidate if SDE integration fails
                continue
        
        if len(posterior_samples) < 100:
            print(f"Warning: Only {len(posterior_samples)} posterior samples accepted")
            
        return torch.stack(posterior_samples) if posterior_samples else torch.zeros(1, 3)
    
    def compute_entropy(self, samples):
        """Compute entropy of parameter samples"""
        if len(samples) == 0:
            return float('inf')
            
        # For continuous variables, use differential entropy approximation
        # H(θ) ≈ log(det(Cov(θ))) + const
        
        # Compute covariance manually for older PyTorch versions
        def compute_covariance(samples):
            mean = samples.mean(dim=0, keepdim=True)
            centered = samples - mean
            return torch.mm(centered.T, centered) / (samples.shape[0] - 1)
        
        cov_matrix = compute_covariance(samples)
        
        # Add small regularization for numerical stability
        cov_matrix += 1e-6 * torch.eye(cov_matrix.shape[0])
        
        try:
            # Try logdet first (newer PyTorch)
            log_det = torch.logdet(cov_matrix)
            return 0.5 * log_det.item()
        except AttributeError:
            # Fallback for older PyTorch
            det_val = torch.det(cov_matrix)
            log_det = torch.log(det_val + 1e-8)
            return 0.5 * log_det.item()
        except:
            # Final fallback: use trace of covariance
            return 0.5 * torch.trace(cov_matrix).item()
    
    def validate_design_methods(self, design_methods):
        """
        Compare multiple design methods on actual information gain
        
        Args:
            design_methods: dict of {'method_name': design_function}
        """
        
        results = {method: [] for method in design_methods.keys()}
        validation_scenarios = []
        
        print(f"Running validation on {self.n_scenarios} scenarios...")
        
        for scenario in tqdm(range(self.n_scenarios)):
            # Sample random true parameters for this scenario
            prior_dist = self._get_prior_distribution()
            true_theta = prior_dist.sample()
            validation_scenarios.append(true_theta)
            
            # Test each design method
            for method_name, design_func in design_methods.items():
                try:
                    # Get design from method
                    if callable(design_func):
                        design = design_func()
                    else:
                        design = design_func  # Fixed design
                    
                    # Compute actual IG
                    actual_ig = self.compute_actual_information_gain(design, true_theta)
                    results[method_name].append(actual_ig)
                    
                except Exception as e:
                    print(f"Error in {method_name} for scenario {scenario}: {e}")
                    results[method_name].append(0.0)  # Penalty for failed designs
        
        return results, validation_scenarios
    
    def compute_statistics(self, results):
        """Compute summary statistics for validation results"""
        
        stats = {}
        for method_name, igs in results.items():
            igs = np.array(igs)
            stats[method_name] = {
                'mean_ig': np.mean(igs),
                'std_ig': np.std(igs),
                'median_ig': np.median(igs),
                'min_ig': np.min(igs),
                'max_ig': np.max(igs),
                'success_rate': np.mean(igs > 0),  # Fraction of scenarios with positive IG
            }
        
        return stats
    
    def statistical_comparison(self, results):
        """Statistical significance testing between methods"""
        from scipy.stats import ttest_ind, mannwhitneyu
        
        method_names = list(results.keys())
        comparisons = {}
        
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                igs1 = np.array(results[method1])
                igs2 = np.array(results[method2])
                
                # T-test
                t_stat, t_pval = ttest_ind(igs1, igs2)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_pval = mannwhitneyu(igs1, igs2, alternative='two-sided')
                
                comparisons[f"{method1}_vs_{method2}"] = {
                    'mean_diff': np.mean(igs1) - np.mean(igs2),
                    't_test': {'statistic': t_stat, 'p_value': t_pval},
                    'mann_whitney': {'statistic': u_stat, 'p_value': u_pval},
                    'effect_size': (np.mean(igs1) - np.mean(igs2)) / np.sqrt((np.var(igs1) + np.var(igs2)) / 2)
                }
        
        return comparisons


def run_validation_study(seir_model, dad_design_net):
    """Complete validation study comparing design methods"""
    
    validator = ActualIGValidator(seir_model, n_validation_scenarios=50, n_posterior_samples=500)
    
    # Define design methods to compare
    design_methods = {
        'DAD_learned': lambda: get_dad_design(dad_design_net),
        'uniform_spacing': torch.tensor([0.1, 0.3, 0.5, 0.7]),
        'early_heavy': torch.tensor([0.05, 0.15, 0.25, 0.35]),
        'late_heavy': torch.tensor([0.4, 0.6, 0.8, 1.0]),
        'expert_design': torch.tensor([0.1, 0.25, 0.5, 0.8]),  # Early/peak/decay
        'random_design': lambda: torch.rand(4).sort()[0],
    }
    
    # Run validation
    results, scenarios = validator.validate_design_methods(design_methods)
    
    # Compute statistics
    stats = validator.compute_statistics(results)
    comparisons = validator.statistical_comparison(results)
    
    # Print results
    print("\n" + "="*50)
    print("ACTUAL INFORMATION GAIN VALIDATION RESULTS")
    print("="*50)
    
    print("\nMethod Performance:")
    for method, stat in stats.items():
        print(f"\n{method}:")
        print(f"  Mean IG: {stat['mean_ig']:.4f} ± {stat['std_ig']:.4f}")
        print(f"  Median IG: {stat['median_ig']:.4f}")
        print(f"  Success Rate: {stat['success_rate']:.1%}")
    
    print("\n" + "-"*30)
    print("Statistical Comparisons:")
    for comp_name, comp_data in comparisons.items():
        print(f"\n{comp_name}:")
        print(f"  Mean Difference: {comp_data['mean_diff']:.4f}")
        print(f"  T-test p-value: {comp_data['t_test']['p_value']:.4f}")
        print(f"  Effect Size: {comp_data['effect_size']:.2f}")
    
    return results, stats, comparisons


def get_dad_design(design_net):
    """Extract design from trained DAD network"""
    design_net.eval()
    with torch.no_grad():
        # Simulate DAD design process
        designs = []
        observations = []
        
        for t in range(4):  # Assuming 4 experiments
            if t == 0:
                # First design
                design = design_net.empty_value
            else:
                # Subsequent designs based on history
                # This is simplified - in practice you'd use the full lazy mechanism
                encoding = sum(design_net.encoder(xi, y) for xi, y in zip(designs, observations))
                design = design_net.emitter(encoding)
            
            designs.append(design.item() if hasattr(design, 'item') else float(design))
            observations.append(2.0)  # Dummy observation for next iteration
    
    return torch.tensor(designs[:4])  # Return first 4 designs


# Example usage:
if __name__ == "__main__":
    # Assuming you have trained SEIR model and design network
    # seir_model = your_trained_seir_model
    # dad_design_net = your_trained_design_network
    
    # results, stats, comparisons = run_validation_study(seir_model, dad_design_net)
    pass