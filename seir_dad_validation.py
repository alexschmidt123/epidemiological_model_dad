"""
Validation Script for SEIR DAD Model
Computes true information gain and compares with baselines
"""

import numpy as np
import torch
import torch.distributions as dist
from scipy.integrate import odeint
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyro
import os
import json

from seir_ode_model import SEIR_ODE, SEIR_DAD_Model
from seir_dad_networks import create_seir_dad_network
from seir_dad_training import SEIR_DAD_Trainer


class TrueIGCalculator:
    """
    Calculate true information gain for SEIR model
    
    Information Gain = H(θ) - E[H(θ|y,ξ)]
    where H is entropy
    """
    
    def __init__(
        self,
        model,
        n_monte_carlo=500,
        n_posterior_samples=200,
        device='cpu'
    ):
        """
        Initialize IG calculator
        
        Parameters:
        -----------
        model : SEIR_DAD_Model
            The SEIR model
        n_monte_carlo : int
            Number of MC samples for expectation
        n_posterior_samples : int
            Number of samples for posterior estimation
        device : str or torch.device
            Device for computations
        """
        self.model = model
        self.n_monte_carlo = n_monte_carlo
        self.n_posterior_samples = n_posterior_samples
        self.device = device
        
        # Prior entropy (analytical for multivariate normal)
        prior_cov = model.theta_prior_scale.cpu().numpy()
        self.prior_entropy = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * prior_cov))
    
    def compute_posterior_samples(self, designs, observations):
        """
        Sample from posterior using rejection sampling
        
        Parameters:
        -----------
        designs : array
            Observation times
        observations : array
            Observed infected counts
        
        Returns:
        --------
        posterior_samples : array
            Samples from posterior distribution
        """
        accepted_samples = []
        prior_mean = self.model.theta_prior_loc.cpu().numpy()
        prior_cov = self.model.theta_prior_scale.cpu().numpy()
        
        # Rejection sampling
        max_attempts = self.n_posterior_samples * 1000
        attempts = 0
        
        while len(accepted_samples) < self.n_posterior_samples and attempts < max_attempts:
            attempts += 1
            
            # Sample from prior
            theta_log = np.random.multivariate_normal(prior_mean, prior_cov)
            theta = np.exp(theta_log)
            
            # Compute likelihood
            log_likelihood = 0.0
            
            for xi, y_obs in zip(designs, observations):
                # Solve ODE
                t_eval = np.linspace(0, xi, 100)
                solution = SEIR_ODE.solve(theta, t_eval, self.model.y0, self.model.N)
                predicted = solution[-1, 2]  # Infected at observation time
                
                # Compute observation likelihood
                noise_std = self.model.noise_scale * np.sqrt(predicted + 1.0)
                log_likelihood += -0.5 * ((y_obs - predicted) / noise_std) ** 2
                log_likelihood += -0.5 * np.log(2 * np.pi * noise_std ** 2)
            
            # Accept/reject based on likelihood
            # Simple threshold-based acceptance
            if log_likelihood > np.log(np.random.uniform()) * 10:  # Scaled acceptance
                accepted_samples.append(theta_log)
        
        if len(accepted_samples) < 10:
            # Not enough samples - use prior as fallback
            print(f"Warning: Only {len(accepted_samples)} posterior samples accepted")
            return np.random.multivariate_normal(prior_mean, prior_cov, self.n_posterior_samples)
        
        return np.array(accepted_samples)
    
    def estimate_posterior_entropy(self, posterior_samples):
        """
        Estimate entropy of posterior distribution
        
        Parameters:
        -----------
        posterior_samples : array
            Samples from posterior
        
        Returns:
        --------
        entropy : float
            Estimated posterior entropy
        """
        if len(posterior_samples) < 2:
            return self.prior_entropy  # Fallback to prior
        
        # Estimate covariance from samples
        posterior_cov = np.cov(posterior_samples.T)
        
        # Add regularization for numerical stability
        posterior_cov += 1e-6 * np.eye(posterior_cov.shape[0])
        
        # Compute entropy (assuming Gaussian approximation)
        try:
            entropy = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * posterior_cov))
        except:
            entropy = self.prior_entropy  # Fallback
        
        return entropy
    
    def compute_information_gain(self, designs):
        """
        Compute true information gain for given designs
        
        Parameters:
        -----------
        designs : array or list
            Observation times
        
        Returns:
        --------
        ig : float
            Information gain
        ig_std : float
            Standard deviation of IG estimate
        """
        posterior_entropies = []
        
        print(f"Computing IG with {self.n_monte_carlo} MC samples...")
        
        for _ in tqdm(range(self.n_monte_carlo)):
            # Sample true parameters
            theta_log = self.model.theta_prior.sample()
            theta = theta_log.exp().cpu().numpy()
            
            # Generate observations
            observations = []
            for xi in designs:
                # Solve ODE
                t_eval = np.linspace(0, xi, 100)
                solution = SEIR_ODE.solve(theta, t_eval, self.model.y0, self.model.N)
                true_infected = solution[-1, 2]
                
                # Add noise
                noise_std = self.model.noise_scale * np.sqrt(true_infected + 1.0)
                y_obs = true_infected + np.random.normal(0, noise_std)
                observations.append(max(0, y_obs))  # Ensure non-negative
            
            # Estimate posterior entropy
            posterior_samples = self.compute_posterior_samples(designs, observations)
            posterior_entropy = self.estimate_posterior_entropy(posterior_samples)
            posterior_entropies.append(posterior_entropy)
        
        # Compute information gain
        expected_posterior_entropy = np.mean(posterior_entropies)
        ig = self.prior_entropy - expected_posterior_entropy
        ig_std = np.std([self.prior_entropy - pe for pe in posterior_entropies])
        
        return max(0, ig), ig_std


class SEIR_Validator:
    """Complete validation suite for SEIR DAD"""
    
    def __init__(
        self,
        model,
        baseline_strategies=None,
        n_monte_carlo=200,
        device='cpu'
    ):
        """
        Initialize validator
        
        Parameters:
        -----------
        model : SEIR_DAD_Model
            Trained DAD model
        baseline_strategies : dict
            Dictionary of baseline design strategies
        n_monte_carlo : int
            Number of MC samples for IG computation
        device : str or torch.device
            Device for computations
        """
        self.model = model
        self.device = device
        
        # IG calculator
        self.ig_calculator = TrueIGCalculator(
            model=model,
            n_monte_carlo=n_monte_carlo,
            device=device
        )
        
        # Default baseline strategies
        if baseline_strategies is None:
            self.baseline_strategies = self.get_default_baselines()
        else:
            self.baseline_strategies = baseline_strategies
    
    def get_default_baselines(self):
        """Get default baseline design strategies"""
        
        T = self.model.T
        
        baselines = {
            'Uniform': np.linspace(2, 30, T),
            'Early': np.linspace(1, 10, T),
            'Late': np.linspace(20, 40, T),
            'Exponential': np.array([2**i for i in range(T)]) * (30 / 2**(T-1)),
            'Peak_Focus': np.linspace(8, 15, T),  # Around typical peak
            'Random': np.sort(np.random.uniform(1, 40, T))
        }
        
        return baselines
    
    def extract_dad_designs(self):
        """Extract designs from trained DAD model"""
        
        self.model.eval()
        self.model.design_net.eval()
        
        all_designs = []
        
        # Run multiple times to get average behavior
        n_runs = 10
        
        with torch.no_grad():
            for _ in range(n_runs):
                trace = pyro.poutine.trace(self.model.model).get_trace()
                designs = []
                for t in range(self.model.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].item()
                    designs.append(xi)
                all_designs.append(designs)
        
        # Return mean designs
        return np.mean(all_designs, axis=0)
    
    def validate_all(self, save_results=True):
        """
        Validate DAD against all baselines
        
        Parameters:
        -----------
        save_results : bool
            Whether to save results to file
        
        Returns:
        --------
        results : dict
            Validation results
        """
        print("\n" + "=" * 60)
        print("SEIR DAD VALIDATION WITH TRUE INFORMATION GAIN")
        print("=" * 60)
        
        results = {}
        
        # Extract DAD designs
        print("\nExtracting DAD designs...")
        dad_designs = self.extract_dad_designs()
        
        # Compute IG for DAD
        print("\nComputing IG for DAD...")
        dad_ig, dad_ig_std = self.ig_calculator.compute_information_gain(dad_designs)
        results['DAD'] = {
            'designs': dad_designs.tolist(),
            'ig': dad_ig,
            'ig_std': dad_ig_std
        }
        
        # Compute IG for baselines
        for name, designs in self.baseline_strategies.items():
            print(f"\nComputing IG for {name}...")
            ig, ig_std = self.ig_calculator.compute_information_gain(designs)
            results[name] = {
                'designs': designs.tolist(),
                'ig': ig,
                'ig_std': ig_std
            }
        
        # Print results
        self.print_results(results)
        
        # Plot results
        self.plot_results(results)
        
        # Save results
        if save_results:
            self.save_results(results)
        
        return results
    
    def print_results(self, results):
        """Print validation results"""
        
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        # Sort by IG
        sorted_results = sorted(results.items(), key=lambda x: x[1]['ig'], reverse=True)
        
        print(f"\n{'Method':<15} {'IG':<10} {'IG_std':<10} {'Designs'}")
        print("-" * 60)
        
        for method, res in sorted_results:
            designs_str = ', '.join([f"{d:.1f}" for d in res['designs'][:3]]) + "..."
            print(f"{method:<15} {res['ig']:<10.4f} {res['ig_std']:<10.4f} [{designs_str}]")
        
        # Winner
        winner = sorted_results[0][0]
        print(f"\n🏆 Best method: {winner} (IG = {sorted_results[0][1]['ig']:.4f})")
        
        # DAD improvement
        dad_ig = results['DAD']['ig']
        best_baseline = max([res['ig'] for name, res in results.items() if name != 'DAD'])
        improvement = (dad_ig - best_baseline) / best_baseline * 100
        
        if improvement > 0:
            print(f"✓ DAD improvement over best baseline: {improvement:.1f}%")
        else:
            print(f"✗ DAD underperforms best baseline by: {-improvement:.1f}%")
    
    def plot_results(self, results):
        """Plot validation results"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # IG comparison
        ax = axes[0]
        methods = list(results.keys())
        igs = [results[m]['ig'] for m in methods]
        ig_stds = [results[m]['ig_std'] for m in methods]
        colors = ['green' if m == 'DAD' else 'gray' for m in methods]
        
        bars = ax.bar(methods, igs, yerr=ig_stds, capsize=5, color=colors, alpha=0.7)
        ax.set_ylabel('Information Gain')
        ax.set_title('Information Gain Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Design patterns
        ax = axes[1]
        for method, res in results.items():
            alpha = 1.0 if method == 'DAD' else 0.5
            linewidth = 2 if method == 'DAD' else 1
            ax.plot(range(1, len(res['designs'])+1), res['designs'], 
                   'o-', label=method, alpha=alpha, linewidth=linewidth)
        
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Observation Time (days)')
        ax.set_title('Design Patterns')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # IG with error bars
        ax = axes[2]
        sorted_results = sorted(results.items(), key=lambda x: x[1]['ig'], reverse=True)
        methods = [m for m, _ in sorted_results]
        igs = [r['ig'] for _, r in sorted_results]
        ig_stds = [r['ig_std'] for _, r in sorted_results]
        
        y_pos = np.arange(len(methods))
        colors = ['green' if m == 'DAD' else 'gray' for m in methods]
        
        ax.barh(y_pos, igs, xerr=ig_stds, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.set_xlabel('Information Gain')
        ax.set_title('Ranked Information Gain')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, results):
        """Save validation results"""
        
        # Create save directory if needed
        save_dir = 'results/seir_validation/'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save results as JSON
        results_path = os.path.join(save_dir, 'validation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        
        # Save plot
        fig = self.plot_results(results)
        fig_path = os.path.join(save_dir, 'validation_plots.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {fig_path}")


def validate_trained_model(model_path, device='cpu'):
    """
    Validate a trained SEIR DAD model
    
    Parameters:
    -----------
    model_path : str
        Path to saved model checkpoint
    device : str or torch.device
        Device for computations
    
    Returns:
    --------
    results : dict
        Validation results
    """
    print("Loading trained model...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Recreate model
    design_net = create_seir_dad_network(
        hidden_dim=config['hidden_dim'],
        encoding_dim=config['encoding_dim'],
        n_hidden_layers=config['n_hidden_layers'],
        activation=config['activation'],
        aggregation=config['aggregation']
    ).to(device)
    
    model = SEIR_DAD_Model(
        design_net=design_net,
        N=config['N'],
        T=config['T'],
        noise_type=config['noise_type'],
        noise_scale=config['noise_scale'],
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    design_net.load_state_dict(checkpoint['design_net_state_dict'])
    
    print("Model loaded successfully!")
    
    # Create validator
    validator = SEIR_Validator(
        model=model,
        n_monte_carlo=200,
        device=device
    )
    
    # Run validation
    results = validator.validate_all(save_results=True)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate SEIR DAD Model")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--n-mc', type=int, default=200, help='Number of MC samples for IG')
    parser.add_argument('--device', type=str, default='cpu', help='Device for computation')
    
    args = parser.parse_args()
    
    # Validate model
    results = validate_trained_model(
        model_path=args.model_path,
        device=args.device
    )
    
    print("\nValidation complete!")