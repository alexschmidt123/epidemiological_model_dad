"""
Data Generation for SEIR DAD Model
Generates training and testing datasets
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from seir_ode_model import SEIR_ODE, SEIR_DAD_Model
import pickle
import os


class SEIR_DataGenerator:
    """Generate synthetic data for SEIR model training and testing"""
    
    def __init__(
        self,
        N=500,
        noise_scale=0.1,
        noise_type='proportional',
        param_ranges=None,
        device='cpu'
    ):
        """
        Initialize data generator
        
        Parameters:
        -----------
        N : int
            Population size
        noise_scale : float
            Scale of observation noise
        noise_type : str
            Type of noise ('proportional', 'constant', or 'hybrid')
        param_ranges : dict
            Ranges for parameters (beta, sigma, gamma)
        device : str or torch.device
            Device for computations
        """
        self.N = N
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.device = device
        
        # Default parameter ranges (reasonable epidemic parameters)
        if param_ranges is None:
            self.param_ranges = {
                'beta': (0.2, 1.0),    # Transmission rate
                'sigma': (0.1, 0.5),   # Incubation rate (1/incubation_period)
                'gamma': (0.05, 0.3)   # Recovery rate (1/infectious_period)
            }
        else:
            self.param_ranges = param_ranges
    
    def sample_parameters(self, n_samples, distribution='uniform'):
        """
        Sample epidemic parameters
        
        Parameters:
        -----------
        n_samples : int
            Number of parameter sets to sample
        distribution : str
            'uniform' or 'lognormal'
        
        Returns:
        --------
        params : np.array
            Array of parameters with shape (n_samples, 3)
        """
        params = np.zeros((n_samples, 3))
        
        if distribution == 'uniform':
            params[:, 0] = np.random.uniform(*self.param_ranges['beta'], n_samples)
            params[:, 1] = np.random.uniform(*self.param_ranges['sigma'], n_samples)
            params[:, 2] = np.random.uniform(*self.param_ranges['gamma'], n_samples)
        
        elif distribution == 'lognormal':
            # Log-normal centered around middle of ranges
            for i, (key, (low, high)) in enumerate(self.param_ranges.items()):
                mean = (low + high) / 2
                std = (high - low) / 4
                params[:, i] = np.random.lognormal(np.log(mean), 0.3, n_samples)
                params[:, i] = np.clip(params[:, i], low, high)
        
        return params
    
    def generate_trajectory(self, params, obs_times):
        """
        Generate a single epidemic trajectory with observations
        
        Parameters:
        -----------
        params : array
            Parameters [beta, sigma, gamma]
        obs_times : array
            Times to make observations
        
        Returns:
        --------
        observations : array
            Noisy observations at specified times
        true_infected : array
            True infected counts at observation times
        full_trajectory : dict
            Complete trajectory for visualization
        """
        # Initial conditions
        E0, I0 = 1, 2
        S0 = self.N - E0 - I0
        R0 = 0
        y0 = [S0, E0, I0, R0]
        
        # Solve ODE for full time range
        t_max = max(obs_times) * 1.1
        t_full = np.linspace(0, t_max, 500)
        solution_full = odeint(SEIR_ODE.dynamics, y0, t_full, args=tuple(params))
        
        # Get values at observation times
        solution_obs = odeint(SEIR_ODE.dynamics, y0, obs_times, args=tuple(params))
        true_infected = solution_obs[:, 2]  # I compartment
        
        # Add observation noise
        observations = self.add_noise(true_infected)
        
        # Full trajectory for visualization
        full_trajectory = {
            'time': t_full,
            'S': solution_full[:, 0],
            'E': solution_full[:, 1],
            'I': solution_full[:, 2],
            'R': solution_full[:, 3],
            'obs_times': obs_times,
            'obs_values': observations,
            'true_values': true_infected
        }
        
        return observations, true_infected, full_trajectory
    
    def add_noise(self, true_values):
        """Add observation noise to true values"""
        
        true_values = np.array(true_values)
        
        if self.noise_type == 'proportional':
            # Noise proportional to sqrt of count (Poisson-like)
            noise_std = self.noise_scale * np.sqrt(true_values + 1.0)
        elif self.noise_type == 'constant':
            noise_std = self.noise_scale * np.ones_like(true_values) * 10
        else:  # hybrid
            noise_std = self.noise_scale * (5.0 + 0.1 * true_values)
        
        noise = np.random.normal(0, noise_std)
        observations = true_values + noise
        
        # Ensure non-negative
        observations = np.maximum(observations, 0)
        
        return observations
    
    def generate_dataset(
        self,
        n_samples=1000,
        T=4,
        design_strategy='uniform',
        save_path=None
    ):
        """
        Generate a complete dataset
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        T : int
            Number of observations per trajectory
        design_strategy : str or callable
            Strategy for choosing observation times:
            - 'uniform': Evenly spaced
            - 'random': Random times
            - 'early': Concentrated early
            - 'late': Concentrated late
            - 'adaptive': Simulated adaptive (peak-focused)
            - callable: Custom function(params) -> obs_times
        save_path : str, optional
            Path to save dataset
        
        Returns:
        --------
        dataset : dict
            Dictionary containing all data
        """
        dataset = {
            'params': [],
            'designs': [],
            'observations': [],
            'true_infected': [],
            'trajectories': []
        }
        
        # Sample parameters
        all_params = self.sample_parameters(n_samples, distribution='uniform')
        
        for i, params in enumerate(all_params):
            # Get observation times based on strategy
            obs_times = self.get_design_times(params, T, design_strategy)
            
            # Generate trajectory
            obs, true_inf, trajectory = self.generate_trajectory(params, obs_times)
            
            # Store data
            dataset['params'].append(params)
            dataset['designs'].append(obs_times)
            dataset['observations'].append(obs)
            dataset['true_infected'].append(true_inf)
            dataset['trajectories'].append(trajectory)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")
        
        # Convert to numpy arrays
        dataset['params'] = np.array(dataset['params'])
        dataset['designs'] = np.array(dataset['designs'])
        dataset['observations'] = np.array(dataset['observations'])
        dataset['true_infected'] = np.array(dataset['true_infected'])
        
        # Add metadata
        dataset['metadata'] = {
            'N': self.N,
            'noise_scale': self.noise_scale,
            'noise_type': self.noise_type,
            'n_samples': n_samples,
            'T': T,
            'design_strategy': design_strategy
        }
        
        # Save if requested
        if save_path:
            self.save_dataset(dataset, save_path)
        
        return dataset
    
    def get_design_times(self, params, T, strategy):
        """
        Get observation times based on strategy
        
        Parameters:
        -----------
        params : array
            Parameters [beta, sigma, gamma]
        T : int
            Number of observations
        strategy : str or callable
            Design strategy
        
        Returns:
        --------
        obs_times : array
            Observation times
        """
        if callable(strategy):
            return strategy(params)
        
        if strategy == 'uniform':
            # Evenly spaced from day 2 to day 30
            return np.linspace(2, 30, T)
        
        elif strategy == 'random':
            # Random times between day 1 and day 40
            times = np.sort(np.random.uniform(1, 40, T))
            return times
        
        elif strategy == 'early':
            # Concentrated in first 15 days
            return np.linspace(1, 15, T)
        
        elif strategy == 'late':
            # Concentrated in days 15-40
            return np.linspace(15, 40, T)
        
        elif strategy == 'adaptive':
            # Focus around expected peak (simulated adaptive behavior)
            # Estimate peak time from parameters
            R0 = params[0] / params[2]  # beta / gamma
            if R0 > 1:
                # Rough estimate of peak time
                peak_time = 10 / params[1]  # Inversely related to sigma
                peak_time = np.clip(peak_time, 5, 30)
                
                # Concentrate observations around peak
                times = peak_time + np.linspace(-5, 10, T)
                times = np.clip(times, 1, 40)
                return np.sort(times)
            else:
                # No epidemic, use uniform
                return np.linspace(2, 30, T)
        
        elif strategy == 'exponential':
            # Exponentially increasing intervals
            times = np.array([2**i for i in range(T)])
            times = times * (30 / times[-1])  # Scale to end at day 30
            return times
        
        else:
            raise ValueError(f"Unknown design strategy: {strategy}")
    
    def save_dataset(self, dataset, path):
        """Save dataset to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {path}")
    
    def load_dataset(self, path):
        """Load dataset from file"""
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Dataset loaded from {path}")
        return dataset
    
    def visualize_samples(self, dataset, n_samples=4):
        """
        Visualize sample trajectories from dataset
        
        Parameters:
        -----------
        dataset : dict
            Generated dataset
        n_samples : int
            Number of samples to visualize
        """
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        indices = np.random.choice(len(dataset['params']), n_samples, replace=False)
        
        for idx, i in enumerate(indices):
            trajectory = dataset['trajectories'][i]
            params = dataset['params'][i]
            
            # Plot full trajectory
            ax = axes[idx, 0]
            ax.plot(trajectory['time'], trajectory['S'], label='S', alpha=0.7)
            ax.plot(trajectory['time'], trajectory['E'], label='E', alpha=0.7)
            ax.plot(trajectory['time'], trajectory['I'], label='I', alpha=0.7)
            ax.plot(trajectory['time'], trajectory['R'], label='R', alpha=0.7)
            
            # Mark observations
            ax.scatter(trajectory['obs_times'], trajectory['obs_values'],
                      color='red', s=50, zorder=5, label='Observations')
            ax.scatter(trajectory['obs_times'], trajectory['true_values'],
                      color='black', s=30, zorder=4, alpha=0.5, label='True I')
            
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Count')
            ax.set_title(f'β={params[0]:.2f}, σ={params[1]:.2f}, γ={params[2]:.2f}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Plot infected only with observations
            ax = axes[idx, 1]
            ax.plot(trajectory['time'], trajectory['I'], 'b-', label='True I(t)', alpha=0.7)
            ax.scatter(trajectory['obs_times'], trajectory['obs_values'],
                      color='red', s=50, label='Noisy Obs', alpha=0.8)
            ax.scatter(trajectory['obs_times'], trajectory['true_values'],
                      color='black', s=30, alpha=0.5, label='True at Obs')
            
            # Mark observation times with vertical lines
            for t in trajectory['obs_times']:
                ax.axvline(t, color='gray', linestyle='--', alpha=0.3)
            
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Infected Count')
            ax.set_title(f'Infected Trajectory (R0≈{params[0]/params[2]:.1f})')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class SEIR_PyroDataset(Dataset):
    """PyTorch Dataset wrapper for SEIR data"""
    
    def __init__(self, data_dict, transform=None):
        """
        Initialize dataset
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary from generate_dataset
        transform : callable, optional
            Transform to apply to samples
        """
        self.params = torch.FloatTensor(data_dict['params'])
        self.designs = torch.FloatTensor(data_dict['designs'])
        self.observations = torch.FloatTensor(data_dict['observations'])
        self.true_infected = torch.FloatTensor(data_dict['true_infected'])
        self.transform = transform
        
    def __len__(self):
        return len(self.params)
    
    def __getitem__(self, idx):
        sample = {
            'params': self.params[idx],
            'designs': self.designs[idx],
            'observations': self.observations[idx],
            'true_infected': self.true_infected[idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def generate_train_test_data(
    n_train=5000,
    n_test=1000,
    T=4,
    save_dir='data/seir/',
    device='cpu'
):
    """
    Generate training and testing datasets
    
    Parameters:
    -----------
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    T : int
        Number of observations per trajectory
    save_dir : str
        Directory to save datasets
    device : str or torch.device
        Device for computations
    
    Returns:
    --------
    train_dataset : dict
        Training dataset
    test_dataset : dict
        Testing dataset
    """
    generator = SEIR_DataGenerator(device=device)
    
    print("Generating training data...")
    train_dataset = generator.generate_dataset(
        n_samples=n_train,
        T=T,
        design_strategy='random',  # Varied designs for training
        save_path=os.path.join(save_dir, 'train_data.pkl')
    )
    
    print("\nGenerating test data...")
    test_dataset = generator.generate_dataset(
        n_samples=n_test,
        T=T,
        design_strategy='uniform',  # Fixed strategy for testing
        save_path=os.path.join(save_dir, 'test_data.pkl')
    )
    
    # Create visualization
    print("\nCreating visualization...")
    fig = generator.visualize_samples(train_dataset, n_samples=3)
    fig.savefig(os.path.join(save_dir, 'sample_trajectories.png'), dpi=150)
    plt.close()
    
    print(f"\nDatasets saved to {save_dir}")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Generate datasets
    train_data, test_data = generate_train_test_data(
        n_train=1000,
        n_test=200,
        T=4,
        save_dir='data/seir/'
    )
    
    # Show sample statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    print(f"Population size: {train_data['metadata']['N']}")
    print(f"Observations per trajectory: {train_data['metadata']['T']}")
    print(f"Noise type: {train_data['metadata']['noise_type']}")
    print(f"Noise scale: {train_data['metadata']['noise_scale']}")
    
    # Parameter statistics
    train_params = train_data['params']
    print(f"\nParameter ranges (training):")
    print(f"  β (transmission): [{train_params[:, 0].min():.3f}, {train_params[:, 0].max():.3f}]")
    print(f"  σ (incubation):   [{train_params[:, 1].min():.3f}, {train_params[:, 1].max():.3f}]")
    print(f"  γ (recovery):     [{train_params[:, 2].min():.3f}, {train_params[:, 2].max():.3f}]")
    
    # R0 distribution
    R0_values = train_params[:, 0] / train_params[:, 2]
    print(f"\nR0 distribution:")
    print(f"  Mean: {R0_values.mean():.2f}")
    print(f"  Std:  {R0_values.std():.2f}")
    print(f"  Range: [{R0_values.min():.2f}, {R0_values.max():.2f}]")