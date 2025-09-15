"""
SEIR ODE Model Definition
Deterministic SEIR model with Gaussian observation noise
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
import pyro
import pyro.distributions as dist
from pyro.contrib.util import lexpand


class SEIR_ODE:
    """Deterministic SEIR ODE model"""
    
    @staticmethod
    def dynamics(y, t, beta, sigma, gamma):
        """
        SEIR model differential equations
        
        Parameters:
        -----------
        y : array
            State vector [S, E, I, R]
        t : float
            Time
        beta : float
            Transmission rate
        sigma : float
            Incubation rate (E to I)
        gamma : float
            Recovery rate
        
        Returns:
        --------
        dydt : array
            Derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        S, E, I, R = y
        N = S + E + I + R
        
        dS = -beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        
        return [dS, dE, dI, dR]
    
    @staticmethod
    def solve(params, t_eval, y0=None, N=500):
        """
        Solve SEIR ODE
        
        Parameters:
        -----------
        params : array or tensor
            Parameters [beta, sigma, gamma]
        t_eval : array
            Time points to evaluate
        y0 : array, optional
            Initial conditions [S0, E0, I0, R0]
        N : int
            Total population
        
        Returns:
        --------
        solution : array
            Solution array with shape (len(t_eval), 4)
        """
        # Convert to numpy if needed
        if torch.is_tensor(params):
            params = params.detach().cpu().numpy()
        
        # Default initial conditions
        if y0 is None:
            E0, I0 = 1, 2  # Start with 1 exposed, 2 infected
            S0 = N - E0 - I0
            R0 = 0
            y0 = [S0, E0, I0, R0]
        
        # Solve ODE
        solution = odeint(SEIR_ODE.dynamics, y0, t_eval, args=tuple(params))
        
        return solution


class SEIR_DAD_Model(nn.Module):
    """
    SEIR model for Deep Adaptive Design
    
    This model:
    - Uses deterministic ODE dynamics
    - Adds Gaussian observation noise
    - Optimizes observation times for maximum information gain
    """
    
    def __init__(
        self,
        design_net,
        N=500,                    # Population size
        T=4,                      # Number of experiments
        E0=1,                     # Initial exposed
        I0=2,                     # Initial infected
        noise_type='proportional', # 'proportional' or 'constant'
        noise_scale=0.1,          # Noise scale parameter
        theta_prior_loc=None,     # Prior mean (log scale)
        theta_prior_scale=None,   # Prior covariance
        device='cpu'
    ):
        super().__init__()
        
        self.design_net = design_net
        self.N = N
        self.T = T
        self.E0 = E0
        self.I0 = I0
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.device = device
        
        # Initial conditions
        self.y0 = [N - E0 - I0, E0, I0, 0]
        
        # Prior parameters for [beta, sigma, gamma] in log space
        if theta_prior_loc is None:
            # Default: R0 ≈ 2-3, incubation ≈ 5 days, infectious ≈ 7 days
            self.theta_prior_loc = torch.tensor([0.4, 0.2, 0.14], device=device).log()
        else:
            self.theta_prior_loc = theta_prior_loc
            
        if theta_prior_scale is None:
            # Moderate uncertainty
            self.theta_prior_scale = torch.eye(3, device=device) * 0.3
        else:
            self.theta_prior_scale = theta_prior_scale
        
        # Prior distribution
        self.theta_prior = dist.MultivariateNormal(
            self.theta_prior_loc, 
            self.theta_prior_scale
        )
        
        # Transforms
        self.softplus = nn.Softplus()
        
    def solve_seir_at_time(self, params, obs_time):
        """
        Solve SEIR ODE and return infected count at observation time
        
        Parameters:
        -----------
        params : tensor
            Parameters [beta, sigma, gamma]
        obs_time : float or tensor
            Time to observe
        
        Returns:
        --------
        infected : tensor
            Number of infected individuals at obs_time
        """
        # Convert obs_time to float
        if torch.is_tensor(obs_time):
            obs_time = obs_time.item()
        
        # Ensure reasonable time bounds
        obs_time = max(0.1, min(obs_time, 100.0))
        
        # Time points for integration
        t_eval = np.linspace(0, obs_time, max(10, int(obs_time * 2)))
        
        # Solve ODE
        solution = SEIR_ODE.solve(params, t_eval, self.y0, self.N)
        
        # Return infected count at final time
        infected = solution[-1, 2]  # I compartment
        
        return torch.tensor(infected, dtype=torch.float32, device=self.device)
    
    def get_observation_noise(self, infected_count):
        """
        Get observation noise standard deviation
        
        Parameters:
        -----------
        infected_count : tensor
            True infected count
        
        Returns:
        --------
        noise_std : tensor
            Standard deviation for observation noise
        """
        if self.noise_type == 'proportional':
            # Noise proportional to count (more realistic)
            noise_std = self.noise_scale * torch.sqrt(infected_count + 1.0)
        elif self.noise_type == 'constant':
            # Constant noise
            noise_std = self.noise_scale * torch.ones_like(infected_count)
        else:
            # Hybrid: minimum noise floor + proportional
            noise_std = self.noise_scale * (5.0 + 0.1 * infected_count)
        
        return noise_std
    
    def model(self):
        """
        Pyro model for Deep Adaptive Design
        
        This defines the generative process:
        1. Sample parameters from prior
        2. For each experiment:
           - Choose observation time (design)
           - Solve ODE to get true infected count
           - Add observation noise
        """
        
        # Register design network with pyro
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)
        
        # Sample parameters from prior (in log space)
        theta_log = pyro.sample("theta", self.theta_prior)
        theta = theta_log.exp()  # Convert to positive values
        
        # Ensure reasonable parameter bounds
        theta = theta.clamp(min=1e-6, max=10.0)
        
        # Storage for history
        y_outcomes = []
        xi_designs = []
        
        # Sequential experiment design
        for t in range(self.T):
            # Get design (observation time) based on history
            # This is where DAD adapts based on previous observations
            from oed.primitives import compute_design
            
            xi = compute_design(
                f"xi{t + 1}",
                self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            
            # Transform to positive time with reasonable bounds
            # Softplus ensures positivity, scaling maps to good range
            xi = 1.0 + 30.0 * self.softplus(xi.squeeze(-1))  # Range: ~[1, 31]
            
            # Solve SEIR ODE to get true infected count
            true_infected = self.solve_seir_at_time(theta, xi)
            
            # Get observation noise
            noise_std = self.get_observation_noise(true_infected)
            
            # Observe with noise
            from oed.primitives import observation_sample
            
            y = observation_sample(
                f"y{t + 1}",
                dist.Normal(true_infected, noise_std)
            )
            
            # Store for history
            y_outcomes.append(y)
            xi_designs.append(xi)
        
        return y_outcomes
    
    def forward(self, num_samples=1):
        """
        Forward pass for generating synthetic data
        
        Parameters:
        -----------
        num_samples : int
            Number of parameter samples to generate
        
        Returns:
        --------
        data : dict
            Dictionary containing parameters, designs, and observations
        """
        data = {
            'params': [],
            'designs': [],
            'observations': [],
            'true_infected': []
        }
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample parameters
                theta_log = self.theta_prior.sample()
                theta = theta_log.exp()
                
                # Run through experiment sequence
                designs = []
                observations = []
                true_counts = []
                
                # Initialize history for design network
                history = []
                
                for t in range(self.T):
                    # Get design
                    if t == 0:
                        xi = self.design_net.empty_value
                    else:
                        # Encode history and get next design
                        encoding = sum(
                            self.design_net.encoder(d, o) 
                            for d, o in history
                        )
                        xi = self.design_net.emitter(encoding)
                    
                    # Transform to time
                    xi = 1.0 + 30.0 * self.softplus(xi.squeeze(-1))
                    
                    # Solve ODE
                    true_infected = self.solve_seir_at_time(theta, xi)
                    
                    # Add noise
                    noise_std = self.get_observation_noise(true_infected)
                    y = dist.Normal(true_infected, noise_std).sample()
                    
                    # Store
                    designs.append(xi.item())
                    observations.append(y.item())
                    true_counts.append(true_infected.item())
                    history.append((xi, y))
                
                data['params'].append(theta.cpu().numpy())
                data['designs'].append(designs)
                data['observations'].append(observations)
                data['true_infected'].append(true_counts)
        
        return data
    
    def evaluate(self, n_trace=5, theta=None):
        """
        Evaluate the model with trained design network
        
        Parameters:
        -----------
        n_trace : int
            Number of evaluation traces
        theta : tensor, optional
            Fixed parameters (if None, sample from prior)
        
        Returns:
        --------
        results : list
            List of evaluation results
        """
        self.design_net.eval()
        
        if theta is not None:
            model = pyro.condition(self.model, data={"theta": theta.log()})
        else:
            model = self.model
        
        results = []
        
        with torch.no_grad():
            for i in range(n_trace):
                trace = pyro.poutine.trace(model).get_trace()
                
                # Extract results
                true_theta = trace.nodes["theta"]["value"].exp()
                run_xis = []
                run_ys = []
                
                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].item()
                    y = trace.nodes[f"y{t + 1}"]["value"].item()
                    run_xis.append(xi)
                    run_ys.append(y)
                
                results.append({
                    'run_id': i + 1,
                    'theta': true_theta.cpu().numpy(),
                    'param_names': ['beta', 'sigma', 'gamma'],
                    'designs': run_xis,
                    'observations': run_ys
                })
        
        return results