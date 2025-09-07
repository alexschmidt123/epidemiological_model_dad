import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.contrib.util import lexpand, rexpand
from pyro.poutine.util import prune_subsample_sites
import torchsde

from oed.primitives import observation_sample, compute_design, latent_sample
from neural.modules import SetEquivariantDesignNetwork


class SEIR_SDE(torch.nn.Module):
    """SEIR SDE model for use in DAD framework"""
    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, population_size):
        super().__init__()
        self.params = params
        self.N = population_size

    def f_and_g(self, t, x):
        with torch.no_grad():
            x.clamp_(0.0, self.N)

        S, E, I = x.T
        beta, sigma, gamma = self.params.T

        # Drift terms
        f_term = torch.stack([
            -beta * S * I / self.N,                    # dS/dt
            beta * S * I / self.N - sigma * E,         # dE/dt
            sigma * E - gamma * I                      # dI/dt
        ], dim=-1)

        # Diffusion terms
        batch_size = S.shape[0]

        g_S = torch.zeros(batch_size, 3)
        g_S[:, 0] = -torch.sqrt(beta * S * I / self.N).squeeze(-1)

        g_E = torch.zeros(batch_size, 3)
        g_E[:, 0] = torch.sqrt(beta * S * I / self.N).squeeze(-1)
        g_E[:, 1] = -torch.sqrt(sigma * E).squeeze(-1)

        g_I = torch.zeros(batch_size, 3)
        g_I[:, 1] = torch.sqrt(sigma * E).squeeze(-1)
        g_I[:, 2] = -torch.sqrt(gamma * I).squeeze(-1)

        g_term = torch.stack([g_S, g_E, g_I], dim=-1)
        return f_term, g_term


class SEIR_DAD_Model(nn.Module):
    """SEIR model adapted for Deep Adaptive Design"""
    
    def __init__(
        self,
        design_net,
        population_size=500.0,
        initial_exposed=1.0,
        initial_infected=2.0,
        T=4,  # number of design experiments
        observation_times=None,  # time points for SDE integration
        theta_prior_loc=None,
        theta_prior_scale=None,
    ):
        super().__init__()
        
        self.design_net = design_net
        self.N = population_size
        self.E0 = initial_exposed
        self.I0 = initial_infected
        self.T = T
        
        # Default observation times (can be adaptive)
        if observation_times is None:
            self.observation_times = torch.linspace(0, 50, 100)
        else:
            self.observation_times = observation_times
            
        # Prior parameters: [beta, sigma, gamma]
        if theta_prior_loc is None:
            self.theta_prior_loc = torch.tensor([0.3, 0.1, 0.1]).log()
        else:
            self.theta_prior_loc = theta_prior_loc
            
        if theta_prior_scale is None:
            self.theta_prior_scale = torch.eye(3) * 0.5 ** 2
        else:
            self.theta_prior_scale = theta_prior_scale
            
        self.softplus = nn.Softplus()

    def solve_seir_sde(self, params, design_time):
        """Solve SEIR SDE up to design_time and return infected count"""
        
        # Initial conditions [S, E, I]
        y0 = torch.tensor([[
            self.N - self.E0 - self.I0, 
            self.E0, 
            self.I0
        ]], device=params.device).expand(params.shape[0], -1)
        
        # Time points: from 0 to design_time
        ts = torch.linspace(0, design_time, 50, device=params.device)
        
        # Create SDE
        sde = SEIR_SDE(params=params, population_size=self.N)
        
        # Solve SDE
        try:
            ys = torchsde.sdeint(sde, y0, ts)  # shape: [time_steps, batch, 3]
            
            # Return infected count at final time
            infected_final = ys[-1, :, 2]  # I compartment at final time
            return infected_final.clamp(min=0)
            
        except Exception as e:
            # Fallback: return deterministic solution or prior
            print(f"SDE integration failed: {e}")
            return torch.ones(params.shape[0], device=params.device) * self.I0

    def model(self):
        """Pyro model for DAD framework"""
        
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        # Sample latent parameters [beta, sigma, gamma]
        theta_prior = dist.MultivariateNormal(
            self.theta_prior_loc, 
            self.theta_prior_scale
        )
        theta_log = latent_sample("theta", theta_prior)
        theta = theta_log.exp()  # Convert to positive parameters
        
        y_outcomes = []
        xi_designs = []
        
        for t in range(self.T):
            # Design experiment: choose observation time
            xi = compute_design(
                f"xi{t + 1}", 
                self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            
            # Ensure positive time and reasonable bounds
            xi = self.softplus(xi.squeeze(-1)).clamp(min=1.0, max=100.0)
            
            # Solve SEIR SDE up to time xi
            infected_count = self.solve_seir_sde(theta, xi)
            
            # Observe with noise (Poisson or Negative Binomial)
            # Using Poisson for simplicity, but could use more complex observation model
            y = observation_sample(
                f"y{t + 1}", 
                dist.Poisson(infected_count + 1e-6)  # Small epsilon for numerical stability
            )
            
            y_outcomes.append(y)
            xi_designs.append(xi)
            
        return y_outcomes

    def eval(self, n_trace=2, theta=None):
        """Evaluation method similar to death process"""
        self.design_net.eval()
        
        if theta is not None:
            model = pyro.condition(self.model, data={"theta": theta})
        else:
            model = self.model
            
        output = []
        with torch.no_grad():
            for i in range(n_trace):
                trace = pyro.poutine.trace(model).get_trace()
                
                true_theta = trace.nodes["theta"]["value"]
                run_xis = []
                run_ys = []
                
                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].item()
                    run_xis.append(xi)
                    
                    y = trace.nodes[f"y{t + 1}"]["value"].item()
                    run_ys.append(y)
                
                run_dict = {
                    "designs": run_xis,
                    "observations": run_ys,
                    "theta": true_theta.exp().cpu().numpy(),  # Convert back to original scale
                    "run_id": i + 1
                }
                output.append(run_dict)
                
        return output


def create_seir_design_network(
    design_dim=1,  # Time dimension
    hidden_dim=128,
    encoding_dim=16,
    num_layers=2
):
    """Create design network for SEIR model"""
    
    from neural.modules import Mlp
    
    # Encoder: takes (time_design, infected_count) pairs
    encoder = Mlp(
        input_dim=2,  # [time, infected_count]
        hidden_dim=hidden_dim,
        output_dim=encoding_dim,
        n_hidden_layers=num_layers
    )
    
    # Emitter: outputs next time design
    emitter = Mlp(
        input_dim=encoding_dim,
        hidden_dim=hidden_dim,
        output_dim=design_dim,
        n_hidden_layers=num_layers
    )
    
    # Create set-equivariant design network
    design_net = SetEquivariantDesignNetwork(
        encoder_network=encoder,
        emission_network=emitter,
        empty_value=torch.ones(design_dim)  # Default first design
    )
    
    return design_net


# Example usage
if __name__ == "__main__":
    device = torch.device("cpu")  # or "cuda"
    
    # Create design network
    design_net = create_seir_design_network().to(device)
    
    # Create SEIR DAD model
    seir_model = SEIR_DAD_Model(
        design_net=design_net,
        T=4,  # 4 sequential experiments
        population_size=500.0
    )
    
    # Test evaluation
    results = seir_model.eval(n_trace=2)
    print("SEIR DAD evaluation results:")
    for result in results:
        print(f"Run {result['run_id']}: designs={result['designs']}, observations={result['observations']}")
