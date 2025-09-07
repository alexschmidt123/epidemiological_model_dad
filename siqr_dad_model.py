import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.contrib.util import lexpand, rexpand
from pyro.poutine.util import prune_subsample_sites
import torchsde

from oed.primitives import observation_sample, compute_design, latent_sample
from neural.modules import SetEquivariantDesignNetwork


class SIQR_SDE(torch.nn.Module):
    """SIQR SDE model for use in DAD framework"""
    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, population_size):
        super().__init__()
        self.params = params  # [beta, alpha, gamma, delta]
        self.N = population_size

    def f_and_g(self, t, x):
        S, I, Q, R = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        with torch.no_grad():
            x.clamp_(0.0, self.N)

        beta, alpha, gamma, delta = self.params[:, 0], self.params[:, 1], self.params[:, 2], self.params[:, 3]

        # Transition rates
        p_inf = beta * S * I / self.N
        p_inf_sqrt = torch.sqrt(p_inf)
        p_rec = gamma * I
        p_qua = alpha * I
        p_qua_sqrt = torch.sqrt(p_qua)
        p_delta = delta * Q
        p_delta_sqrt = torch.sqrt(p_delta)
        p_rec_sqrt = torch.sqrt(p_rec)

        # Drift terms (f)
        f_S = -p_inf
        f_I = p_inf - p_qua - p_rec
        f_Q = p_qua - p_delta
        f_R = p_rec + p_delta

        f_term = torch.stack([f_S, f_I, f_Q, f_R], dim=-1)

        # Diffusion terms (g)
        g_S = torch.stack([
            -p_inf_sqrt, 
            torch.zeros_like(p_inf_sqrt), 
            torch.zeros_like(p_inf_sqrt), 
            torch.zeros_like(p_inf_sqrt)
        ], dim=-1)
        
        g_I = torch.stack([
            p_inf_sqrt, 
            torch.sqrt(p_rec + p_qua), 
            torch.zeros_like(p_inf_sqrt), 
            torch.zeros_like(p_inf_sqrt)
        ], dim=-1)
        
        g_Q = torch.stack([
            torch.zeros_like(p_qua_sqrt), 
            p_qua_sqrt, 
            -p_delta_sqrt, 
            torch.zeros_like(p_inf_sqrt)
        ], dim=-1)
        
        g_R = torch.stack([
            torch.zeros_like(p_inf_sqrt), 
            p_rec_sqrt, 
            p_delta_sqrt, 
            torch.zeros_like(p_inf_sqrt)
        ], dim=-1)

        g_term = torch.stack([g_S, g_I, g_Q, g_R], dim=-1)
        return f_term, g_term


class SIQR_DAD_Model(nn.Module):
    """SIQR model adapted for Deep Adaptive Design"""
    
    def __init__(
        self,
        design_net,
        population_size=500.0,
        initial_infected=5.0,
        T=4,  # number of design experiments
        theta_prior_loc=None,
        theta_prior_scale=None,
        design_type="time",  # "time" or "intervention"
    ):
        super().__init__()
        
        self.design_net = design_net
        self.N = population_size
        self.I0 = initial_infected
        self.T = T
        self.design_type = design_type
        
        # Prior parameters: [beta, alpha, gamma, delta]
        if theta_prior_loc is None:
            self.theta_prior_loc = torch.tensor([0.9, 0.1, 0.2, 0.2]).log()
        else:
            self.theta_prior_loc = theta_prior_loc
            
        if theta_prior_scale is None:
            self.theta_prior_scale = torch.eye(4) * 0.5 ** 2
        else:
            self.theta_prior_scale = theta_prior_scale
            
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def solve_siqr_sde(self, params, design_input):
        """Solve SIQR SDE with given parameters and design"""
        
        if self.design_type == "time":
            # Design is observation time
            end_time = design_input
            # Initial conditions [S, I, Q, R]
            y0 = torch.tensor([[
                self.N - self.I0, 
                self.I0, 
                0.0, 
                0.0
            ]], device=params.device).expand(params.shape[0], -1)
            
            # Time integration
            ts = torch.linspace(0, end_time, 50, device=params.device)
            
        elif self.design_type == "intervention":
            # Design is intervention strength (e.g., quarantine rate modifier)
            end_time = 20.0  # Fixed observation time
            intervention_strength = design_input
            
            # Modify alpha (quarantine rate) based on intervention
            params_modified = params.clone()
            params_modified[:, 1] = params[:, 1] * (1 + intervention_strength)  # Increase alpha
            
            y0 = torch.tensor([[
                self.N - self.I0, 
                self.I0, 
                0.0, 
                0.0
            ]], device=params.device).expand(params.shape[0], -1)
            
            ts = torch.linspace(0, end_time, 50, device=params.device)
            params = params_modified
        
        # Create and solve SDE
        sde = SIQR_SDE(params=params, population_size=self.N)
        
        try:
            ys = torchsde.sdeint(sde, y0, ts)  # shape: [time_steps, batch, 4]
            
            # Return multiple observables: [I, Q] at final time
            infected_final = ys[-1, :, 1]  # I compartment
            quarantined_final = ys[-1, :, 2]  # Q compartment
            
            return {
                'infected': infected_final.clamp(min=0),
                'quarantined': quarantined_final.clamp(min=0),
                'total_cases': (infected_final + quarantined_final).clamp(min=0)
            }
            
        except Exception as e:
            print(f"SDE integration failed: {e}")
            # Fallback values
            batch_size = params.shape[0]
            return {
                'infected': torch.ones(batch_size, device=params.device) * self.I0,
                'quarantined': torch.zeros(batch_size, device=params.device),
                'total_cases': torch.ones(batch_size, device=params.device) * self.I0
            }

    def model(self):
        """Pyro model for DAD framework"""
        
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        # Sample latent parameters [beta, alpha, gamma, delta]
        theta_prior = dist.MultivariateNormal(
            self.theta_prior_loc, 
            self.theta_prior_scale
        )
        theta_log = latent_sample("theta", theta_prior)
        theta = theta_log.exp()  # Convert to positive parameters
        
        y_outcomes = []
        xi_designs = []
        
        for t in range(self.T):
            # Design experiment
            xi = compute_design(
                f"xi{t + 1}", 
                self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            
            if self.design_type == "time":
                # Time design: ensure positive and reasonable bounds
                xi = self.softplus(xi.squeeze(-1)).clamp(min=1.0, max=50.0)
            elif self.design_type == "intervention":
                # Intervention strength: bounded between 0 and 2 (0-200% increase)
                xi = 2.0 * self.sigmoid(xi.squeeze(-1))
            
            # Solve SIQR SDE
            outcomes = self.solve_siqr_sde(theta, xi)
            
            # Multi-dimensional observation
            # Option 1: Observe total cases (infected + quarantined)
            total_cases = outcomes['total_cases']
            
            # Option 2: Observe infected and quarantined separately
            # infected = outcomes['infected']
            # quarantined = outcomes['quarantined']
            
            # Observation model - using Negative Binomial for over-dispersion
            concentration = 10.0  # Shape parameter for negative binomial
            y = observation_sample(
                f"y{t + 1}", 
                dist.NegativeBinomial(
                    total_count=concentration,
                    probs=concentration / (concentration + total_cases + 1e-6)
                )
            )
            
            # Alternative: Multi-dimensional observation
            # y_infected = observation_sample(f"y_infected_{t + 1}", 
            #                               dist.Poisson(infected + 1e-6))
            # y_quarantined = observation_sample(f"y_quarantined_{t + 1}", 
            #                                   dist.Poisson(quarantined + 1e-6))
            # y = torch.stack([y_infected, y_quarantined], dim=-1)
            
            y_outcomes.append(y)
            xi_designs.append(xi)
            
        return y_outcomes

    def eval(self, n_trace=2, theta=None):
        """Evaluation method"""
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
                    "theta": true_theta.exp().cpu().numpy(),  # [beta, alpha, gamma, delta]
                    "run_id": i + 1,
                    "design_type": self.design_type
                }
                output.append(run_dict)
                
        return output


def create_siqr_design_network(
    design_dim=1,
    design_type="time",
    hidden_dim=128,
    encoding_dim=16,
    num_layers=2
):
    """Create design network for SIQR model"""
    
    from neural.modules import Mlp
    
    # Input dimension depends on observation type
    if design_type == "time":
        input_dim = 2  # [time_design, total_cases] or [time_design, cases]
    else:  # intervention
        input_dim = 2  # [intervention_strength, total_cases]
    
    # Encoder: takes (design, observation) pairs
    encoder = Mlp(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=encoding_dim,
        n_hidden_layers=num_layers
    )
    
    # Emitter: outputs next design
    emitter = Mlp(
        input_dim=encoding_dim,
        hidden_dim=hidden_dim,
        output_dim=design_dim,
        n_hidden_layers=num_layers
    )
    
    # Create set-equivariant design network
    if design_type == "time":
        empty_value = torch.tensor([10.0])  # Default first time
    else:  # intervention
        empty_value = torch.tensor([0.5])   # Default intervention strength
    
    design_net = SetEquivariantDesignNetwork(
        encoder_network=encoder,
        emission_network=emitter,
        empty_value=empty_value
    )
    
    return design_net


# Example usage
if __name__ == "__main__":
    device = torch.device("cpu")  # or "cuda"
    
    # Create design network for time-based designs
    design_net_time = create_siqr_design_network(
        design_type="time"
    ).to(device)
    
    # Create SIQR DAD model
    siqr_model = SIQR_DAD_Model(
        design_net=design_net_time,
        T=4,  # 4 sequential experiments
        population_size=500.0,
        design_type="time"
    )
    
    # Test evaluation
    results = siqr_model.eval(n_trace=2)
    print("SIQR DAD evaluation results:")
    for result in results:
        print(f"Run {result['run_id']}: designs={result['designs']}, observations={result['observations']}")
        print(f"True parameters (beta, alpha, gamma, delta): {result['theta']}")
    
    # Example with intervention designs
    design_net_intervention = create_siqr_design_network(
        design_type="intervention"
    ).to(device)
    
    siqr_intervention_model = SIQR_DAD_Model(
        design_net=design_net_intervention,
        T=3,
        design_type="intervention"
    )
    
    print("\nSIQR Intervention DAD results:")
    results_intervention = siqr_intervention_model.eval(n_trace=1)
    for result in results_intervention:
        print(f"Intervention strengths: {result['designs']}")
        print(f"Observed cases: {result['observations']}")
