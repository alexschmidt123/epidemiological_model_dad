import argparse
from typing import Tuple

import mlflow
import mlflow.pytorch

import torch
from torch import nn
import pyro
import pyro.distributions as dist
from pyro.infer.util import torch_item
from pyro.poutine.util import prune_subsample_sites
from pyro.contrib.util import lexpand, rexpand
from tqdm import trange

from neural.modules import (
    SetEquivariantDesignNetwork,
    BatchDesignBaseline,
)
from oed.primitives import observation_sample, compute_design
from oed.design import OED
from contrastive.mi import (
    PriorContrastiveEstimationScoreGradient,
)
from experiment_tools.pyro_tools import auto_seed


def integrate_seir(beta: torch.Tensor, sigma_e: torch.Tensor, gamma: torch.Tensor,
                   S0: torch.Tensor, E0: torch.Tensor, I0: torch.Tensor, R0: torch.Tensor,
                   t: torch.Tensor, n_steps: int = 200) -> torch.Tensor:
    beta = beta.float(); sigma_e = sigma_e.float(); gamma = gamma.float()
    S = S0.float(); E = E0.float(); I = I0.float(); R = R0.float()
    t = t.float().clamp(min=0.0)
    n_steps = int(max(1, n_steps))
    dt = t / n_steps
    for _ in range(n_steps):
        dS = -beta * S * I
        dE = beta * S * I - sigma_e * E
        dI = sigma_e * E - gamma * I
        dR = gamma * I
        S = (S + dt * dS).clamp(min=0.0)
        E = (E + dt * dE).clamp(min=0.0)
        I = (I + dt * dI).clamp(min=0.0)
        R = (R + dt * dR).clamp(min=0.0)
        total = (S + E + I + R).clamp(min=1e-12)
        S = S / total; E = E / total; I = I / total; R = R / total
    return I


class EncoderNetwork(nn.Module):
    def __init__(self, design_dim, osbervation_dim, hidden_dim, encoding_dim, n_hidden_layers=2, activation=nn.Softplus):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.activation_layer = activation()
        self.input_layer = nn.Linear(design_dim + osbervation_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation()) for _ in range(n_hidden_layers - 1)]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y, **kwargs):
        inputs = torch.stack([xi, y], dim=-1)
        x = self.input_layer(inputs)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x


class EmitterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=2, activation=nn.Softplus):
        super().__init__()
        self.activation_layer = activation()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation()) for _ in range(n_hidden_layers - 1)]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, r):
        x = self.input_layer(r)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x


class SEIRDesignModel(nn.Module):
    def __init__(
        self,
        design_net: nn.Module,
        beta_loc: torch.Tensor = None,
        beta_scale: torch.Tensor = None,
        gamma_loc: torch.Tensor = None,
        gamma_scale: torch.Tensor = None,
        sigmae_loc: torch.Tensor = None,
        sigmae_scale: torch.Tensor = None,
        noise_scale: torch.Tensor = None,
        S0: float = 0.99,
        E0: float = 0.0,
        I0: float = 0.01,
        R0: float = 0.0,
        T: int = 3,
    ):
        super().__init__()
        self.design_net = design_net
        self.T = T
        self.beta_loc = beta_loc if beta_loc is not None else torch.tensor(-0.2)
        self.beta_scale = beta_scale if beta_scale is not None else torch.tensor(0.5)
        self.gamma_loc = gamma_loc if gamma_loc is not None else torch.tensor(-1.0)
        self.gamma_scale = gamma_scale if gamma_scale is not None else torch.tensor(0.5)
        self.sigmae_loc = sigmae_loc if sigmae_loc is not None else torch.tensor(-0.7)
        self.sigmae_scale = sigmae_scale if sigmae_scale is not None else torch.tensor(0.5)
        self.noise_scale = noise_scale if noise_scale is not None else torch.tensor(0.02)
        self.register_buffer("S0", torch.tensor(S0))
        self.register_buffer("E0", torch.tensor(E0))
        self.register_buffer("I0", torch.tensor(I0))
        self.register_buffer("R0", torch.tensor(R0))
        self.softplus = nn.Softplus()

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)
        beta = pyro.sample("beta", dist.LogNormal(self.beta_loc, self.beta_scale))
        gamma = pyro.sample("gamma", dist.LogNormal(self.gamma_loc, self.gamma_scale))
        sigma_e = pyro.sample("sigma_e", dist.LogNormal(self.sigmae_loc, self.sigmae_scale))
        y_outcomes = []
        xi_designs = []
        for t in range(self.T):
            xi = compute_design(f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes)))
            xi = self.softplus(xi.squeeze(-1))
            I_t = integrate_seir(beta, sigma_e, gamma, self.S0, self.E0, self.I0, self.R0, xi)
            y = observation_sample(f"y{t + 1}", dist.Normal(loc=I_t.clamp(min=0.0, max=1.0), scale=self.noise_scale))
            y_outcomes.append(y)
            xi_designs.append(xi)
        return y_outcomes

    def rollout(self, n_rollout: int, grid: torch.Tensor):
        """Vectorized rollout to compute log probs on a parameter grid like death process."""
        self.design_net.eval()
        grid_size = grid.shape[0]

        def vectorized_model():
            with pyro.plate("vectorization", n_rollout):
                return self.model()

        with torch.no_grad():
            trace = pyro.poutine.trace(vectorized_model).get_trace()
            trace = prune_subsample_sites(trace)
            trace.compute_log_prob()

            data = {
                name: lexpand(node["value"], grid_size)
                for name, node in trace.nodes.items()
                if node.get("subtype") in ["observation_sample", "design_sample"]
            }
            # grid expected shape: (grid_size, 3) for beta, gamma, sigma_e
            data["beta"] = rexpand(grid[..., 0], n_rollout)
            data["gamma"] = rexpand(grid[..., 1], n_rollout)
            data["sigma_e"] = rexpand(grid[..., 2], n_rollout)

            def conditional_model():
                with pyro.plate_stack("vectorization", (grid_size, n_rollout)):
                    pyro.condition(self.model, data=data)()

            condition_trace = pyro.poutine.trace(conditional_model).get_trace()
            condition_trace = prune_subsample_sites(condition_trace)
            condition_trace.compute_log_prob()
        return condition_trace


def single_run(
    seed: int,
    num_steps: int,
    num_inner_samples: int,
    num_outer_samples: int,
    lr: float,
    gamma_sched: float,
    T: int,
    device: str,
    hidden_dim: int,
    encoding_dim: int,
    num_layers: int,
    arch: str,
    noise_scale: float,
) -> Tuple[SEIRDesignModel, list]:
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)

    if arch == "static":
        design_net = BatchDesignBaseline(T, 1).to(device)
    else:
        encoder = EncoderNetwork(1, 1, hidden_dim, encoding_dim, n_hidden_layers=num_layers)
        emitter = EmitterNetwork(encoding_dim, hidden_dim, 1, n_hidden_layers=num_layers)
        if arch == "sum":
            design_net = SetEquivariantDesignNetwork(encoder, emitter, empty_value=torch.ones(1)).to(device)
        else:
            raise ValueError(f"Unexpected architecture specification: '{arch}'.")

    model = SEIRDesignModel(
        design_net=design_net,
        noise_scale=torch.tensor(noise_scale, device=device),
        T=T,
    ).to(device)

    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": lr, "betas": [0.9, 0.999], "weight_decay": 0.0},
            "gamma": gamma_sched,
        }
    )
    pce_loss = PriorContrastiveEstimationScoreGradient(num_outer_samples, num_inner_samples)
    oed = OED(model.model, scheduler, pce_loss)

    loss_history = []
    tbar = trange(1, num_steps + 1, desc="Loss: 0.000 ")
    for i in tbar:
        loss = oed.step()
        loss = torch_item(loss)
        tbar.set_description("Loss: {:.3f} ".format(loss))
        loss_history.append(loss)
        if i % 1000 == 0:
            scheduler.step()

    return model, loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Adaptive Design example: SEIR ODE.")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=5000, type=int)
    parser.add_argument("--num-inner-samples", default=10, type=int)
    parser.add_argument("--num-outer-samples", default=20, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--num-experiments", default=3, type=int)  # == T
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--hidden-dim", default=64, type=int)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--num-layers", default=1, type=int)
    parser.add_argument("--arch", default="sum", type=str, choices=["static", "sum"]) 
    parser.add_argument("--noise-scale", default=0.02, type=float)
    parser.add_argument("--mlflow-experiment-name", default="seir process", type=str)
    args = parser.parse_args()

    pyro.clear_param_store()
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("num_experiments", args.num_experiments)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("hidden_dim", args.hidden_dim)
        mlflow.log_param("encoding_dim", args.encoding_dim)
        mlflow.log_param("num_layers", args.num_layers)
        mlflow.log_param("gamma", args.gamma)
        mlflow.log_param("arch", args.arch)
        mlflow.log_param("num_steps", args.num_steps)
        mlflow.log_param("num_inner_samples", args.num_inner_samples)
        mlflow.log_param("num_outer_samples", args.num_outer_samples)
        mlflow.log_param("noise_scale", args.noise_scale)

        model, loss_history = single_run(
            seed=args.seed,
            num_steps=args.num_steps,
            num_inner_samples=args.num_inner_samples,
            num_outer_samples=args.num_outer_samples,
            lr=args.lr,
            gamma_sched=args.gamma,
            T=args.num_experiments,
            device=args.device,
            hidden_dim=args.hidden_dim,
            encoding_dim=args.encoding_dim,
            num_layers=args.num_layers,
            arch=args.arch,
            noise_scale=args.noise_scale,
        )

        # Log simple convergence metric
        if len(loss_history) >= 52:
            mlflow.log_metric(
                "loss_diff50", sum(loss_history[-51:-1]) / max(sum(loss_history[0:50]), 1e-8) - 1
            )

        # Log model artifact for evaluation
        mlflow.pytorch.log_model(model.cpu(), "model")


