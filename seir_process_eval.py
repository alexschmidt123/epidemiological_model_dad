import argparse
import math
import os

import torch
import pyro
import pandas as pd

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from experiment_tools.output_utils import get_mlflow_meta
from experiment_tools.pyro_tools import auto_seed
from pyro.poutine.util import prune_subsample_sites
from pyro.contrib.util import lexpand, rexpand


def rollout_model(model, n_rollout: int, grid: torch.Tensor, fixed_xis=None):
    """Standalone rollout using a Pyro model callable, without requiring a method on the class."""
    grid_size = grid.shape[0]

    def vectorized_model():
        with pyro.plate("vectorization", n_rollout):
            return model.model()

    with torch.no_grad():
        trace = pyro.poutine.trace(vectorized_model).get_trace()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()

        data = {
            name: lexpand(node["value"], grid_size)
            for name, node in trace.nodes.items()
            if node.get("subtype") in ["observation_sample", "design_sample"]
        }
        # Optionally override designs with fixed times
        if fixed_xis is not None:
            for t_idx, xi_val in enumerate(fixed_xis, start=1):
                key = f"xi{t_idx}"
                if key in data:
                    # broadcast xi_val to the shape of data[key]
                    data[key] = torch.ones_like(data[key]) * float(xi_val)
        data["beta"] = rexpand(grid[..., 0], n_rollout)
        data["gamma"] = rexpand(grid[..., 1], n_rollout)
        data["sigma_e"] = rexpand(grid[..., 2], n_rollout)

        def conditional_model():
            with pyro.plate_stack("vectorization", (grid_size, n_rollout)):
                pyro.condition(model.model, data=data)()

        condition_trace = pyro.poutine.trace(conditional_model).get_trace()
        condition_trace = prune_subsample_sites(condition_trace)
        condition_trace.compute_log_prob()
    return condition_trace


def evaluate_policy(
    experiment_id,
    run_id=None,
    seed=-1,
    n_rollout=1000,  # number of rollouts
    device="cpu",
):

    pyro.clear_param_store()
    seed = auto_seed(seed)
    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")
    from_source = []
    client = MlflowClient()
    resolved_experiment_id = experiment_id
    if run_id is None and experiment_id is None:
        raise ValueError("Provide either --run-id or --experiment-id to evaluate.")
    if run_id:
        # Resolve the correct experiment_id from the run
        try:
            resolved_experiment_id = client.get_run(run_id).info.experiment_id
        except Exception:
            resolved_experiment_id = experiment_id
        experiment_run_ids = [run_id]
        from_source = [False]
    else:
        filter_string = ""
        meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)
        meta = [m for m in meta if "eval_seed" not in m.data.params.keys()]
        experiment_run_ids = [run.info.run_id for run in meta]
        from_source = [
            True if "from_source" in m.data.params.keys() else False for m in meta
        ]
    print(experiment_run_ids)

    for i, run_id in enumerate(experiment_run_ids):
        if from_source[i]:
            client = MlflowClient()
            metric = client.get_metric_history(run_id, "information_gain")
            igs = [m.value for m in metric]
            n_rollout = len(igs)
            num_experiments = int(client.get_run(run_id).data.params["num_experiments"])
            information_gain = torch.tensor(igs)
        else:
            model_location = f"mlruns/{resolved_experiment_id}/{run_id}/artifacts/model"
            seir_model = mlflow.pytorch.load_model(model_location, map_location=device)
            num_experiments = getattr(seir_model, "T", 0) or int(client.get_run(run_id).data.params.get("num_experiments", 0))

            # Compute EIG over a parameter grid for THIS run (model)
            with torch.no_grad():
                # LogNormal prior params taken from training defaults
                beta_loc, beta_scale = torch.tensor(-0.2), torch.tensor(0.5)
                gamma_loc, gamma_scale = torch.tensor(-1.0), torch.tensor(0.5)
                sigmae_loc, sigmae_scale = torch.tensor(-0.7), torch.tensor(0.5)
                # Build grid by sampling from prior (quasi-grid)
                grid_n = 500
                beta_grid = torch.distributions.LogNormal(beta_loc, beta_scale).sample((grid_n,))
                gamma_grid = torch.distributions.LogNormal(gamma_loc, gamma_scale).sample((grid_n,))
                sigmae_grid = torch.distributions.LogNormal(sigmae_loc, sigmae_scale).sample((grid_n,))
                grid = torch.stack([beta_grid, gamma_grid, sigmae_grid], dim=-1).to(device)

                # One rollout with vectorization to get prior and posterior log probs
                data_run = rollout_model(seir_model, n_rollout, grid)
                prior_log_prob = (
                    torch.distributions.LogNormal(beta_loc, beta_scale).log_prob(grid[:, 0])
                    + torch.distributions.LogNormal(gamma_loc, gamma_scale).log_prob(grid[:, 1])
                    + torch.distributions.LogNormal(sigmae_loc, sigmae_scale).log_prob(grid[:, 2])
                )
                mesh_density = math.exp(-prior_log_prob.logsumexp(0).item())
                posterior_log_prob_run = sum(
                    node["log_prob"]
                    for node in data_run.nodes.values()
                    if node["type"] == "sample" and node.get("subtype") != "design_sample"
                )
                posterior_log_prob_run = (
                    posterior_log_prob_run - posterior_log_prob_run.logsumexp(0) - math.log(mesh_density)
                )
                posterior_entropy_run = (mesh_density * posterior_log_prob_run.exp() * (-posterior_log_prob_run)).sum(0)
                prior_entropy = (mesh_density * prior_log_prob.exp() * (-prior_log_prob)).sum(0)
                information_gain = prior_entropy - posterior_entropy_run

        print(information_gain.mean(), information_gain.std() / math.sqrt(n_rollout))
        res = pd.DataFrame(
            {
                "EIG_mean": information_gain.mean().item(),
                "EIG_se": (information_gain.std() / math.sqrt(n_rollout)).item(),
            },
            index=[num_experiments],
        )

        res.to_csv("mlflow_outputs/seir_eval.csv")
        with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
            mlflow.log_param("eval_seed", seed)
            mlflow.log_artifact(
                "mlflow_outputs/seir_eval.csv", artifact_path="evaluation"
            )
            mlflow.log_metric("eval_MI", information_gain.mean().item())

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design: SEIR ODE Evaluation."
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--experiment-id", default=None, type=str)
    parser.add_argument("--run-id", default=None, type=str)
    parser.add_argument("--num-rollouts", default=10000, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    evaluate_policy(
        seed=args.seed,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        device=args.device,
        n_rollout=args.num_rollouts,
    )


