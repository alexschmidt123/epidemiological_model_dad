import os
import argparse
import torch
import pyro
from tqdm import trange
import mlflow
import mlflow.pytorch

from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from contrastive.mi import PriorContrastiveEstimationScoreGradient

# Import our new models
from seir_dad_model import SEIR_DAD_Model, create_seir_design_network
from siqr_dad_model import SIQR_DAD_Model, create_siqr_design_network


def train_seir_dad(
    seed,
    num_steps,
    num_inner_samples,
    num_outer_samples,
    lr,
    gamma,
    T,
    device,
    hidden_dim,
    encoding_dim,
    num_layers,
    population_size,
    mlflow_experiment_name,
):
    """Train SEIR DAD model"""
    
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)
    mlflow.set_experiment(mlflow_experiment_name)

    # Log parameters
    mlflow.log_param("model_type", "SEIR")
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("num_outer_samples", num_outer_samples)
    mlflow.log_param("population_size", population_size)

    # Create design network
    design_net = create_seir_design_network(
        design_dim=1,
        hidden_dim=hidden_dim,
        encoding_dim=encoding_dim,
        num_layers=num_layers
    ).to(device)

    # Create SEIR model
    seir_model = SEIR_DAD_Model(
        design_net=design_net,
        T=T,
        population_size=population_size
    )

    # Setup optimizer
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({
        "optimizer": optimizer,
        "optim_args": {"lr": lr, "betas": [0.9, 0.999], "weight_decay": 0},
        "gamma": gamma,
    })

    # Setup loss function
    pce_loss = PriorContrastiveEstimationScoreGradient(
        num_outer_samples, num_inner_samples
    )

    # Create OED object
    oed = OED(seir_model.model, scheduler, pce_loss)

    # Training loop
    loss_history = []
    t = trange(1, num_steps + 1, desc="Loss: 0.000")
    
    for i in t:
        try:
            loss = oed.step()
            loss_val = loss.item() if hasattr(loss, 'item') else loss
            t.set_description(f"Loss: {loss_val:.3f}")
            loss_history.append(loss_val)
            
            if i % 50 == 0:
                mlflow.log_metric("loss", oed.evaluate_loss())
            if i % 1000 == 0:
                scheduler.step()
                
        except Exception as e:
            print(f"Error at step {i}: {e}")
            break

    # Evaluate trained model
    try:
        runs_output = seir_model.eval(n_trace=5)
        print("SEIR DAD training completed successfully!")
        
        # Log final results
        mlflow.log_metric("final_loss", loss_history[-1] if loss_history else float('inf'))
        
        # Save model
        mlflow.pytorch.log_model(seir_model.cpu(), "model")
        
        return {
            "design_network": design_net.cpu(),
            "model": seir_model,
            "loss_history": loss_history,
            "runs_output": runs_output,
            "seed": seed
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {
            "loss_history": loss_history,
            "seed": seed
        }


def train_siqr_dad(
    seed,
    num_steps,
    num_inner_samples,
    num_outer_samples,
    lr,
    gamma,
    T,
    device,
    hidden_dim,
    encoding_dim,
    num_layers,
    population_size,
    design_type,
    mlflow_experiment_name,
):
    """Train SIQR DAD model"""
    
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)
    mlflow.set_experiment(mlflow_experiment_name)

    # Log parameters
    mlflow.log_param("model_type", "SIQR")
    mlflow.log_param("design_type", design_type)
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("num_outer_samples", num_outer_samples)
    mlflow.log_param("population_size", population_size)

    # Create design network
    design_net = create_siqr_design_network(
        design_dim=1,
        design_type=design_type,
        hidden_dim=hidden_dim,
        encoding_dim=encoding_dim,
        num_layers=num_layers
    ).to(device)

    # Create SIQR model
    siqr_model = SIQR_DAD_Model(
        design_net=design_net,
        T=T,
        population_size=population_size,
        design_type=design_type
    )

    # Setup optimizer
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({
        "optimizer": optimizer,
        "optim_args": {"lr": lr, "betas": [0.9, 0.999], "weight_decay": 0},
        "gamma": gamma,
    })

    # Setup loss function
    pce_loss = PriorContrastiveEstimationScoreGradient(
        num_outer_samples, num_inner_samples
    )

    # Create OED object
    oed = OED(siqr_model.model, scheduler, pce_loss)

    # Training loop
    loss_history = []
    t = trange(1, num_steps + 1, desc="Loss: 0.000")
    
    for i in t:
        try:
            loss = oed.step()
            loss_val = loss.item() if hasattr(loss, 'item') else loss
            t.set_description(f"Loss: {loss_val:.3f}")
            loss_history.append(loss_val)
            
            if i % 50 == 0:
                mlflow.log_metric("loss", oed.evaluate_loss())
            if i % 1000 == 0:
                scheduler.step()
                
        except Exception as e:
            print(f"Error at step {i}: {e}")
            break

    # Evaluate trained model
    try:
        runs_output = siqr_model.eval(n_trace=5)
        print("SIQR DAD training completed successfully!")
        
        # Log final results
        mlflow.log_metric("final_loss", loss_history[-1] if loss_history else float('inf'))
        
        # Save model
        mlflow.pytorch.log_model(siqr_model.cpu(), "model")
        
        return {
            "design_network": design_net.cpu(),
            "model": siqr_model,
            "loss_history": loss_history,
            "runs_output": runs_output,
            "seed": seed
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {
            "loss_history": loss_history,
            "seed": seed
        }


def main():
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design for SEIR/SIQR models"
    )
    
    # Model selection
    parser.add_argument("--model", choices=["seir", "siqr"], default="seir",
                       help="Choose between SEIR or SIQR model")
    parser.add_argument("--design-type", choices=["time", "intervention"], default="time",
                       help="Design type for SIQR model (time or intervention)")
    
    # Training parameters
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=2000, type=int)
    parser.add_argument("--num-inner-samples", default=10, type=int)
    parser.add_argument("--num-outer-samples", default=20, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    
    # Model parameters
    parser.add_argument("--num-experiments", default=4, type=int)
    parser.add_argument("--population-size", default=500.0, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    
    # Network architecture
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    
    # MLflow
    parser.add_argument("--mlflow-experiment-name", default="epidemic_dad", type=str)
    
    args = parser.parse_args()
    
    print(f"Training {args.model.upper()} DAD model...")
    print(f"Device: {args.device}")
    print(f"Number of experiments: {args.num_experiments}")
    print(f"Training steps: {args.num_steps}")
    
    if args.model == "seir":
        results = train_seir_dad(
            seed=args.seed,
            num_steps=args.num_steps,
            num_inner_samples=args.num_inner_samples,
            num_outer_samples=args.num_outer_samples,
            lr=args.lr,
            gamma=args.gamma,
            T=args.num_experiments,
            device=args.device,
            hidden_dim=args.hidden_dim,
            encoding_dim=args.encoding_dim,
            num_layers=args.num_layers,
            population_size=args.population_size,
            mlflow_experiment_name=args.mlflow_experiment_name,
        )
    elif args.model == "siqr":
        results = train_siqr_dad(
            seed=args.seed,
            num_steps=args.num_steps,
            num_inner_samples=args.num_inner_samples,
            num_outer_samples=args.num_outer_samples,
            lr=args.lr,
            gamma=args.gamma,
            T=args.num_experiments,
            device=args.device,
            hidden_dim=args.hidden_dim,
            encoding_dim=args.encoding_dim,
            num_layers=args.num_layers,
            population_size=args.population_size,
            design_type=args.design_type,
            mlflow_experiment_name=args.mlflow_experiment_name,
        )
    
    print("Training completed!")
    if "runs_output" in results:
        print("Sample evaluation results:")
        for run in results["runs_output"][:2]:  # Show first 2 runs
            print(f"  Run {run['run_id']}: designs={run['designs'][:3]}..., observations={run['observations'][:3]}...")


if __name__ == "__main__":
    main()