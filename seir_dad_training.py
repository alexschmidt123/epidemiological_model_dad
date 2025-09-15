"""
Training Script for SEIR DAD Model
Implements the training loop with Prior Contrastive Estimation
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import pyro
import pyro.optim
from pyro.infer.util import torch_item
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import our modules
from seir_ode_model import SEIR_DAD_Model
from seir_dad_networks import create_seir_dad_network, BatchDesignBaseline
from seir_data_generation import SEIR_DataGenerator

# Import DAD components (assuming these exist from the death process code)
import sys
sys.path.append('..')  # Add parent directory
from oed.design import OED
from contrastive.mi import PriorContrastiveEstimationScoreGradient
from experiment_tools.pyro_tools import auto_seed


class SEIR_DAD_Trainer:
    """Trainer class for SEIR DAD model"""
    
    def __init__(
        self,
        config=None,
        device='cpu'
    ):
        """
        Initialize trainer
        
        Parameters:
        -----------
        config : dict
            Training configuration
        device : str or torch.device
            Device for training
        """
        self.device = device
        
        # Default configuration
        default_config = {
            # Model parameters
            'N': 500,
            'T': 4,
            'noise_scale': 0.1,
            'noise_type': 'proportional',
            
            # Network architecture
            'hidden_dim': 128,
            'encoding_dim': 16,
            'n_hidden_layers': 2,
            'activation': 'softplus',
            'aggregation': 'sum',
            
            # Training parameters
            'num_steps': 2000,
            'lr': 0.001,
            'gamma': 0.95,  # LR decay
            'lr_schedule_step': 500,  # Decay LR every N steps
            
            # Loss parameters (Prior Contrastive Estimation)
            'num_outer_samples': 20,
            'num_inner_samples': 10,
            
            # Other
            'seed': -1,
            'save_dir': 'models/seir/',
            'experiment_name': None
        }
        
        # Merge with provided config
        if config is None:
            self.config = default_config
        else:
            self.config = {**default_config, **config}
        
        # Set seed
        self.seed = auto_seed(self.config['seed'])
        
        # Create save directory
        if self.config['experiment_name'] is None:
            self.config['experiment_name'] = f"seir_dad_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.save_dir = os.path.join(
            self.config['save_dir'],
            self.config['experiment_name']
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Initialize model and networks
        self._build_model()
    
    def _build_model(self):
        """Build the DAD model and networks"""
        
        # Create design network
        self.design_net = create_seir_dad_network(
            hidden_dim=self.config['hidden_dim'],
            encoding_dim=self.config['encoding_dim'],
            n_hidden_layers=self.config['n_hidden_layers'],
            activation=self.config['activation'],
            aggregation=self.config['aggregation']
        ).to(self.device)
        
        # Create SEIR DAD model
        self.model = SEIR_DAD_Model(
            design_net=self.design_net,
            N=self.config['N'],
            T=self.config['T'],
            noise_type=self.config['noise_type'],
            noise_scale=self.config['noise_scale'],
            device=self.device
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam
        self.scheduler = pyro.optim.ExponentialLR({
            'optimizer': optimizer,
            'optim_args': {
                'lr': self.config['lr'],
                'betas': [0.9, 0.999],
                'weight_decay': 0
            },
            'gamma': self.config['gamma']
        })
        
        # Setup loss (Prior Contrastive Estimation with Score Gradient)
        self.pce_loss = PriorContrastiveEstimationScoreGradient(
            num_outer_samples=self.config['num_outer_samples'],
            num_inner_samples=self.config['num_inner_samples']
        )
        
        # Create OED object
        self.oed = OED(
            self.model.model,
            self.scheduler,
            self.pce_loss
        )
    
    def train(self):
        """
        Train the DAD model
        
        Returns:
        --------
        history : dict
            Training history
        """
        print("=" * 60)
        print(f"Training SEIR DAD Model")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Experiment: {self.config['experiment_name']}")
        print(f"Save directory: {self.save_dir}")
        print("-" * 40)
        
        # Training history
        history = {
            'loss': [],
            'eval_loss': [],
            'designs': [],
            'step': []
        }
        
        # Progress bar
        t = trange(1, self.config['num_steps'] + 1, desc="Loss: 0.000")
        
        for step in t:
            # Training step
            loss = self.oed.step()
            loss_val = torch_item(loss)
            
            # Update progress bar
            t.set_description(f"Loss: {loss_val:.4f}")
            
            # Store history
            history['loss'].append(loss_val)
            history['step'].append(step)
            
            # Periodic evaluation
            if step % 100 == 0:
                eval_loss = self.oed.evaluate_loss()
                history['eval_loss'].append(eval_loss)
                
                # Extract current designs
                with torch.no_grad():
                    trace = pyro.poutine.trace(self.model.model).get_trace()
                    designs = []
                    for t_idx in range(self.config['T']):
                        xi = trace.nodes[f"xi{t_idx + 1}"]["value"].item()
                        designs.append(xi)
                    history['designs'].append(designs)
            
            # Learning rate schedule
            if step % self.config['lr_schedule_step'] == 0:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config['lr'] * (self.config['gamma'] ** (step // self.config['lr_schedule_step']))
                print(f"\nStep {step}: LR decreased to {current_lr:.6f}")
            
            # Periodic saving
            if step % 500 == 0:
                self.save_checkpoint(step, history)
        
        print("\nTraining completed!")
        print(f"Final loss: {history['loss'][-1]:.4f}")
        
        # Save final model
        self.save_checkpoint('final', history)
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def save_checkpoint(self, step, history):
        """Save model checkpoint"""
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'design_net_state_dict': self.design_net.state_dict(),
            'optimizer_state_dict': self.scheduler.optim.state_dict() if hasattr(self.scheduler, 'optim') else None,
            'config': self.config,
            'history': history,
            'seed': self.seed
        }
        
        path = os.path.join(self.save_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.design_net.load_state_dict(checkpoint['design_net_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and hasattr(self.scheduler, 'optim'):
            self.scheduler.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded: {path}")
        return checkpoint['history']
    
    def plot_training_history(self, history):
        """Plot and save training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curve
        ax = axes[0, 0]
        ax.plot(history['step'], history['loss'], alpha=0.7, label='Training Loss')
        if history['eval_loss']:
            eval_steps = history['step'][::100][:len(history['eval_loss'])]
            ax.plot(eval_steps, history['eval_loss'], 'r-', alpha=0.7, label='Eval Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Smoothed loss
        ax = axes[0, 1]
        window = 50
        if len(history['loss']) > window:
            smoothed = np.convolve(history['loss'], np.ones(window)/window, mode='valid')
            ax.plot(history['step'][window-1:], smoothed, alpha=0.7)
        else:
            ax.plot(history['step'], history['loss'], alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Smoothed Loss')
        ax.set_title(f'Smoothed Training Loss (window={window})')
        ax.grid(True, alpha=0.3)
        
        # Design evolution
        if history['designs']:
            ax = axes[1, 0]
            designs_array = np.array(history['designs'])
            for t in range(self.config['T']):
                ax.plot(designs_array[:, t], label=f'Design {t+1}', alpha=0.7)
            ax.set_xlabel('Evaluation')
            ax.set_ylabel('Design Time (days)')
            ax.set_title('Design Evolution During Training')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Final designs
            ax = axes[1, 1]
            final_designs = designs_array[-1]
            ax.bar(range(1, len(final_designs)+1), final_designs, alpha=0.7)
            ax.set_xlabel('Experiment')
            ax.set_ylabel('Observation Time (days)')
            ax.set_title('Final Learned Designs')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved: {fig_path}")
    
    def evaluate(self, n_scenarios=10):
        """
        Evaluate trained model
        
        Parameters:
        -----------
        n_scenarios : int
            Number of scenarios to evaluate
        
        Returns:
        --------
        results : list
            Evaluation results
        """
        print("\nEvaluating trained model...")
        
        self.model.eval()
        self.design_net.eval()
        
        results = self.model.evaluate(n_trace=n_scenarios)
        
        # Print summary
        print("\nEvaluation Summary:")
        print("-" * 40)
        
        for i, result in enumerate(results[:3]):  # Show first 3
            print(f"\nScenario {i+1}:")
            print(f"  Parameters: β={result['theta'][0]:.3f}, σ={result['theta'][1]:.3f}, γ={result['theta'][2]:.3f}")
            print(f"  Designs: {[f'{d:.1f}' for d in result['designs']]}")
            print(f"  Observations: {[f'{o:.1f}' for o in result['observations']]}")
        
        return results


def train_baseline_models(config, device='cpu'):
    """
    Train baseline models for comparison
    
    Parameters:
    -----------
    config : dict
        Training configuration
    device : str or torch.device
        Device for training
    
    Returns:
    --------
    baselines : dict
        Trained baseline models
    """
    print("\n" + "=" * 60)
    print("Training Baseline Models")
    print("=" * 60)
    
    baselines = {}
    
    # Static baseline (learns fixed designs)
    print("\nTraining Static Baseline...")
    
    static_net = BatchDesignBaseline(T=config['T']).to(device)
    static_model = SEIR_DAD_Model(
        design_net=static_net,
        N=config['N'],
        T=config['T'],
        device=device
    )
    
    # Setup training for static baseline
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({
        'optimizer': optimizer,
        'optim_args': {'lr': config['lr']},
        'gamma': config['gamma']
    })
    
    pce_loss = PriorContrastiveEstimationScoreGradient(
        num_outer_samples=config['num_outer_samples'],
        num_inner_samples=config['num_inner_samples']
    )
    
    oed = OED(static_model.model, scheduler, pce_loss)
    
    # Train static baseline
    static_losses = []
    for step in trange(config['num_steps'] // 2, desc="Static Baseline"):
        loss = oed.step()
        static_losses.append(torch_item(loss))
        if step % 500 == 0:
            scheduler.step()
    
    baselines['static'] = {
        'model': static_model,
        'network': static_net,
        'losses': static_losses
    }
    
    print(f"Static baseline final loss: {static_losses[-1]:.4f}")
    
    return baselines


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SEIR DAD Model")
    parser.add_argument('--num-steps', type=int, default=1500, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--encoding-dim', type=int, default=16, help='Encoding dimension')
    parser.add_argument('--num-outer', type=int, default=20, help='Number of outer samples for PCE')
    parser.add_argument('--num-inner', type=int, default=10, help='Number of inner samples for PCE')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'num_steps': args.num_steps,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'encoding_dim': args.encoding_dim,
        'num_outer_samples': args.num_outer,
        'num_inner_samples': args.num_inner,
        'seed': args.seed,
        'experiment_name': args.name
    }
    
    # Create trainer
    trainer = SEIR_DAD_Trainer(config=config, device=args.device)
    
    # Train model
    history = trainer.train()
    
    # Evaluate
    results = trainer.evaluate(n_scenarios=10)
    
    # Train baselines for comparison
    baselines = train_baseline_models(trainer.config, device=args.device)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)