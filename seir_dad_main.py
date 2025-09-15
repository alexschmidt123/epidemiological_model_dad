"""
Main script for SEIR Deep Adaptive Design (DAD)
Complete pipeline: data generation, training, testing, and validation
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import all our modules
from seir_ode_model import SEIR_DAD_Model
from seir_data_generation import generate_train_test_data, SEIR_DataGenerator
from seir_dad_networks import create_seir_dad_network, BatchDesignBaseline
from seir_dad_training import SEIR_DAD_Trainer, train_baseline_models
from seir_dad_validation import SEIR_Validator, validate_trained_model


def setup_directories():
    """Create necessary directories for the project"""
    dirs = [
        'data/seir',
        'models/seir',
        'results/seir_validation',
        'logs',
        'figures'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Directories created")


def run_complete_pipeline(args):
    """Run the complete DAD pipeline"""
    
    print("\n" + "="*70)
    print(" SEIR DEEP ADAPTIVE DESIGN - COMPLETE PIPELINE ")
    print("="*70)
    
    # Setup
    device = torch.device(args.device)
    print(f"\n📍 Using device: {device}")
    
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed: {args.seed}")
    
    # Create experiment name
    if args.name is None:
        args.name = f"seir_dad_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"📝 Experiment: {args.name}")
    
    # Configuration
    config = {
        # Model parameters
        'N': args.population,
        'T': args.num_experiments,
        'noise_scale': args.noise_scale,
        'noise_type': 'proportional',
        
        # Network architecture
        'hidden_dim': args.hidden_dim,
        'encoding_dim': args.encoding_dim,
        'n_hidden_layers': args.num_layers,
        'activation': 'softplus',
        'aggregation': 'sum',
        
        # Training parameters
        'num_steps': args.num_steps,
        'lr': args.lr,
        'gamma': 0.95,
        'lr_schedule_step': 500,
        
        # Loss parameters
        'num_outer_samples': args.num_outer,
        'num_inner_samples': args.num_inner,
        
        # Other
        'seed': args.seed,
        'experiment_name': args.name
    }
    
    # Save configuration
    config_path = f'results/{args.name}_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuration saved to {config_path}")
    
    # ============================================================
    # STEP 1: DATA GENERATION
    # ============================================================
    if args.generate_data or args.pipeline == 'all':
        print("\n" + "-"*60)
        print("STEP 1: DATA GENERATION")
        print("-"*60)
        
        train_data, test_data = generate_train_test_data(
            n_train=args.n_train,
            n_test=args.n_test,
            T=args.num_experiments,
            save_dir='data/seir/',
            device=device
        )
        
        print(f"✓ Generated {args.n_train} training samples")
        print(f"✓ Generated {args.n_test} test samples")
    
    # ============================================================
    # STEP 2: TRAINING
    # ============================================================
    if args.train or args.pipeline == 'all':
        print("\n" + "-"*60)
        print("STEP 2: TRAINING DAD MODEL")
        print("-"*60)
        
        # Create trainer
        trainer = SEIR_DAD_Trainer(config=config, device=device)
        
        # Train model
        print("\n🚀 Starting training...")
        history = trainer.train()
        
        # Save training results
        results_dir = f'results/{args.name}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Evaluate on test scenarios
        print("\n📊 Evaluating trained model...")
        eval_results = trainer.evaluate(n_scenarios=10)
        
        # Save evaluation results
        eval_path = os.path.join(results_dir, 'evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"✓ Training complete! Final loss: {history['loss'][-1]:.4f}")
        print(f"✓ Model saved to: {trainer.save_dir}")
        
        # Train baselines for comparison
        if args.train_baselines:
            print("\n🎯 Training baseline models...")
            baselines = train_baseline_models(config, device=device)
            print("✓ Baselines trained")
    
    # ============================================================
    # STEP 3: VALIDATION WITH TRUE IG
    # ============================================================
    if args.validate or args.pipeline == 'all':
        print("\n" + "-"*60)
        print("STEP 3: VALIDATION WITH TRUE INFORMATION GAIN")
        print("-"*60)
        
        if args.model_path:
            # Validate specific model
            results = validate_trained_model(
                model_path=args.model_path,
                device=device
            )
        else:
            # Validate the just-trained model
            if 'trainer' in locals():
                validator = SEIR_Validator(
                    model=trainer.model,
                    n_monte_carlo=args.n_mc_validation,
                    device=device
                )
                results = validator.validate_all(save_results=True)
            else:
                print("❌ No model to validate. Train a model first or provide --model-path")
                return
        
        print("\n✓ Validation complete!")
    
    # ============================================================
    # STEP 4: COMPREHENSIVE ANALYSIS
    # ============================================================
    if args.analyze or args.pipeline == 'all':
        print("\n" + "-"*60)
        print("STEP 4: COMPREHENSIVE ANALYSIS")
        print("-"*60)
        
        if 'trainer' in locals() and 'results' in locals():
            analyze_results(trainer, results, args.name)
        else:
            print("❌ Need both training and validation results for analysis")
    
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE! ")
    print("="*70)
    print(f"\n📁 All results saved in: results/{args.name}/")


def analyze_results(trainer, validation_results, experiment_name):
    """Perform comprehensive analysis of results"""
    
    print("\n📈 Performing comprehensive analysis...")
    
    # Create analysis directory
    analysis_dir = f'results/{experiment_name}/analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Design Pattern Analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract multiple design trajectories
    n_samples = 20
    all_designs = []
    
    trainer.model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            trace = pyro.poutine.trace(trainer.model.model).get_trace()
            designs = []
            for t in range(trainer.config['T']):
                xi = trace.nodes[f"xi{t + 1}"]["value"].item()
                designs.append(xi)
            all_designs.append(designs)
    
    all_designs = np.array(all_designs)
    
    # Plot design distribution
    ax = axes[0, 0]
    for i in range(trainer.config['T']):
        data = all_designs[:, i]
        parts = ax.violinplot([data], positions=[i+1], widths=0.7, 
                              showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)
    
    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Observation Time (days)')
    ax.set_title('Distribution of Learned Designs')
    ax.grid(True, alpha=0.3)
    
    # Plot design correlations
    ax = axes[0, 1]
    if trainer.config['T'] > 1:
        correlation_matrix = np.corrcoef(all_designs.T)
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(trainer.config['T']))
        ax.set_yticks(range(trainer.config['T']))
        ax.set_xticklabels([f'T{i+1}' for i in range(trainer.config['T'])])
        ax.set_yticklabels([f'T{i+1}' for i in range(trainer.config['T'])])
        ax.set_title('Design Correlation Matrix')
        plt.colorbar(im, ax=ax)
    
    # Information Gain Comparison
    ax = axes[1, 0]
    methods = list(validation_results.keys())
    igs = [validation_results[m]['ig'] for m in methods]
    
    # Sort by IG
    sorted_pairs = sorted(zip(methods, igs), key=lambda x: x[1], reverse=True)
    methods, igs = zip(*sorted_pairs)
    
    colors = ['green' if m == 'DAD' else 'skyblue' for m in methods]
    bars = ax.barh(range(len(methods)), igs, color=colors, alpha=0.8)
    
    # Highlight DAD
    dad_idx = methods.index('DAD')
    bars[dad_idx].set_edgecolor('darkgreen')
    bars[dad_idx].set_linewidth(2)
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel('Information Gain')
    ax.set_title('Information Gain Ranking')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage improvement text
    dad_ig = validation_results['DAD']['ig']
    for i, (method, ig) in enumerate(zip(methods, igs)):
        if method != 'DAD':
            improvement = (dad_ig - ig) / ig * 100 if ig > 0 else 0
            color = 'green' if improvement > 0 else 'red'
            ax.text(ig, i, f' {improvement:+.1f}%', va='center', 
                   fontsize=8, color=color)
    
    # Epidemic trajectory with designs
    ax = axes[1, 1]
    
    # Sample parameters and solve ODE
    theta = trainer.model.theta_prior.sample().exp().cpu().numpy()
    t_range = np.linspace(0, 50, 500)
    from seir_ode_model import SEIR_ODE
    solution = SEIR_ODE.solve(theta, t_range, trainer.model.y0, trainer.model.N)
    
    # Plot infected trajectory
    ax.plot(t_range, solution[:, 2], 'b-', label='Infected', linewidth=2)
    ax.fill_between(t_range, 0, solution[:, 2], alpha=0.3)
    
    # Mark DAD designs
    dad_designs = validation_results['DAD']['designs']
    for i, xi in enumerate(dad_designs):
        idx = np.argmin(np.abs(t_range - xi))
        ax.scatter(xi, solution[idx, 2], color='red', s=100, zorder=5)
        ax.axvline(xi, color='red', linestyle='--', alpha=0.5)
        ax.text(xi, solution[idx, 2], f'  T{i+1}', fontsize=9)
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Infected Count')
    ax.set_title('Example Trajectory with DAD Designs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'SEIR DAD Analysis - {experiment_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(analysis_dir, 'comprehensive_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Analysis plots saved to {fig_path}")
    
    # Generate summary report
    generate_summary_report(trainer, validation_results, analysis_dir)


def generate_summary_report(trainer, validation_results, save_dir):
    """Generate a text summary report"""
    
    report_path = os.path.join(save_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" SEIR DEEP ADAPTIVE DESIGN - SUMMARY REPORT \n")
        f.write("="*70 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Population size: {trainer.config['N']}\n")
        f.write(f"Number of experiments: {trainer.config['T']}\n")
        f.write(f"Noise scale: {trainer.config['noise_scale']}\n")
        f.write(f"Network architecture: {trainer.config['hidden_dim']}-{trainer.config['encoding_dim']}\n")
        f.write(f"Training steps: {trainer.config['num_steps']}\n")
        f.write(f"Learning rate: {trainer.config['lr']}\n\n")
        
        # Results
        f.write("VALIDATION RESULTS\n")
        f.write("-"*40 + "\n")
        
        # Sort by IG
        sorted_results = sorted(validation_results.items(), 
                               key=lambda x: x[1]['ig'], reverse=True)
        
        f.write(f"{'Method':<15} {'IG':<10} {'Designs'}\n")
        f.write("-"*40 + "\n")
        
        for method, res in sorted_results:
            designs_str = ', '.join([f"{d:.1f}" for d in res['designs']])
            f.write(f"{method:<15} {res['ig']:<10.4f} [{designs_str}]\n")
        
        # Performance summary
        f.write("\nPERFORMANCE SUMMARY\n")
        f.write("-"*40 + "\n")
        
        dad_ig = validation_results['DAD']['ig']
        baseline_igs = [res['ig'] for name, res in validation_results.items() 
                        if name != 'DAD']
        
        if baseline_igs:
            best_baseline = max(baseline_igs)
            worst_baseline = min(baseline_igs)
            mean_baseline = np.mean(baseline_igs)
            
            f.write(f"DAD Information Gain: {dad_ig:.4f}\n")
            f.write(f"Best Baseline IG: {best_baseline:.4f}\n")
            f.write(f"Mean Baseline IG: {mean_baseline:.4f}\n")
            f.write(f"Worst Baseline IG: {worst_baseline:.4f}\n\n")
            
            improvement_best = (dad_ig - best_baseline) / best_baseline * 100
            improvement_mean = (dad_ig - mean_baseline) / mean_baseline * 100
            
            f.write(f"Improvement over best baseline: {improvement_best:+.1f}%\n")
            f.write(f"Improvement over mean baseline: {improvement_mean:+.1f}%\n")
            
            if improvement_best > 0:
                f.write("\n✓ DAD OUTPERFORMS ALL BASELINES\n")
            else:
                f.write("\n✗ DAD underperforms best baseline\n")
    
    print(f"✓ Summary report saved to {report_path}")


def quick_test():
    """Quick test to verify everything works"""
    
    print("\n" + "="*60)
    print(" QUICK TEST MODE ")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Mini configuration for testing
    config = {
        'N': 500,
        'T': 3,
        'noise_scale': 0.1,
        'hidden_dim': 32,
        'encoding_dim': 8,
        'n_hidden_layers': 1,
        'num_steps': 100,
        'lr': 0.001,
        'num_outer_samples': 5,
        'num_inner_samples': 5,
        'experiment_name': 'quick_test'
    }
    
    print("\n1. Testing data generation...")
    generator = SEIR_DataGenerator(device=device)
    data = generator.generate_dataset(n_samples=10, T=3)
    print(f"✓ Generated {len(data['params'])} samples")
    
    print("\n2. Testing network creation...")
    design_net = create_seir_dad_network(
        hidden_dim=config['hidden_dim'],
        encoding_dim=config['encoding_dim']
    )
    print("✓ Network created")
    
    print("\n3. Testing model...")
    model = SEIR_DAD_Model(
        design_net=design_net,
        N=config['N'],
        T=config['T'],
        device=device
    )
    print("✓ Model created")
    
    print("\n4. Testing training (mini)...")
    trainer = SEIR_DAD_Trainer(config=config, device=device)
    
    # Just a few steps
    from tqdm import trange
    for _ in trange(10, desc="Quick training"):
        loss = trainer.oed.step()
    print("✓ Training works")
    
    print("\n5. Testing validation...")
    validator = SEIR_Validator(
        model=trainer.model,
        n_monte_carlo=10,
        device=device
    )
    
    # Just test IG calculation
    test_designs = np.array([5.0, 10.0, 15.0])
    ig, ig_std = validator.ig_calculator.compute_information_gain(test_designs)
    print(f"✓ IG calculation works: {ig:.4f} ± {ig_std:.4f}")
    
    print("\n✅ All tests passed! System is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEIR Deep Adaptive Design - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --pipeline all
  
  # Just train a model
  python main.py --train --num-steps 1000
  
  # Validate existing model
  python main.py --validate --model-path models/seir/checkpoint_final.pt
  
  # Quick test
  python main.py --test
        """
    )
    
    # Pipeline control
    parser.add_argument('--pipeline', choices=['all', 'custom'], default='custom',
                       help='Run complete pipeline or custom steps')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test to verify setup')
    
    # Individual steps
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate training/test data')
    parser.add_argument('--train', action='store_true',
                       help='Train DAD model')
    parser.add_argument('--validate', action='store_true',
                       help='Validate with true IG')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform comprehensive analysis')
    
    # Data parameters
    parser.add_argument('--n-train', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--n-test', type=int, default=200,
                       help='Number of test samples')
    
    # Model parameters
    parser.add_argument('--population', type=int, default=500,
                       help='Population size')
    parser.add_argument('--num-experiments', type=int, default=4,
                       help='Number of sequential experiments')
    parser.add_argument('--noise-scale', type=float, default=0.1,
                       help='Observation noise scale')
    
    # Network architecture
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--encoding-dim', type=int, default=16,
                       help='Encoding dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of hidden layers')
    
    # Training parameters
    parser.add_argument('--num-steps', type=int, default=1500,
                       help='Number of training steps')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num-outer', type=int, default=20,
                       help='Number of outer samples for PCE')
    parser.add_argument('--num-inner', type=int, default=10,
                       help='Number of inner samples for PCE')
    parser.add_argument('--train-baselines', action='store_true',
                       help='Also train baseline models')
    
    # Validation parameters
    parser.add_argument('--n-mc-validation', type=int, default=200,
                       help='Number of MC samples for IG validation')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model for validation')
    
    # General parameters
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=-1,
                       help='Random seed (-1 for random)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Run appropriate mode
    if args.test:
        quick_test()
    else:
        run_complete_pipeline(args)