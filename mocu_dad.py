# mocu_dad_corrected.py
# Corrected implementation fixing the belief update and simulation issues

import os, json, math, argparse
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, wilcoxon
from scipy.integrate import odeint
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class ExperimentConfig:
    """Configuration class for experiment parameters with validation."""
    particles: int = 1500
    episodes: int = 1000
    beliefs: int = 500
    tests: int = 120
    horizon: int = 5
    my: int = 80
    seeds: int = 2
    save_oracle: bool = False
    seed: int = 0
    
    # Model parameters
    beta_low: float = 0.2
    beta_high: float = 0.6
    gamma_low: float = 0.05
    gamma_high: float = 0.25
    S0: float = 0.99
    I0: float = 0.01
    R0: float = 0.0
    sigma_obs: float = 0.004
    
    # Time grid
    t_min: float = 1.0
    t_max: float = 20.0
    n_candidates: int = 18
    
    # Training parameters
    lr_policy: float = 2e-3
    lr_value: float = 2e-3
    lr_mpnn: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    mpnn_epochs: int = 160
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.particles > 0, "particles must be positive"
        assert self.episodes > 0, "episodes must be positive"
        assert self.beliefs > 0, "beliefs must be positive"
        assert self.tests > 0, "tests must be positive"
        assert self.horizon > 0, "horizon must be positive"
        assert 0 < self.beta_low < self.beta_high, "Invalid beta range"
        assert 0 < self.gamma_low < self.gamma_high, "Invalid gamma range"
        assert 0 < self.S0 <= 1, "S0 must be in (0,1]"
        assert 0 <= self.I0 <= 1, "I0 must be in [0,1]"
        assert 0 <= self.R0 <= 1, "R0 must be in [0,1]"
        assert abs(self.S0 + self.I0 + self.R0 - 1.0) < 1e-6, "S0+I0+R0 must equal 1"


class SIRSimulator:
    """Fixed SIR simulator using scipy.integrate for accuracy and no caching."""
    
    def __init__(self, dt: float = 0.02):
        self.dt = dt
    
    def simulate_single(self, beta: float, gamma: float, S0: float, I0: float, R0: float,
                       t_grid: np.ndarray) -> np.ndarray:
        """Simulate single SIR trajectory using scipy.integrate.odeint for accuracy."""
        
        def sir_ode(y, t, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I
            dIdt = beta * S * I - gamma * I
            dRdt = gamma * I
            return [dSdt, dIdt, dRdt]
        
        y0 = [S0, I0, R0]
        
        try:
            sol = odeint(sir_ode, y0, t_grid, args=(beta, gamma), rtol=1e-8, atol=1e-10)
            I_vals = sol[:, 1]  # Extract I(t)
            return np.maximum(0.0, I_vals)  # Ensure non-negativity
        except Exception as e:
            logger.warning(f"SIR simulation failed for β={beta:.3f}, γ={gamma:.3f}: {e}")
            return I0 * np.exp(-gamma * t_grid)  # Fallback
    
    def simulate_ensemble_at_time(self, betas: np.ndarray, gammas: np.ndarray,
                                 S0: float, I0: float, R0: float, t: float) -> np.ndarray:
        """Simulate ensemble at a specific time point (no caching)."""
        I_vals = np.zeros(len(betas))
        
        for j in range(len(betas)):
            try:
                t_grid = np.array([0.0, t]) if t > 0 else np.array([0.0])
                I_traj = self.simulate_single(float(betas[j]), float(gammas[j]), S0, I0, R0, t_grid)
                I_vals[j] = I_traj[-1]  # Take value at time t
            except Exception as e:
                logger.warning(f"Particle {j} simulation failed: {e}")
                I_vals[j] = I0 * np.exp(-float(gammas[j]) * t)  # Fallback
        
        return I_vals


class MOCUCalculator:
    """MOCU (Maximum Over Current Uncertainty) calculator with numerical stability."""
    
    @staticmethod
    def alpha_star_vec(betas: np.ndarray, gammas: np.ndarray, S0: float) -> np.ndarray:
        """Compute alpha* values with numerical stability."""
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = gammas / (betas * S0)
            alpha = np.maximum(0.0, 1.0 - ratio)
            alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        return alpha
    
    @staticmethod
    def compute_mocu(betas: np.ndarray, gammas: np.ndarray,
                    weights: np.ndarray, S0: float) -> float:
        """Compute MOCU with improved numerical stability."""
        alpha = MOCUCalculator.alpha_star_vec(betas, gammas, S0)
        xi_star = float(np.max(alpha))
        exp_alpha = float(np.sum(weights * alpha))
        return max(0.0, xi_star - exp_alpha)
    
    @staticmethod
    def gaussian_logpdf(x: float, mu_vec: np.ndarray, sigma: float) -> np.ndarray:
        """Compute log PDF with numerical stability."""
        log_norm = -0.5 * np.log(2 * np.pi * sigma**2)
        log_exp = -0.5 * ((x - mu_vec) / sigma)**2
        return log_norm + log_exp


class BeliefProcessor:
    """Fixed belief processor with proper simulation calls."""
    
    def __init__(self, simulator: SIRSimulator):
        self.simulator = simulator
    
    @staticmethod
    def update_belief_fixed(betas: np.ndarray, gammas: np.ndarray, weights: np.ndarray,
                           observations: List[Tuple[float, float]], S0: float, I0: float, R0: float,
                           sigma_obs: float, simulator: SIRSimulator) -> np.ndarray:
        """Update belief weights using fresh simulations for each observation."""
        logw = np.log(weights + 1e-300)
        
        for t_obs, y_obs in observations:
            # Simulate all particles at observation time
            I_predictions = simulator.simulate_ensemble_at_time(
                betas, gammas, S0, I0, R0, t_obs
            )
            
            # Check prediction diversity
            pred_std = np.std(I_predictions)
            if pred_std < 1e-6:
                logger.warning(f"Low prediction diversity at t={t_obs:.1f}: std={pred_std:.2e}")
            
            # Update log weights
            log_likelihood = MOCUCalculator.gaussian_logpdf(y_obs, I_predictions, sigma_obs)
            logw += log_likelihood
        
        # Normalize in log space
        logw_max = np.max(logw)
        logw -= logw_max
        w_new = np.exp(logw)
        w_sum = np.sum(w_new)
        
        if w_sum > 1e-300:
            w_new /= w_sum
        else:
            logger.warning("All weights too small, reverting to uniform")
            w_new = np.ones(len(weights)) / len(weights)
        
        return w_new
    
    @staticmethod
    def extract_features(betas: np.ndarray, gammas: np.ndarray,
                        weights: np.ndarray, S0: float) -> np.ndarray:
        """Extract belief summary features with robust statistics."""
        # Weighted statistics
        mb = float(np.sum(weights * betas))
        mg = float(np.sum(weights * gammas))
        sb = float(np.sqrt(np.sum(weights * (betas - mb)**2)))
        sg = float(np.sqrt(np.sum(weights * (gammas - mg)**2)))
        
        # Range statistics
        bmin, bmax = float(np.min(betas)), float(np.max(betas))
        gmin, gmax = float(np.min(gammas)), float(np.max(gammas))
        
        # MOCU
        M = MOCUCalculator.compute_mocu(betas, gammas, weights, S0)
        
        # Additional robust features
        R0_vals = betas / gammas
        mR0 = float(np.sum(weights * R0_vals))
        sR0 = float(np.sqrt(np.sum(weights * (R0_vals - mR0)**2)))
        
        features = np.array([
            mb, mg, sb, sg, bmin, bmax, gmin, gmax, S0, M, mR0, sR0
        ], dtype=np.float32)
        
        return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


class ImprovedMPNN(nn.Module):
    """Enhanced MPNN with better architecture and regularization."""
    
    def __init__(self, node_in: int = 5, hidden: int = 64, T: int = 2, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.dropout = dropout
        
        self.enc = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.msg = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.upd = nn.GRUCell(hidden, hidden)
        
        self.read = nn.Sequential(
            nn.Linear(2 * hidden + 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, beta_node, gamma_node, t_scalar, M_scalar):
        if beta_node.dim() == 1:
            beta_node = beta_node.unsqueeze(0)
        if gamma_node.dim() == 1:
            gamma_node = gamma_node.unsqueeze(0)
        if t_scalar.dim() == 1:
            t_scalar = t_scalar.unsqueeze(0)
        if M_scalar.dim() == 1:
            M_scalar = M_scalar.unsqueeze(0)
        
        hb = self.enc(beta_node)
        hg = self.enc(gamma_node)
        
        for _ in range(self.T):
            mb = self.msg(hg)
            mg = self.msg(hb)
            hb_new = self.upd(mb, hb)
            hg_new = self.upd(mg, hg)
            hb = hb + hb_new
            hg = hg + hg_new
        
        H = torch.cat([hb, hg], dim=1)
        x = torch.cat([H, t_scalar, M_scalar], dim=1)
        return self.read(x)


class ImprovedPolicyNetwork(nn.Module):
    """Enhanced policy network with LayerNorm (no BatchNorm for single samples)."""
    
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_actions)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class ImprovedValueNetwork(nn.Module):
    """Enhanced value network with LayerNorm (no BatchNorm for single samples)."""
    
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class ExperimentRunner:
    """Main experiment runner with improved organization and error handling."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.simulator = SIRSimulator()
        
        # Set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Create output directory
        self.outdir = Path("./outputs")
        self.outdir.mkdir(exist_ok=True)
        
        # Save configuration
        with open(self.outdir / "config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)
        
        logger.info(f"Experiment initialized with config: {config}")
    
    def setup_experiment(self):
        """Initialize particles and candidate times (no precomputed cache)."""
        logger.info("Setting up experiment...")
        
        # Initial particle belief
        self.betas0 = np.random.uniform(
            self.config.beta_low, self.config.beta_high, size=self.config.particles
        )
        self.gammas0 = np.random.uniform(
            self.config.gamma_low, self.config.gamma_high, size=self.config.particles
        )
        self.weights0 = np.ones(self.config.particles) / self.config.particles
        
        # Candidate times (no cache)
        self.candidate_times = np.linspace(
            self.config.t_min, self.config.t_max, self.config.n_candidates
        )
        
        # Initialize belief processor
        self.belief_processor = BeliefProcessor(self.simulator)
        
        logger.info(f"Setup complete. Particles: {self.config.particles}")
        logger.info(f"β range: [{self.betas0.min():.3f}, {self.betas0.max():.3f}]")
        logger.info(f"γ range: [{self.gammas0.min():.3f}, {self.gammas0.max():.3f}]")
        logger.info(f"R₀ range: [{(self.betas0/self.gammas0).min():.2f}, {(self.betas0/self.gammas0).max():.2f}]")
    
    def train_mpnn_surrogate(self) -> ImprovedMPNN:
        """Train MPNN surrogate with improved data generation and training."""
        logger.info("Training MPNN surrogate...")
        
        rows = []
        for _ in tqdm(range(self.config.beliefs), desc="Generating training data"):
            b, g, w = self._sample_realistic_belief()
            for t in self.candidate_times:
                try:
                    R_est, _, _ = self._compute_expected_remaining_mocu(t, b, g, w)
                    bn, gn, ts, Ms = self._build_mpnn_inputs(b, g, w, t)
                    rows.append({"bn": bn, "gn": gn, "t": ts, "M": Ms, "R": R_est})
                except Exception as e:
                    logger.warning(f"Failed to compute R for t={t}: {e}")
                    continue
        
        if not rows:
            raise RuntimeError("No valid training data generated")
        
        # Prepare tensors
        beta_nodes = torch.stack([torch.tensor(r["bn"]) for r in rows])
        gamma_nodes = torch.stack([torch.tensor(r["gn"]) for r in rows])
        t_tensors = torch.stack([torch.tensor(r["t"]) for r in rows])
        M_tensors = torch.stack([torch.tensor(r["M"]) for r in rows])
        R_targets = torch.tensor([r["R"] for r in rows]).unsqueeze(1)
        
        # Create data loader
        dataset = TensorDataset(beta_nodes, gamma_nodes, t_tensors, M_tensors, R_targets)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Initialize model
        mpnn = ImprovedMPNN(node_in=5, hidden=64, T=2, dropout=0.1)
        optimizer = optim.Adam(mpnn.parameters(), lr=self.config.lr_mpnn, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        criterion = nn.MSELoss()
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.mpnn_epochs):
            mpnn.train()
            total_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                pred = mpnn(*batch[:-1])
                loss = criterion(pred, batch[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mpnn.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            if patience_counter > 20:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"MPNN training completed. Final loss: {best_loss:.6f}")
        return mpnn
    
    def _sample_realistic_belief(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample realistic posterior belief by simulating observations."""
        w = self.weights0.copy()
        n_steps = np.random.randint(1, 4)
        
        for _ in range(n_steps):
            t_obs = np.random.choice(self.candidate_times)
            anc = np.random.choice(np.arange(len(w)), p=w)
            
            # Simulate observation
            t_grid = np.array([0.0, t_obs])
            I_traj = self.simulator.simulate_single(
                self.betas0[anc], self.gammas0[anc], 
                self.config.S0, self.config.I0, self.config.R0, t_grid
            )
            y = float(I_traj[-1] + np.random.normal(0.0, self.config.sigma_obs))
            
            w = BeliefProcessor.update_belief_fixed(
                self.betas0, self.gammas0, w, [(t_obs, y)],
                self.config.S0, self.config.I0, self.config.R0,
                self.config.sigma_obs, self.simulator
            )
        
        return self.betas0, self.gammas0, w
    
    def _compute_expected_remaining_mocu(self, t: float, betas: np.ndarray, 
                                       gammas: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
        """Compute expected remaining MOCU with proper simulation."""
        M_current = MOCUCalculator.compute_mocu(betas, gammas, weights, self.config.S0)
        
        # Generate predictive samples
        anc = np.random.choice(np.arange(len(weights)), size=self.config.my, p=weights, replace=True)
        
        # Simulate observations at time t for selected particles
        I_predictions = self.simulator.simulate_ensemble_at_time(
            betas[anc], gammas[anc], self.config.S0, self.config.I0, self.config.R0, t
        )
        y_samples = I_predictions + np.random.normal(0.0, self.config.sigma_obs, size=self.config.my)
        
        # Compute expected MOCU after each possible observation
        R_vals = []
        for y in y_samples:
            try:
                w_new = BeliefProcessor.update_belief_fixed(
                    betas, gammas, weights, [(t, y)], 
                    self.config.S0, self.config.I0, self.config.R0, 
                    self.config.sigma_obs, self.simulator
                )
                M_new = MOCUCalculator.compute_mocu(betas, gammas, w_new, self.config.S0)
                R_vals.append(M_new)
            except Exception as e:
                logger.warning(f"MOCU computation failed for y={y}: {e}")
                R_vals.append(M_current)
        
        R_est = float(np.mean(R_vals))
        EMR = float(M_current - R_est)
        
        return R_est, EMR, M_current
    
    def _build_mpnn_inputs(self, betas: np.ndarray, gammas: np.ndarray, 
                          weights: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build MPNN input features from belief state."""
        features = BeliefProcessor.extract_features(betas, gammas, weights, self.config.S0)
        
        mb, mg, sb, sg = features[0], features[1], features[2], features[3]
        bmin, bmax = features[4], features[5]
        gmin, gmax = features[6], features[7]
        M = features[9]
        
        beta_node = np.array([mb, sb, bmin, bmax, self.config.S0], dtype=np.float32)
        gamma_node = np.array([mg, sg, gmin, gmax, self.config.S0], dtype=np.float32)
        t_scalar = np.array([t], dtype=np.float32)
        M_scalar = np.array([M], dtype=np.float32)
        
        return beta_node, gamma_node, t_scalar, M_scalar
    
    def validate_surrogate(self, mpnn: ImprovedMPNN) -> None:
        """Validate surrogate model with rank correlation analysis."""
        logger.info("Validating surrogate model...")
        
        spears = []
        for _ in range(80):
            b, g, w = self._sample_realistic_belief()
            R_true, R_hat = [], []
            
            for t in self.candidate_times:
                try:
                    Rt, _, _ = self._compute_expected_remaining_mocu(t, b, g, w)
                    bn, gn, ts, Ms = self._build_mpnn_inputs(b, g, w, t)
                    with torch.no_grad():
                        mpnn.eval()
                        Rh = mpnn(
                            torch.tensor(bn).unsqueeze(0),
                            torch.tensor(gn).unsqueeze(0),
                            torch.tensor(ts).unsqueeze(0),
                            torch.tensor(Ms).unsqueeze(0)
                        ).item()
                    
                    R_true.append(Rt)
                    R_hat.append(Rh)
                except Exception as e:
                    logger.warning(f"Validation failed for t={t}: {e}")
                    continue
            
            if len(R_true) > 1:
                rho, _ = spearmanr(R_true, R_hat)
                spears.append(rho if np.isfinite(rho) else 0.0)
        
        pd.DataFrame({"spearman_r": spears}).to_csv(
            self.outdir / "surrogate_rank_correlation.csv", index=False
        )
        
        mean_rho = np.mean(spears)
        logger.info(f"Surrogate validation complete. Mean Spearman correlation: {mean_rho:.3f}")
    
    def train_dad_policy(self) -> Tuple[ImprovedPolicyNetwork, ImprovedValueNetwork]:
        """Train DAD policy using actor-critic with improvements."""
        logger.info("Training DAD policy...")
        
        # Feature dimension is now 12
        feature_dim = 12
        policy = ImprovedPolicyNetwork(in_dim=feature_dim, n_actions=len(self.candidate_times))
        value = ImprovedValueNetwork(in_dim=feature_dim)
        
        opt_p = optim.Adam(policy.parameters(), lr=self.config.lr_policy, weight_decay=1e-5)
        opt_v = optim.Adam(value.parameters(), lr=self.config.lr_value, weight_decay=1e-5)
        
        scheduler_p = optim.lr_scheduler.ExponentialLR(opt_p, gamma=0.999)
        scheduler_v = optim.lr_scheduler.ExponentialLR(opt_v, gamma=0.999)
        
        for episode in tqdm(range(self.config.episodes), desc="Training DAD policy"):
            try:
                beta_star = float(np.random.uniform(self.config.beta_low, self.config.beta_high))
                gamma_star = float(np.random.uniform(self.config.gamma_low, self.config.gamma_high))
                
                logps, ents, feats, rewards = self._rollout_episode(beta_star, gamma_star, policy)
                
                if not rewards:
                    continue
                
                returns = self._compute_returns(rewards)
                returns_t = torch.tensor(returns, dtype=torch.float32)
                states_t = torch.stack(feats, dim=0)
                
                values_t = value(states_t).squeeze(1)
                advantages = returns_t - values_t.detach()
                
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                policy_loss = -(torch.stack(logps) * advantages).mean()
                entropy_loss = -self.config.entropy_coef * torch.stack(ents).mean()
                total_policy_loss = policy_loss + entropy_loss
                
                value_loss = self.config.value_coef * nn.MSELoss()(values_t, returns_t)
                
                opt_p.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.config.grad_clip)
                opt_p.step()
                
                opt_v.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value.parameters(), self.config.grad_clip)
                opt_v.step()
                
                if episode % 100 == 0:
                    scheduler_p.step()
                    scheduler_v.step()
                
                if episode % 200 == 0:
                    logger.info(f"Episode {episode}: Policy Loss: {policy_loss.item():.4f}, "
                              f"Value Loss: {value_loss.item():.4f}, "
                              f"Mean Reward: {np.mean(rewards):.4f}")
                
            except Exception as e:
                logger.warning(f"Episode {episode} failed: {e}")
                continue
        
        logger.info("DAD policy training completed")
        return policy, value
    
    def _rollout_episode(self, beta_star: float, gamma_star: float, 
                        policy: ImprovedPolicyNetwork) -> Tuple[List, List, List, List]:
        """Execute a single rollout episode with fixed simulation."""
        betas = self.betas0.copy()
        gammas = self.gammas0.copy()
        weights = self.weights0.copy()
        
        logps, ents, feats, rewards = [], [], [], []
        
        for step in range(self.config.horizon):
            try:
                # Extract belief features
                feat = torch.tensor(
                    BeliefProcessor.extract_features(betas, gammas, weights, self.config.S0)
                ).float().unsqueeze(0)
                
                # Policy forward pass
                policy.eval()
                with torch.no_grad():
                    logits = policy(feat)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    probs = torch.clamp(probs, min=1e-8, max=1.0)
                    probs = probs / probs.sum()
                    
                    m = torch.distributions.Categorical(probs=probs)
                    a_idx = int(m.sample().item())
                
                policy.train()
                
                # Store policy info
                logps.append(m.log_prob(torch.tensor(a_idx)))
                ents.append(m.entropy())
                feats.append(feat.squeeze(0))
                
                # Execute action
                chosen_t = float(self.candidate_times[a_idx])
                
                # Generate observation using true parameters
                t_grid = np.array([0.0, chosen_t]) if chosen_t > 0 else np.array([0.0])
                I_true_traj = self.simulator.simulate_single(
                    beta_star, gamma_star, self.config.S0, self.config.I0, self.config.R0, t_grid
                )
                y = float(I_true_traj[-1] + np.random.normal(0.0, self.config.sigma_obs))
                
                # Update belief
                M_before = MOCUCalculator.compute_mocu(betas, gammas, weights, self.config.S0)
                weights = BeliefProcessor.update_belief_fixed(
                    betas, gammas, weights, [(chosen_t, y)], 
                    self.config.S0, self.config.I0, self.config.R0, 
                    self.config.sigma_obs, self.simulator
                )
                M_after = MOCUCalculator.compute_mocu(betas, gammas, weights, self.config.S0)
                
                # Compute reward (EMR)
                reward = float(M_before - M_after)
                rewards.append(reward)
                
            except Exception as e:
                logger.warning(f"Step {step} failed: {e}")
                break
        
        return logps, ents, feats, rewards
    
    def _compute_returns(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """Compute discounted returns."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        return list(reversed(returns))
    
    def evaluate_methods(self, mpnn: ImprovedMPNN, policy: ImprovedPolicyNetwork) -> pd.DataFrame:
        """Evaluate all methods with comprehensive metrics."""
        logger.info("Evaluating methods...")
        
        all_records = []
        
        for seed_eval in range(self.config.seeds):
            eval_seed = self.config.seed + 1000 + seed_eval
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            
            logger.info(f"Evaluation seed {seed_eval + 1}/{self.config.seeds}")
            
            for test_idx in tqdm(range(self.config.tests), desc=f"Testing seed {seed_eval}", leave=False):
                try:
                    beta_star = float(np.random.uniform(self.config.beta_low, self.config.beta_high))
                    gamma_star = float(np.random.uniform(self.config.gamma_low, self.config.gamma_high))
                    
                    M_dad = self._run_strategy(beta_star, gamma_star, "DAD", policy=policy)
                    M_mpnn = self._run_strategy(beta_star, gamma_star, "MPNN-greedy", mpnn=mpnn)
                    
                    record = {
                        "seed_eval": seed_eval,
                        "test_idx": test_idx,
                        "beta_star": beta_star,
                        "gamma_star": gamma_star,
                        "final_M_DAD": M_dad[-1],
                        "AUC_M_DAD": self._compute_auc(M_dad),
                        "final_M_MPNN": M_mpnn[-1],
                        "AUC_M_MPNN": self._compute_auc(M_mpnn),
                    }
                    
                    if self.config.save_oracle:
                        M_true = self._run_strategy(beta_star, gamma_star, "True-greedy")
                        record.update({
                            "final_M_TRUE": M_true[-1],
                            "AUC_M_TRUE": self._compute_auc(M_true)
                        })
                    
                    all_records.append(record)
                    
                except Exception as e:
                    logger.warning(f"Test {test_idx} in seed {seed_eval} failed: {e}")
                    continue
        
        if not all_records:
            raise RuntimeError("All evaluations failed")
        
        df = pd.DataFrame(all_records)
        df.to_csv(self.outdir / "detailed.csv", index=False)
        
        logger.info(f"Evaluation completed. {len(all_records)} successful tests.")
        return df
    
    def _run_strategy(self, beta_star: float, gamma_star: float, strategy: str, 
                     policy: Optional[ImprovedPolicyNetwork] = None,
                     mpnn: Optional[ImprovedMPNN] = None) -> List[float]:
        """Run a single strategy with fixed simulation."""
        betas = self.betas0.copy()
        gammas = self.gammas0.copy()
        weights = self.weights0.copy()
        
        M_hist = [MOCUCalculator.compute_mocu(betas, gammas, weights, self.config.S0)]
        
        for step in range(self.config.horizon):
            try:
                if strategy == "DAD":
                    if policy is None:
                        raise ValueError("Policy required for DAD strategy")
                    a_idx = self._select_action_dad(betas, gammas, weights, policy)
                
                elif strategy == "MPNN-greedy":
                    if mpnn is None:
                        raise ValueError("MPNN required for MPNN-greedy strategy")
                    a_idx = self._select_action_mpnn(betas, gammas, weights, mpnn)
                
                elif strategy == "True-greedy":
                    a_idx = self._select_action_true_greedy(betas, gammas, weights)
                
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                # Execute action
                chosen_t = float(self.candidate_times[a_idx])
                
                # Generate observation using true parameters
                t_grid = np.array([0.0, chosen_t]) if chosen_t > 0 else np.array([0.0])
                I_true_traj = self.simulator.simulate_single(
                    beta_star, gamma_star, self.config.S0, self.config.I0, self.config.R0, t_grid
                )
                y = float(I_true_traj[-1] + np.random.normal(0.0, self.config.sigma_obs))
                
                # Update belief
                weights = BeliefProcessor.update_belief_fixed(
                    betas, gammas, weights, [(chosen_t, y)], 
                    self.config.S0, self.config.I0, self.config.R0, 
                    self.config.sigma_obs, self.simulator
                )
                
                # Record MOCU
                M_new = MOCUCalculator.compute_mocu(betas, gammas, weights, self.config.S0)
                M_hist.append(M_new)
                
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed at step {step}: {e}")
                M_hist.append(M_hist[-1])
        
        return M_hist
    
    def _select_action_dad(self, betas: np.ndarray, gammas: np.ndarray, 
                          weights: np.ndarray, policy: ImprovedPolicyNetwork) -> int:
        """Select action using DAD policy."""
        feat = torch.tensor(
            BeliefProcessor.extract_features(betas, gammas, weights, self.config.S0)
        ).float().unsqueeze(0)
        
        with torch.no_grad():
            policy.eval()
            logits = policy(feat)
            a_idx = int(torch.argmax(logits, dim=1).item())
        
        return a_idx
    
    def _select_action_mpnn(self, betas: np.ndarray, gammas: np.ndarray, 
                           weights: np.ndarray, mpnn: ImprovedMPNN) -> int:
        """Select action using MPNN-greedy strategy."""
        Rs = []
        for t in self.candidate_times:
            try:
                bn, gn, ts, Ms = self._build_mpnn_inputs(betas, gammas, weights, t)
                with torch.no_grad():
                    mpnn.eval()
                    R_pred = mpnn(
                        torch.tensor(bn).unsqueeze(0),
                        torch.tensor(gn).unsqueeze(0),
                        torch.tensor(ts).unsqueeze(0),
                        torch.tensor(Ms).unsqueeze(0)
                    ).item()
                Rs.append(R_pred)
            except Exception as e:
                logger.warning(f"MPNN prediction failed for t={t}: {e}")
                Rs.append(float('inf'))
        
        return int(np.argmin(Rs))
    
    def _select_action_true_greedy(self, betas: np.ndarray, gammas: np.ndarray, 
                                  weights: np.ndarray) -> int:
        """Select action using true greedy strategy (oracle)."""
        Rs = []
        for t in self.candidate_times:
            try:
                Rt, _, _ = self._compute_expected_remaining_mocu(t, betas, gammas, weights)
                Rs.append(Rt)
            except Exception as e:
                logger.warning(f"True greedy computation failed for t={t}: {e}")
                Rs.append(float('inf'))
        
        return int(np.argmin(Rs))
    
    def _compute_auc(self, values: List[float]) -> float:
        """Compute area under curve with error handling."""
        if len(values) <= 1:
            return 0.0
        
        xs = np.arange(len(values))
        try:
            return float(np.trapz(values, xs))
        except Exception:
            return float(np.sum(values))
    
    def analyze_results(self, df: pd.DataFrame) -> None:
        """Comprehensive statistical analysis with improved visualizations."""
        logger.info("Analyzing results...")
        
        summary_stats = self._compute_summary_statistics(df)
        summary_stats.to_csv(self.outdir / "summary.csv", index=False)
        
        self._create_visualizations(df, summary_stats)
        
        logger.info("\n=== SUMMARY (DAD vs MPNN-greedy) ===")
        logger.info(f"\n{summary_stats.to_string(index=False)}")
        
        logger.info(f"\nOutputs saved in {self.outdir}:")
        for filename in ["summary.csv", "detailed.csv", "final_mocu_bar.png", 
                        "auc_mocu_bar.png", "surrogate_rank_correlation.csv", 
                        "config.json", "performance_comparison.png"]:
            filepath = self.outdir / filename
            if filepath.exists():
                logger.info(f" - {filepath}")
    
    def _compute_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute comprehensive summary statistics."""
        
        def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
            if len(x) == 0:
                return (0.0, 0.0)
            
            rng = np.random.default_rng(42)
            boots = []
            n = len(x)
            
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                boots.append(float(np.mean(x[idx])))
            
            boots = np.sort(np.array(boots))
            lo_idx = int((alpha / 2.0) * n_boot)
            hi_idx = int((1.0 - alpha / 2.0) * n_boot)
            
            return (boots[lo_idx], boots[hi_idx])
        
        def compute_metric_stats(col_dad: str, col_mpnn: str) -> Dict[str, Any]:
            x_dad = df[col_dad].dropna().values
            x_mpnn = df[col_mpnn].dropna().values
            
            if len(x_dad) == 0 or len(x_mpnn) == 0:
                return {
                    "dad_mean": 0.0, "dad_ci": (0.0, 0.0),
                    "mpnn_mean": 0.0, "mpnn_ci": (0.0, 0.0),
                    "diff_ci": (0.0, 0.0), "p_value": 1.0
                }
            
            dad_mean = float(np.mean(x_dad))
            mpnn_mean = float(np.mean(x_mpnn))
            dad_ci = bootstrap_ci(x_dad)
            mpnn_ci = bootstrap_ci(x_mpnn)
            
            min_len = min(len(x_dad), len(x_mpnn))
            diff = x_dad[:min_len] - x_mpnn[:min_len]
            diff_ci = bootstrap_ci(diff)
            
            try:
                if len(diff) > 0:
                    _, p_value = wilcoxon(diff)
                else:
                    p_value = 1.0
            except Exception:
                p_value = float('nan')
            
            return {
                "dad_mean": dad_mean, "dad_ci": dad_ci,
                "mpnn_mean": mpnn_mean, "mpnn_ci": mpnn_ci,
                "diff_ci": diff_ci, "p_value": p_value
            }
        
        final_stats = compute_metric_stats("final_M_DAD", "final_M_MPNN")
        auc_stats = compute_metric_stats("AUC_M_DAD", "AUC_M_MPNN")
        
        summary = pd.DataFrame({
            "Metric": ["Final MOCU (lower better)", "AUC(MOCU) (lower better)"],
            "DAD_mean": [final_stats["dad_mean"], auc_stats["dad_mean"]],
            "DAD_95CI_low": [final_stats["dad_ci"][0], auc_stats["dad_ci"][0]],
            "DAD_95CI_high": [final_stats["dad_ci"][1], auc_stats["dad_ci"][1]],
            "MPNN_mean": [final_stats["mpnn_mean"], auc_stats["mpnn_mean"]],
            "MPNN_95CI_low": [final_stats["mpnn_ci"][0], auc_stats["mpnn_ci"][0]],
            "MPNN_95CI_high": [final_stats["mpnn_ci"][1], auc_stats["mpnn_ci"][1]],
            "Diff_95CI_low": [final_stats["diff_ci"][0], auc_stats["diff_ci"][0]],
            "Diff_95CI_high": [final_stats["diff_ci"][1], auc_stats["diff_ci"][1]],
            "Wilcoxon_p": [final_stats["p_value"], auc_stats["p_value"]]
        })
        
        return summary
    
    def _create_visualizations(self, df: pd.DataFrame, summary: pd.DataFrame) -> None:
        """Create enhanced visualizations."""
        plt.style.use('default')
        
        self._create_bar_plot(summary, "final_mocu_bar.png", "Final MOCU (lower is better)",
                             "DAD_mean", "MPNN_mean", "DAD_95CI_low", "DAD_95CI_high",
                             "MPNN_95CI_low", "MPNN_95CI_high")
        
        self._create_bar_plot(summary, "auc_mocu_bar.png", "AUC of MOCU vs steps (lower is better)",
                             "DAD_mean", "MPNN_mean", "DAD_95CI_low", "DAD_95CI_high",
                             "MPNN_95CI_low", "MPNN_95CI_high")
        
        self._create_scatter_comparison(df)
    
    def _create_bar_plot(self, summary: pd.DataFrame, filename: str, title: str,
                        dad_mean_col: str, mpnn_mean_col: str,
                        dad_low_col: str, dad_high_col: str,
                        mpnn_low_col: str, mpnn_high_col: str) -> None:
        """Create bar plot with confidence intervals."""
        plt.figure(figsize=(8, 6))
        
        row_idx = 0 if "Final" in title else 1
        dad_mean = summary.loc[row_idx, dad_mean_col]
        mpnn_mean = summary.loc[row_idx, mpnn_mean_col]
        dad_low = summary.loc[row_idx, dad_low_col]
        dad_high = summary.loc[row_idx, dad_high_col]
        mpnn_low = summary.loc[row_idx, mpnn_low_col]
        mpnn_high = summary.loc[row_idx, mpnn_high_col]
        
        means = [dad_mean, mpnn_mean]
        yerr_low = [dad_mean - dad_low, mpnn_mean - mpnn_low]
        yerr_high = [dad_high - dad_mean, mpnn_high - mpnn_mean]
        
        bars = plt.bar(["DAD", "MPNN-greedy"], means, 
                      yerr=[yerr_low, yerr_high], capsize=8,
                      color=['#2E86C1', '#E74C3C'], alpha=0.8)
        
        plt.ylabel("MOCU Value")
        plt.title(title)
        plt.grid(axis='y', alpha=0.3)
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.outdir / filename, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _create_scatter_comparison(self, df: pd.DataFrame) -> None:
        """Create scatter plot comparing methods."""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df["final_M_DAD"], df["final_M_MPNN"], alpha=0.6, s=30)
        min_val = min(df["final_M_DAD"].min(), df["final_M_MPNN"].min())
        max_val = max(df["final_M_DAD"].max(), df["final_M_MPNN"].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel("DAD Final MOCU")
        plt.ylabel("MPNN Final MOCU")
        plt.title("Final MOCU Comparison")
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(df["AUC_M_DAD"], df["AUC_M_MPNN"], alpha=0.6, s=30)
        min_val = min(df["AUC_M_DAD"].min(), df["AUC_M_MPNN"].min())
        max_val = max(df["AUC_M_DAD"].max(), df["AUC_M_MPNN"].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel("DAD AUC MOCU")
        plt.ylabel("MPNN AUC MOCU")
        plt.title("AUC MOCU Comparison")
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.outdir / "performance_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def run_experiment(self) -> None:
        """Run the complete experiment pipeline."""
        logger.info("Starting MOCU DAD experiment...")
        
        try:
            self.setup_experiment()
            mpnn = self.train_mpnn_surrogate()
            self.validate_surrogate(mpnn)
            policy, value = self.train_dad_policy()
            results_df = self.evaluate_methods(mpnn, policy)
            self.analyze_results(results_df)
            
            logger.info("Experiment completed successfully!")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise


def main():
    """Main entry point with improved argument parsing."""
    parser = argparse.ArgumentParser(
        description="MOCU DAD Focus Experiment - Corrected Implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--particles", type=int, default=1500, help="Number of particles")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--beliefs", type=int, default=500, help="Training beliefs for MPNN")
    parser.add_argument("--tests", type=int, default=120, help="Test instances")
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon")
    parser.add_argument("--my", type=int, default=80, help="MC samples for R(x;Pi)")
    parser.add_argument("--seeds", type=int, default=2, help="Evaluation seeds")
    parser.add_argument("--save_oracle", action="store_true", help="Evaluate oracle")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    parser.add_argument("--beta_low", type=float, default=0.2, help="Beta lower bound")
    parser.add_argument("--beta_high", type=float, default=0.6, help="Beta upper bound")
    parser.add_argument("--gamma_low", type=float, default=0.05, help="Gamma lower bound")
    parser.add_argument("--gamma_high", type=float, default=0.25, help="Gamma upper bound")
    
    parser.add_argument("--lr_policy", type=float, default=2e-3, help="Policy learning rate")
    parser.add_argument("--lr_value", type=float, default=2e-3, help="Value learning rate")
    parser.add_argument("--lr_mpnn", type=float, default=1e-3, help="MPNN learning rate")
    parser.add_argument("--mpnn_epochs", type=int, default=160, help="MPNN training epochs")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(**vars(args))
    runner = ExperimentRunner(config)
    runner.run_experiment()


if __name__ == "__main__":
    main()