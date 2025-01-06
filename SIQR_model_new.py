import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import trange, tqdm
import matplotlib.pyplot as plt


# --- Encoder and Emitter Networks ---
class EncoderNetwork(nn.Module):
    def __init__(self, design_dim, observation_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(design_dim + observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim)
        )

    def forward(self, xi, y):
        inputs = torch.cat([xi, y], dim=-1)
        return self.network(inputs)


class EmitterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, r):
        return self.network(r)


# --- Set Equivariant Design Network ---
class SetEquivariantDesignNetwork(nn.Module):
    def __init__(self, encoder, emitter, empty_value):
        super().__init__()
        self.encoder = encoder
        self.emitter = emitter
        self.register_buffer("prototype", empty_value.clone())

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            sum_encoding = self.prototype.new_zeros(self.encoder.network[-1].out_features)
        else:
            sum_encoding = sum(self.encoder(xi=design, y=obs) for design, obs in design_obs_pairs)
        return self.emitter(sum_encoding)


# --- Loss Function: PCE Loss ---
def pce_loss(prior_samples, observed_samples, num_inner_samples=10):
    batch_size = prior_samples.size(0)
    repeated_prior = prior_samples.unsqueeze(1).repeat(1, num_inner_samples, 1)
    observed_samples = observed_samples.repeat(1, num_inner_samples, 1)

    # Pairwise distances
    distances = torch.norm(repeated_prior - observed_samples, p=2, dim=-1)
    positive_pairs = distances.diagonal(dim1=0, dim2=1)
    contrastive_pairs = distances.mean(dim=-1)

    # Mutual Information (approximate)
    mi_loss = -(positive_pairs.mean() - contrastive_pairs.mean())
    return mi_loss


# --- Helper Functions ---
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    data = torch.load(file_path)

    # Use only the infected compartment (`ys`)
    ys = data["ys"][-1, :]  # Last time point
    ys = ys.unsqueeze(-1) if ys.ndim == 1 else ys  # Ensure 2D

    print(f"ys shape: {ys.shape}, prior_samples shape: {data['prior_samples'].shape}")
    return ys, data["prior_samples"]


# --- Training Loop ---
def train_dad_model(model, prior_samples, observed_samples, epochs, lr=1e-3, use_noise=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    print(f"Initial xi shape: {prior_samples.shape}, y shape: {observed_samples.shape}")

    # Ensure observed_samples is 2D
    observed_samples = observed_samples.unsqueeze(-1) if observed_samples.ndim == 1 else observed_samples

    for epoch in trange(epochs, desc="Training Progress"):
        optimizer.zero_grad()

        # Introduce noise in observed samples
        if use_noise:
            noisy_observed_samples = observed_samples + torch.randn_like(observed_samples) * 0.01
        else:
            noisy_observed_samples = observed_samples

        # Prepare (xi, y) pairs
        designs = torch.randn_like(prior_samples)
        xi_y_pairs = [(designs[i].unsqueeze(0), noisy_observed_samples[i].unsqueeze(0)) for i in range(len(designs))]

        # Forward pass through model
        outputs = model(*xi_y_pairs)

        # Loss calculation
        loss = pce_loss(prior_samples, outputs)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Plot the loss history
    plt.plot(loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    return loss_history


# --- EIG Evaluation ---
def evaluate_model(model, prior_samples, observed_samples, strategy, n_rollout=5000):
    model.eval()
    information_gain = []

    # Ensure observed_samples is 2D
    observed_samples = observed_samples.unsqueeze(-1) if observed_samples.ndim == 1 else observed_samples

    start_time = time.time()

    with torch.no_grad():
        for _ in tqdm(range(n_rollout), desc=f"Evaluating {strategy}"):
            if strategy == "DAD":
                designs = model() + torch.randn_like(prior_samples) * 0.01
            elif strategy == "Fixed":
                designs = prior_samples
            elif strategy == "Random":
                designs = torch.rand_like(prior_samples)
            else:
                raise ValueError(f"Invalid strategy: {strategy}")

            # Calculate EIG using entropy approximation
            prior_entropy = -(prior_samples ** 2).mean()
            posterior_entropy = -((designs - observed_samples) ** 2).mean()
            ig = prior_entropy - posterior_entropy
            information_gain.append(ig.item())

    elapsed_time = time.time() - start_time

    information_gain = torch.tensor(information_gain)
    eig_mean = information_gain.mean().item()
    eig_se = (information_gain.std() / n_rollout**0.5).item()
    return eig_mean, eig_se, elapsed_time


# --- Main Experiment ---
def main():
    # Paths to the data
    TRAIN_DATA_PATH = "data/siqr_sde_data.pt"
    TEST_DATA_PATH = "data/siqr_sde_data_test.pt"

    # Load SEIR SDE data
    train_observed, train_prior_samples = load_data(TRAIN_DATA_PATH)
    test_observed, test_prior_samples = load_data(TEST_DATA_PATH)

    # Ensure test_observed is 2D
    test_observed = test_observed.unsqueeze(-1) if test_observed.ndim == 1 else test_observed

    design_dim = train_prior_samples.size(-1)
    observation_dim = 1  # Only tracking the infected compartment (I)
    hidden_dim = 256
    encoding_dim = 128
    epochs = 300
    lr = 1e-3
    n_rollout = 5000

    encoder = EncoderNetwork(design_dim, observation_dim, hidden_dim, encoding_dim)
    emitter = EmitterNetwork(encoding_dim, hidden_dim, design_dim)
    model = SetEquivariantDesignNetwork(encoder, emitter, empty_value=torch.zeros(design_dim))

    # Train DAD Model
    print("Training DAD Model...")
    start_train = time.time()
    loss_history = train_dad_model(model, train_prior_samples, train_observed, epochs, lr, use_noise=True)
    train_time = time.time() - start_train
    print(f"Training Time: {train_time:.2f} seconds")

    # Evaluate Models
    print("\n--- Evaluating Models ---")
    strategies = ["DAD", "Fixed", "Random"]
    for strategy in strategies:
        eig_mean, eig_se, elapsed_time = evaluate_model(model, test_prior_samples, test_observed, strategy, n_rollout)
        print(f"{strategy} Strategy - EIG Mean: {eig_mean:.4f}, EIG SE: {eig_se:.4f}, Time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
