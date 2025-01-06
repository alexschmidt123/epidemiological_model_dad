import os
import time
import torch
import torch.nn as nn
from tqdm import trange

# --- Encoder and Emitter Networks ---
class EncoderNetwork(nn.Module):
    def __init__(self, design_dim, observation_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.design_dim = design_dim  # Save design dimension
        self.observation_dim = observation_dim  # Save observation dimension
        self.network = nn.Sequential(
            nn.Linear(design_dim + observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim)
        )

    def forward(self, xi, y):
        # Ensure xi and y have correct dimensions
        if xi.ndim == 1:
            xi = xi.unsqueeze(0)  # Add batch dimension if missing
        if y.ndim == 1:
            y = y.unsqueeze(0)  # Add batch dimension if missing

        # Concatenate inputs
        inputs = torch.cat([xi, y], dim=-1)

        # Debugging prints
        print(f"xi shape: {xi.shape}, y shape: {y.shape}, concatenated shape: {inputs.shape}")

        # Validate concatenated input dimensions
        expected_dim = self.design_dim + self.observation_dim
        if inputs.size(-1) != expected_dim:
            raise ValueError(
                f"Input dimensions do not match: got {inputs.size(-1)}, expected {expected_dim}"
            )

        # Pass through the network
        return self.network(inputs)


class EmitterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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
        encodings = []
        for pair in design_obs_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"Each pair in design_obs_pairs must contain exactly two elements, but got: {pair}")
            design, obs = pair
            encodings.append(self.encoder(xi=design, y=obs))
        sum_encoding = torch.sum(torch.stack(encodings), dim=0)
        return self.emitter(sum_encoding)




# --- PCE Loss Function ---
class PriorContrastiveEstimationScoreGradient:
    def __init__(self, num_outer_samples, num_inner_samples):
        self.num_outer_samples = num_outer_samples
        self.num_inner_samples = num_inner_samples

    def compute_observation_log_prob(self, model, prior_samples, observations):
        """
        Compute the log-probabilities of the model's outputs compared to observations.
        """
        batch_size = prior_samples.size(0)

        # Create pairs from the batch
        design_obs_pairs = [(prior_samples[i], observations[:, i]) for i in range(batch_size)]

        # Forward pass through the model
        outputs = model(*design_obs_pairs)

        # Ensure the outputs have the same shape as the observations
        if outputs.shape != observations.shape[1:]:  # Observation shape: [time_steps, batch]
            raise ValueError(
                f"Shape mismatch: outputs.shape = {outputs.shape}, "
                f"observations.shape = {observations.shape}"
            )

        # Compute log-probability (example: Gaussian log-likelihood)
        log_prob = -((outputs - observations.T) ** 2).sum(dim=-1)
        return log_prob



    def differentiable_loss(self, model, prior_samples, observed_samples):
        """
        Computes the differentiable loss for the model.
        """
        obs_log_prob_primary = self.compute_observation_log_prob(model, prior_samples, observed_samples)
        obs_log_prob_contrastive = self.compute_observation_log_prob(model, observed_samples, prior_samples)

        # Ensure both tensors have compatible dimensions
        obs_log_prob_primary = obs_log_prob_primary.unsqueeze(0)  # Add a batch dimension
        obs_log_prob_contrastive = obs_log_prob_contrastive.unsqueeze(1)  # Align dimensions

        # Combine probabilities and compute the loss
        obs_log_prob_combined = torch.cat(
            [obs_log_prob_primary.expand_as(obs_log_prob_contrastive), obs_log_prob_contrastive], dim=0
        ).logsumexp(dim=0)

        loss = (obs_log_prob_combined - obs_log_prob_primary.squeeze(0)).mean()

        return loss


    def loss(self, model, prior_samples, observed_samples):
        return self.differentiable_loss(model, prior_samples, observed_samples).item()


# --- Training Function ---
def train_dad_model(model, prior_samples, observed_samples, epochs, lr=1e-3, gamma=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_history = []

    pce = PriorContrastiveEstimationScoreGradient(num_outer_samples=20, num_inner_samples=10)

    for epoch in trange(epochs, desc="Training Progress"):
        optimizer.zero_grad()

        # Ensure prior_samples and observed_samples are tensors
        if not isinstance(prior_samples, torch.Tensor) or not isinstance(observed_samples, torch.Tensor):
            raise ValueError("prior_samples and observed_samples must be torch tensors.")

        # Compute loss
        loss = pce.loss(model, prior_samples, observed_samples)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")

    return loss_history


# --- Evaluation Function ---
def evaluate_model(model, prior_samples, observed_samples, strategy, n_rollout=1000):
    model.eval()
    information_gain = []
    start_time = time.time()

    for _ in range(n_rollout):
        if strategy == "DAD":
            designs = model() + torch.randn_like(prior_samples) * 0.01
        elif strategy == "Fixed":
            designs = prior_samples
        elif strategy == "Random":
            designs = torch.rand_like(prior_samples)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        prior_log_prob = -(prior_samples ** 2).sum(dim=-1)
        posterior_log_prob = -((designs - observed_samples) ** 2).sum(dim=-1)

        prior_entropy = -(prior_log_prob.exp() * prior_log_prob).sum()
        posterior_entropy = -(posterior_log_prob.exp() * posterior_log_prob).sum()

        ig = prior_entropy - posterior_entropy
        information_gain.append(ig.item())

    elapsed_time = time.time() - start_time
    eig_mean = torch.tensor(information_gain).mean().item()
    eig_std = torch.tensor(information_gain).std().item()
    return eig_mean, eig_std, elapsed_time


# --- Main Function ---
def main():
    TRAIN_DATA_PATH = "data/seir_sde_data.pt"
    TEST_DATA_PATH = "data/seir_sde_data_test.pt"

    train_data = torch.load(TRAIN_DATA_PATH)
    test_data = torch.load(TEST_DATA_PATH)

    train_prior_samples, train_observed = train_data["prior_samples"], train_data["ys"]
    test_prior_samples, test_observed = test_data["prior_samples"], test_data["ys"]

    design_dim = train_prior_samples.size(-1)
    observation_dim = 1
    hidden_dim = 256
    encoding_dim = 128
    epochs = 500
    lr = 1e-3
    gamma = 0.1
    print(f"train_prior_samples shape: {train_prior_samples.shape}")
    print(f"train_observed shape: {train_observed.shape}")

    encoder = EncoderNetwork(
        design_dim=train_prior_samples.size(-1),  # Dimensionality of designs
        observation_dim=train_observed.size(-1),  # Dimensionality of observations
        hidden_dim=hidden_dim,
        encoding_dim=encoding_dim
    )

    emitter = EmitterNetwork(encoding_dim, hidden_dim, design_dim)
    model = SetEquivariantDesignNetwork(encoder, emitter, empty_value=torch.zeros(design_dim))

    print("Training DAD Model...")
    loss_history = train_dad_model(model, train_prior_samples, train_observed, epochs, lr, gamma)

    print("\nEvaluating Models...")
    strategies = ["DAD", "Fixed", "Random"]
    for strategy in strategies:
        eig_mean, eig_std, elapsed_time = evaluate_model(model, test_prior_samples, test_observed, strategy)
        print(f"{strategy} Strategy - EIG Mean: {eig_mean:.4f}, EIG Std: {eig_std:.4f}, Time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
