import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import torch.nn.init as init

# Paths to the data
TRAIN_DATA_PATH = "data/seir_sde_data.pt"
TEST_DATA_PATH = "data/seir_sde_data_test.pt"

# Helper functions to load SEIR data
def load_seir_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    data = torch.load(file_path)
    ys_final = data['ys'][-1, :]  # Using the last time point for infected numbers
    return ys_final, data['prior_samples']

# Xavier weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

# Define Encoder Network
class EncoderNetwork(nn.Module):
    def __init__(self, design_dim, observation_dim, hidden_dim, encoding_dim, n_hidden_layers=3, activation=nn.ReLU):
        super(EncoderNetwork, self).__init__()
        self.activation = activation
        self.input_layer = nn.Linear(design_dim + observation_dim, hidden_dim)
        self.middle_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), self.activation()) for _ in range(n_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y):
        inputs = torch.cat([xi, y], dim=-1)
        x = self.activation()(self.input_layer(inputs))
        x = self.middle_layers(x)
        encoding = self.output_layer(x)
        return encoding

# Define Emitter Network
class EmitterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=3, activation=nn.ReLU):
        super(EmitterNetwork, self).__init__()
        self.activation = activation
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.middle_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), self.activation()) for _ in range(n_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, r):
        x = self.activation()(self.input_layer(r))
        x = self.middle_layers(x)
        output = self.output_layer(x)
        return output

# Define DAD Model with Encoder and Emitter
class DADModel(nn.Module):
    def __init__(self, design_dim, observation_dim, hidden_dim, encoding_dim, emitter_output_dim, n_hidden_layers=3):
        super(DADModel, self).__init__()
        self.encoder = EncoderNetwork(design_dim, observation_dim, hidden_dim, encoding_dim, n_hidden_layers)
        self.emitter = EmitterNetwork(encoding_dim, hidden_dim, emitter_output_dim, n_hidden_layers)
        self.apply(init_weights)  # Initialize weights with Xavier

    def forward(self, xi, y):
        r = self.encoder(xi, y)
        new_xi = self.emitter(r)
        return new_xi

def spce_loss(proposed_designs, true_designs, num_outer_samples=50, num_inner_samples=10):
    """
    Computes a contrastive loss function to optimize mutual information.
    """
    batch_size, design_dim = proposed_designs.size()

    # Expand proposed designs for pairwise comparison
    proposed_repeated = proposed_designs.unsqueeze(1).repeat(1, num_inner_samples, 1)  # Shape: [batch, num_inner_samples, design_dim]
    proposed_flat = proposed_repeated.view(-1, design_dim)  # Flatten to [batch * num_inner_samples, design_dim]

    # Repeat true designs along the second dimension for pairwise comparison
    true_repeated = true_designs.repeat_interleave(num_outer_samples, dim=0)  # Shape: [batch * num_outer_samples, design_dim]

    # Compute pairwise distances
    distances = torch.cdist(proposed_flat, true_repeated, p=2)  # Shape: [batch * num_inner_samples, batch * num_outer_samples]

    # Reshape distances for primary and contrastive computations
    distances = distances.view(batch_size, num_inner_samples, batch_size, num_outer_samples)

    # Mutual information estimation: primary samples
    primary_distances = distances.diagonal(dim1=0, dim2=2)  # Extract primary samples (diagonal)
    primary_log_prob = -primary_distances.mean(dim=-1)  # Average over outer samples

    # Contrastive samples: negative examples
    contrastive_distances = distances.mean(dim=1)  # Average over inner samples
    contrastive_log_prob = -contrastive_distances.mean(dim=-1)  # Average over outer samples

    mi_loss = -(primary_log_prob.mean() - contrastive_log_prob.mean())
    return mi_loss 


def train_dad_model(
    model, dataloader, optimizer, scheduler, epochs=100, device="cpu"
):
    """
    Trains the DAD model using the sPCE loss function.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xi_batch, y_batch in dataloader:
            xi_batch, y_batch = xi_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Forward pass through the model
            proposed_designs = model(xi_batch, y_batch)

            # Compute the loss
            loss = spce_loss(proposed_designs, xi_batch, num_outer_samples=50, num_inner_samples=10)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xi_batch.size(0)

        avg_loss = epoch_loss / len(dataloader.dataset)
        scheduler.step(avg_loss)  # Adjust learning rate based on avg loss

        # Print progress every 10 epochs or at the start
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")


# Function to propose new designs using DAD model
def propose_new_designs_dad(model, initial_designs, observations, device='cpu'):
    model.eval()
    with torch.no_grad():
        new_designs = model(initial_designs.to(device), observations.to(device))
    return new_designs

# Function to propose designs using Fixed strategy
def propose_fixed_designs(initial_designs, num_samples):
    return initial_designs[:num_samples]

# Function to propose designs using Random strategy
def propose_random_designs(design_dim, num_samples, device='cpu'):
    return torch.rand(num_samples, design_dim).to(device)

# Evaluation Function with Expected Information Gain (EIG)
def evaluate_model(model, test_data, strategy, num_samples, device='cpu'):
    test_y, test_xi = test_data
    start_time = time.time()  # Measure time in ms

    if strategy == "DAD":
        proposed_designs = propose_new_designs_dad(model, test_xi[:num_samples], test_y[:num_samples], device)
    elif strategy == "Fixed":
        proposed_designs = propose_fixed_designs(test_xi, num_samples)
    elif strategy == "Random":
        design_dim = test_xi.shape[1]
        proposed_designs = propose_random_designs(design_dim, num_samples, device=device)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    end_time = time.time()
    experiment_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Calculate EIG based on the predicted and true y values
    predicted_y = proposed_designs[:, 0]  # Assume infected number is the first element in designs
    true_y = test_y[:num_samples]
    eig = torch.mean((predicted_y - true_y)**2).item()  # Calculating EIG

    return eig, experiment_time

# Main Experiment
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load training and test data
    train_y, train_xi = load_seir_data(TRAIN_DATA_PATH)
    test_y, test_xi = load_seir_data(TEST_DATA_PATH)
    
    # Hyperparameters
    design_dim = train_xi.shape[1]
    observation_dim = 1  # Tracking only infected individuals
    hidden_dim = 128  # Increased hidden dimension for better capacity
    encoding_dim = 64  # Increased encoding dimension
    emitter_output_dim = design_dim
    num_samples = 50  # Number of samples for testing
    epochs = 150  # Number of training epochs
    learning_rate = 0.001  # Learning rate
    lr_scheduler_gamma = 0.8  # Slower decay factor

    # Prepare training data
    train_dataset = torch.utils.data.TensorDataset(train_xi, train_y.unsqueeze(1))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize DAD model with modified architecture
    dad_model = DADModel(design_dim, observation_dim, hidden_dim, encoding_dim, emitter_output_dim, n_hidden_layers=3).to(device)
    
    optimizer = optim.Adam(dad_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True)

    # Train DAD Model
    print("\n--- Training DAD Model ---")
    train_dad_model(dad_model, train_dataloader, optimizer, scheduler, epochs=epochs, device=device)
    
    # Evaluation on Test Data
    test_data = (test_y.unsqueeze(1), test_xi)
    
    print("\n--- Evaluating Models ---")
    
    # DAD Model Evaluation
    eig_dad, time_dad = evaluate_model(dad_model, test_data, strategy="DAD", num_samples=num_samples, device=device)
    print(f"DAD Model - EIG: {eig_dad:.4f}, Time: {time_dad:.4f} ms")
    
    # Fixed Design Evaluation
    eig_fixed, time_fixed = evaluate_model(dad_model, test_data, strategy="Fixed", num_samples=num_samples, device=device)
    print(f"Fixed Design - EIG: {eig_fixed:.4f}, Time: {time_fixed:.4f} ms")
    
    # Random Design Evaluation
    eig_random, time_random = evaluate_model(dad_model, test_data, strategy="Random", num_samples=num_samples, device=device)
    print(f"Random Design - EIG: {eig_random:.4f}, Time: {time_random:.4f} ms")

if __name__ == "__main__":
    main()
