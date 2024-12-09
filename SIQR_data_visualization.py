import torch
import matplotlib.pyplot as plt

# Load the data
data = torch.load("data/siqr_sde_data.pt")

# Extract time steps, compartment values, and initial conditions
ts = data["ts"]
ys = data["ys"]  # Infected compartment (index 1 in SIQR model)
num_samples = ys.shape[1]  # Total number of samples in the saved data

# Select a sample to visualize
sample_index = 1  # Change this index to visualize different samples
infected = ys[:, sample_index]  # Infected compartment over time for the selected sample

# Plot the infected compartment over time
plt.figure(figsize=(10, 6))
plt.plot(ts, infected, label="Infected (I)", color="red")
plt.xlabel("Time")
plt.ylabel("Population Count")
plt.title(f"Infected Compartment Over Time for Sample {sample_index}")
plt.legend()
plt.grid()
plt.show()
