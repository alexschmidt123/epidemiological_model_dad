import torch
import matplotlib.pyplot as plt

def visualize_infected(ys, ts, initial_params, sample_index=0):
    """
    Visualize the Infected compartment (I) for a single sample over time with initial parameters shown in the plot.
    
    Args:
    ys: Tensor containing the infected compartment data.
    ts: Tensor containing the time steps.
    initial_params: Tuple of initial parameters (N, I0, beta, alpha, gamma, delta).
    sample_index: The index of the sample to plot (default is the first sample).
    """
    N, I0, beta, alpha, gamma, delta = initial_params

    plt.figure(figsize=(10, 6))

    # Plot infected data for the specified sample (sample_index)
    plt.plot(ts.numpy(), ys[:, sample_index].numpy(), label=f"Infected (Sample {sample_index + 1})", color='blue')

    # Add the initial parameters to the title and plot
    plt.title(f"Infected (I) Compartment for Sample {sample_index + 1}")
    plt.suptitle(f"Initial Parameters: Population={N}, Infected_0={I0}\n"
                 f"Beta={beta}, Alpha={alpha}, Gamma={gamma}, Delta={delta}", fontsize=10)
    plt.xlabel("Time (days)")
    plt.ylabel("Infected Population")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_saved_parameters(file_path):
    """
    Load and print parameters from the saved .pt file that contains only the Infected data (I).
    """
    data = torch.load(file_path)
    
    print("Prior samples (theta parameters):")
    print(data["prior_samples"])
    
    print("\nTime grid (ts):")
    print(data["ts"])
    
    print("\nDelta t (dt):")
    print(data["dt"])
    
    print("\nInfected data (ys) shape:")
    print(data["ys"].shape)  # This should show the shape of the infected compartment (time_steps, samples)

    # Extract the initial population and parameters from the saved data
    N = data["N"]
    I0 = data["I0"]

    # Assuming we have the average of theta parameters to print
    beta, alpha, gamma, delta = data["prior_samples"].mean(0).tolist()

    # Visualize the infected data for sample 1
    visualize_infected(data["ys"], data["ts"], initial_params=(N, I0, beta, alpha, gamma, delta), sample_index=0)
    print(data)


if __name__ == "__main__":
    file_path = "data/siqr_sde_data.pt"  # Update with the path to your SIQR .pt file
    print_saved_parameters(file_path)
