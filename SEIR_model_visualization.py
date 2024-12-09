import matplotlib.pyplot as plt

# Visualization Function for EIG, Time, and Loss
def visualize_results(eig_dad, time_dad, eig_fixed, time_fixed, eig_random, time_random, loss_history):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Loss Curve for DAD
    axs[0].plot(loss_history, label="DAD Loss", color='b')
    axs[0].set_title("DAD Loss Curve")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot 2: EIG Comparison (Bar Chart)
    strategies = ['DAD', 'Fixed', 'Random']
    eig_values = [eig_dad, eig_fixed, eig_random]
    axs[1].bar(strategies, eig_values, color=['b', 'g', 'r'])
    axs[1].set_title("Expected Information Gain (EIG) Comparison")
    axs[1].set_ylabel("EIG")

    # Plot 3: Time Comparison (Bar Chart)
    time_values = [time_dad, time_fixed, time_random]
    axs[2].bar(strategies, time_values, color=['b', 'g', 'r'])
    axs[2].set_title("Time Comparison")
    axs[2].set_ylabel("Time (milliseconds)")

    plt.tight_layout()
    plt.show()

# Main function to run the visualization
def main():
    # Results from your experiment
    eig_dad, time_dad = 2131.3611, 0.3920  # DAD results
    eig_fixed, time_fixed = 2131.3479, 0.0172  # Fixed design results
    eig_random, time_random = 2116.5908, 0.4718  # Random design results

    # Loss history from the training of the DAD model
    loss_history = [0.0227, 0.0113, 0.0083, 0.0021, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019]  # Every 10 epochs

    # Visualize the results
    visualize_results(eig_dad, time_dad, eig_fixed, time_fixed, eig_random, time_random, loss_history)

if __name__ == "__main__":
    main()
