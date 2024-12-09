import matplotlib.pyplot as plt

# Visualization Function for EIG, Time, and Loss
def visualize_results(eig_dad, time_dad, eig_fixed, time_fixed, eig_random, time_random, loss_history):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Loss Curve for DAD
    axs[0].plot(range(1, len(loss_history) + 1), loss_history, label="DAD Loss", color='b')
    axs[0].set_title("DAD Training Loss Curve")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot 2: EIG Comparison (Bar Chart)
    strategies = ['DAD', 'Fixed', 'Random']
    eig_values = [eig_dad, eig_fixed, eig_random]
    bars1 = axs[1].bar(strategies, eig_values, color=['b', 'g', 'r'])
    axs[1].set_title("Expected Information Gain (EIG) Comparison")
    axs[1].set_ylabel("EIG")

    # Add value labels to the EIG bars
    for bar in bars1:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    # Plot 3: Time Comparison (Bar Chart)
    time_values = [time_dad, time_fixed, time_random]
    bars2 = axs[2].bar(strategies, time_values, color=['b', 'g', 'r'])
    axs[2].set_title("Time Comparison")
    axs[2].set_ylabel("Time (milliseconds)")

    # Add value labels to the Time bars
    for bar in bars2:
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Results from the experiment
eig_dad, time_dad = 4354.5986, 0.6018  # DAD results
eig_fixed, time_fixed = 3471.4072, 0.0031  # Fixed design results
eig_random, time_random = 3463.8748, 0.3493  # Random design results

# Loss history from the training of the DAD model
loss_history = [
    -0.087264, -0.128755, -0.149517, -0.155273, -0.160228, 
    -0.168820, -0.173709, -0.176574, -0.177242, -0.181705, 
    -0.182964, -0.183384, -0.181516, -0.186231, -0.189535, 
    -0.189221
]  # Loss at specified epochs

# Visualize the results
visualize_results(eig_dad, time_dad, eig_fixed, time_fixed, eig_random, time_random, loss_history)
