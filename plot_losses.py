import numpy as np
import matplotlib.pyplot as plt

# Load the three numpy files
auxfree_path = "output/bal_0.0-auxfree_0.001/loss_records.npy"
bal_path = "output/bal_0.001-auxfree_0.0/loss_records.npy" 
both_path = "output/bal_0.001-auxfree_0.001/loss_records.npy"

auxfree_records = np.load(auxfree_path)
bal_records = np.load(bal_path)
both_records = np.load(both_path)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('Training Metrics Comparison')

# Plot LLM Loss
ax1.plot(auxfree_records[:,3], label='Aux-Loss-Free', alpha=0.7)
ax1.plot(bal_records[:,3], label='Balance Loss', alpha=0.7)
ax1.plot(both_records[:,3], label='Both Methods', alpha=0.7)
ax1.set_ylabel('LM Loss')
ax1.set_xlabel('Training Steps')
ax1.legend()
ax1.grid(True)

# Plot Aux Loss
ax2.plot(auxfree_records[:,4], label='Aux-Loss-Free', alpha=0.7)
ax2.plot(bal_records[:,4], label='Balance Loss', alpha=0.7)
ax2.plot(both_records[:,4], label='Both Methods', alpha=0.7)
ax2.set_ylabel('Aux Loss')
ax2.set_xlabel('Training Steps')
ax2.legend()
ax2.grid(True)

# Plot MaxVio
ax3.plot(auxfree_records[:,5], label='Aux-Loss-Free', alpha=0.7)
ax3.plot(bal_records[:,5], label='Balance Loss', alpha=0.7)
ax3.plot(both_records[:,5], label='Both Methods', alpha=0.7)
ax3.set_ylabel('MaxVio')
ax3.set_xlabel('Training Steps')
ax3.legend()
ax3.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('loss_comparison.png')
plt.close() 