import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Data
try:
    df = pd.read_csv('hpipe_experiment_data.csv')
except FileNotFoundError:
    print("Error: hpipe_experiment_data.csv not found. Please run hpipe_experiment.py first.")
    exit()

# Convert seconds to ms
df['Baseline_ms'] = df['Baseline'] * 1000
df['Hpipe_ms'] = df['Hpipe'] * 1000

# Setup Plot
plt.figure(figsize=(10, 6))
plt.title('Resilience Test: Dynamic Pipeline Reconfiguration (ResNet50)', fontsize=14, pad=20)
plt.xlabel('Batch Index (Time)', fontsize=12)
plt.ylabel('Batch Latency (ms)', fontsize=12)

# Plot Lines
# Baseline: Red dashed line
plt.plot(df['Batch'], df['Baseline_ms'], label='Baseline (Stop-and-Redeploy)', 
         color='#d62728', linestyle='--', linewidth=2, alpha=0.8)

# Hpipe: Blue solid line
plt.plot(df['Batch'], df['Hpipe_ms'], label='Hpipe (Instant Handoff)', 
         color='#1f77b4', linewidth=2.5)

# Highlight the Spike Region
spike_val = df.loc[60, 'Baseline_ms']
plt.annotate(f'Downtime Spike\n{spike_val:.0f} ms', xy=(60, spike_val), xytext=(70, spike_val + 200),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='red')

# Highlight Hpipe smoothness
hpipe_val = df.loc[60, 'Hpipe_ms']
plt.annotate(f'Seamless\n{hpipe_val:.0f} ms', xy=(60, hpipe_val), xytext=(45, hpipe_val - 300),
             arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=10, color='blue')

# Add region annotations
plt.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
plt.text(25, 2000, 'Phase 1: Stable', ha='center', alpha=0.6)
plt.text(55, 2000, 'Phase 2:\nInterference', ha='center', color='orange', fontweight='bold')
plt.text(80, 2000, 'Phase 3: Recovery', ha='center', alpha=0.6)

plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('hpipe_resilience.png', dpi=300)
print("Saved hpipe_resilience.png")
