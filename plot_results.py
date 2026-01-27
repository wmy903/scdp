import matplotlib.pyplot as plt
import numpy as np

# === 实验数据 (来自你的日志) ===
models = ['ResNet50', 'MobileNetV2', 'ViT', 'Llama']

# 数据格式: [Rank0, Rank1, Rank2, Rank3] (单位 ms)
baseline_lats = {
    'ResNet50': [7.02, 6.96, 6.47, 5.82],
    'MobileNetV2': [3.48, 2.81, 1.87, 1.65],
    'ViT': [5.55, 7.23, 7.20, 0.11],
    'Llama': [2.91, 2.70, 2.68, 2.48]
}

scdp_lats = {
    'ResNet50': [6.93, 6.75, 6.47, 6.25],
    'MobileNetV2': [3.30, 2.84, 1.81, 1.86],
    'ViT': [5.55, 4.72, 4.58, 4.89],
    'Llama': [2.42, 2.45, 2.85, 3.15]
}

# 吞吐量数据 (img/s)
baseline_thr = [4561.40, 9201.28, 4426.08, 10980.48]
scdp_thr = [4614.60, 9707.74, 5767.08, 10174.33]

def plot_latency_breakdown():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Load Balance Analysis: Latency per Rank (Lower Variance is Better)', fontsize=16)
    
    ranks = ['R0', 'R1', 'R2', 'R3']
    x = np.arange(len(ranks))
    width = 0.35
    
    for idx, model in enumerate(models):
        ax = axes[idx//2, idx%2]
        
        rects1 = ax.bar(x - width/2, baseline_lats[model], width, label='Baseline', color='#d62728', alpha=0.7)
        rects2 = ax.bar(x + width/2, scdp_lats[model], width, label='SCDP (Ours)', color='#1f77b4', alpha=0.8)
        
        ax.set_title(model)
        ax.set_ylabel('Latency (ms)')
        ax.set_xticks(x)
        ax.set_xticklabels(ranks)
        if idx == 0: ax.legend()
        
        # Add Cycle Time Line (Max Latency) representing the bottleneck
        max_base = max(baseline_lats[model])
        max_scdp = max(scdp_lats[model])
        ax.axhline(y=max_base, color='#d62728', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=max_scdp, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('exp_latency_breakdown.png', dpi=300)
    print("Saved exp_latency_breakdown.png")

def plot_throughput_speedup():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Calculate Speedup
    speedup = [s/b for s, b in zip(scdp_thr, baseline_thr)]
    
    # Plot bars
    bars = ax.bar(x, speedup, width=0.5, color=['#2ca02c' if s > 1 else 'gray' for s in speedup], alpha=0.8)
    
    # Baseline line
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Baseline')
    
    ax.set_ylabel('Normalized Throughput (vs Baseline)')
    ax.set_title('Throughput Improvement of SCDP')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.8, 1.4)
    
    # Add labels
    for bar, val in zip(bars, speedup):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('exp_throughput_speedup.png', dpi=300)
    print("Saved exp_throughput_speedup.png")

if __name__ == "__main__":
    plot_latency_breakdown()
    plot_throughput_speedup()
