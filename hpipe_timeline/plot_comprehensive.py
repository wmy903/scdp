import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 自动定位路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "hpipe_exp_results_v4")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "hpipe_v4_figures")

if not os.path.exists(DATA_DIR):
    print(f"[Error] Data not found at {DATA_DIR}")
    exit()
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

MODELS = ['resnet50', 'mobilenet_v2', 'vit', 'llama']
BATCH_SIZE = 32

def plot_complete_metrics(model_name):
    csv_path = os.path.join(DATA_DIR, f"{model_name}_timeline.csv")
    try:
        df = pd.read_csv(csv_path)
    except:
        return

    # 计算实时吞吐量
    df['Base_Thr'] = BATCH_SIZE / df['Baseline_Latency_s']
    df['Hpipe_Thr'] = BATCH_SIZE / df['Hpipe_Latency_s']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.15)
    batches = df['Batch']

    # --- 1. Latency (ms) ---
    ax1.plot(batches, df['Baseline_Latency_s']*1000, color='#d62728', linestyle='--', label='Baseline')
    ax1.plot(batches, df['Hpipe_Latency_s']*1000, color='#1f77b4', linestyle='-', label='Hpipe', linewidth=2)
    ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name}: Real-time Monitoring', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # 标注停机尖峰
    spike = df.loc[60, 'Baseline_Latency_s'] * 1000
    if spike > 500:
        ax1.annotate(f'Reconfig\n({spike:.0f} ms)', xy=(60, spike), xytext=(65, spike),
                     arrowprops=dict(facecolor='red', shrink=0.05), color='#d62728')

    # --- 2. Throughput (Samples/s) ---
    ax2.plot(batches, df['Base_Thr'], color='#d62728', linestyle='--', label='Baseline')
    ax2.plot(batches, df['Hpipe_Thr'], color='#1f77b4', linestyle='-', label='Hpipe', linewidth=2)
    ax2.set_ylabel('Throughput (img/s)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # 标注回升优势
    end_base = df['Base_Thr'].iloc[-5:].mean()
    end_hpipe = df['Hpipe_Thr'].iloc[-5:].mean()
    if end_base > 0:
        gain = (end_hpipe - end_base) / end_base * 100
        ax2.annotate(f'+{gain:.0f}% Gain', xy=(90, end_hpipe), xytext=(90, end_hpipe + (end_hpipe*0.2)),
                     ha='center', color='#1f77b4', fontweight='bold', arrowprops=dict(arrowstyle='->', color='#1f77b4'))

    # --- 3. Static Memory (MB) ---
    ax3.plot(batches, df['Baseline_Memory_MB'], color='#d62728', linestyle='--', label='Baseline')
    ax3.plot(batches, df['Hpipe_Memory_MB'], color='#1f77b4', linestyle='-', label='Hpipe')
    ax3.set_ylabel('Static Memory (MB)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Batch Index', fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # 标注显存阶梯
    mem_before = df['Baseline_Memory_MB'].iloc[55]
    mem_after = df['Baseline_Memory_MB'].iloc[65]
    if mem_after > mem_before + 0.1:
        ax3.annotate('Reload Weights', xy=(60, (mem_before+mem_after)/2), xytext=(45, mem_after),
                     arrowprops=dict(facecolor='black', shrink=0.05), ha='center', fontsize=9)

    # 公共区域
    for ax in [ax1, ax2, ax3]:
        ax.axvspan(50, 100, color='orange', alpha=0.1, label='Interference')
        ax.axvline(60, color='black', linestyle=':', alpha=0.5)

    save_path = os.path.join(OUTPUT_DIR, f"{model_name}_complete.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {save_path}")

if __name__ == "__main__":
    for m in MODELS: plot_complete_metrics(m)
