import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

OUTPUT_DIR = "hpipe_exp_results"

def plot_timeline(csv_file):
    model_name = os.path.basename(csv_file).replace('_timeline.csv', '')
    df = pd.read_csv(csv_file)
    
    # 1. Latency Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Batch'], df['Baseline_Latency_s']*1000, label='Baseline', color='red', linestyle='--')
    plt.plot(df['Batch'], df['Hpipe_Latency_s']*1000, label='Hpipe', color='blue')
    plt.title(f'{model_name} Latency Timeline')
    plt.ylabel('Latency (ms)')
    plt.xlabel('Batch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_latency.png")
    plt.close()
    
    # 2. Memory Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Batch'], df['Baseline_Memory_MB'], label='Baseline', color='red', linestyle='--')
    plt.plot(df['Batch'], df['Hpipe_Memory_MB'], label='Hpipe (Rank 0)', color='blue')
    plt.title(f'{model_name} Memory Timeline (Rank 0)')
    plt.ylabel('Memory (MB)')
    plt.xlabel('Batch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_memory.png")
    plt.close()

def plot_summary():
    try:
        df = pd.read_csv(f"{OUTPUT_DIR}/summary_metrics.csv")
    except:
        return

    models = df['Model']
    x = range(len(models))
    width = 0.35
    
    # 1. Reconfig Time Bar Chart
    plt.figure(figsize=(8, 6))
    plt.bar([i - width/2 for i in x], df['Reconfig_Time_Base']*1000, width, label='Baseline', color='red', alpha=0.7)
    plt.bar([i + width/2 for i in x], df['Reconfig_Time_Hpipe']*1000, width, label='Hpipe', color='blue', alpha=0.7)
    plt.xticks(x, models)
    plt.ylabel('Reconfiguration Time (ms)')
    plt.title('Reconfiguration Overhead Comparison')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/summary_reconfig_time.png")
    plt.close()

    # 2. Memory Overhead
    plt.figure(figsize=(8, 6))
    plt.bar([i - width/2 for i in x], df['Max_Mem_Base'], width, label='Baseline', color='red', alpha=0.7)
    plt.bar([i + width/2 for i in x], df['Max_Mem_Hpipe'], width, label='Hpipe', color='blue', alpha=0.7)
    plt.xticks(x, models)
    plt.ylabel('Peak Memory (MB)')
    plt.title('Memory Overhead Comparison')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/summary_memory.png")
    plt.close()

if __name__ == "__main__":
    files = glob.glob(f"{OUTPUT_DIR}/*_timeline.csv")
    for f in files:
        plot_timeline(f)
        print(f"Plots generated for {f}")
    
    plot_summary()
    print("Summary plots generated.")
