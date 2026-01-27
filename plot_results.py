import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# 读取数据
df = pd.read_csv('experiment_results.csv')

# 定义四个关键指标
metrics = [
    'Cycle Time (ms)', 
    'Total Latency (ms)', 
    'Bubble Rate (%)', 
    'Throughput (img/s)'
]

# 定义算法颜色映射，保证所有图中颜色一致
# 您可以根据论文配色方案修改这里
palette = {
    'baseline': '#95a5a6',  # 灰色
    'scdp': '#3498db',      # 蓝色
    'optimal': '#e74c3c'    # 红色
}

# 获取所有模型列表
models = df['Model'].unique()

for model in models:
    # 筛选当前模型的数据
    model_data = df[df['Model'] == model]
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Performance Metrics for {model}', fontsize=16, fontweight='bold', y=1.02)
    
    # 扁平化 axes 方便遍历
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 绘制柱状图
        sns.barplot(
            data=model_data, 
            x='Algorithm', 
            y=metric, 
            hue='Algorithm', 
            palette=palette, 
            ax=ax,
            edgecolor='black', # 给柱子加黑边，适合黑白打印
            dodge=False # 不需要偏移，因为x轴就是分类依据
        )
        
        # 优化标签和标题
        ax.set_title(metric, fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel(metric)
        
        # 在柱子上标注数值
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=11)
            
        # 移除图例（因为x轴已经很清楚了）
        # 安全地移除图例：如果图例存在才移除
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        # 微调Y轴范围，让柱子不要顶到头
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] * 1.15)

    plt.tight_layout()
    
    # 保存图片
    filename = f'{model}_performance.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved: {filename}")
    
    plt.show() # 如果在 notebook 中运行，这行会显示图片
