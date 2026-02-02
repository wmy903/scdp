import torch
import torch.fx
import numpy as np
import time
from tabulate import tabulate 

# 引用现有模块
from scdp_core import (
    GraphPartitioner, Coarsener, DPPartitioner, 
    AdapipePartitioner, PicoPartitioner, DagPPartitioner, 
    AlgoGraph
)
from runtime_utils import get_model_and_input, profile_model

# === 配置修改 ===
# [修改] 将 Batch Size 调整为 64，测试高负载下的算法表现
BATCH_SIZE = 64
WORLD_SIZE = 4
# 模拟通信带宽 (10 GB/s)
COMM_BETA = 1.0 / (10 * 1024**3 / 1000.0) 

def warning_message():
    print("\n" + "!"*80)
    print(f"[实验配置更新] 当前 Batch Size = {BATCH_SIZE}")
    print("[算法说明]")
    print("1. Optimal: 理论上限 (无粗化 Full DP, 无噪声)。")
    print("2. SCDP (Clean): 拥有准确Profiler，且针对小模型(MobileNet/ViT)禁用了强制合并。")
    print("3. Baselines (Clean): 拥有准确Profiler的基线算法 (忽略通信代价)。")
    print("4. Baselines (Noisy): 模拟预测误差场景 (30%/40%/50% 噪声)。")
    print("!"*80 + "\n")

def inject_noise(profile_data, noise_ratio):
    """
    给算子性能数据注入随机噪声
    """
    noisy_data = {}
    for name, cost in profile_data.items():
        factor = 1.0 + np.random.uniform(-noise_ratio, noise_ratio)
        noisy_data[name] = max(0.001, cost * factor)
    return noisy_data

class PlanEvaluator:
    def __init__(self, algo_graph: AlgoGraph, comm_beta):
        self.graph = algo_graph # 这是一个包含真实(Clean)权重的图
        self.beta = comm_beta

    def evaluate(self, plan):
        results = {}
        node_to_stage = {}
        for stage_id, nodes in plan.items():
            for node_name in nodes:
                node_to_stage[node_name] = stage_id
        
        for stage_id in range(len(plan)):
            stage_nodes = plan[stage_id]
            comp_cost = 0.0
            for nid in stage_nodes:
                if nid in self.graph.nodes:
                    comp_cost += self.graph.nodes[nid].compute_cost 
            
            comm_bytes = 0.0
            for nid in stage_nodes:
                if nid not in self.graph.adj: continue
                for neighbor in self.graph.adj[nid]:
                    if neighbor not in node_to_stage: continue 
                    target_stage = node_to_stage[neighbor]
                    # 计算跨 Stage 的通信开销
                    if target_stage == stage_id + 1:
                        edge_key = (nid, neighbor)
                        if edge_key in self.graph.edges:
                            comm_bytes += self.graph.edges[edge_key].tensor_bytes
            
            comm_cost = comm_bytes * self.beta
            results[stage_id] = {
                'compute': comp_cost,
                'comm': comm_cost,
                'total': comp_cost + comm_cost
            }
        return results

def get_adaptive_scdp_params(model_name, total_nodes):
    name = model_name.lower()
    # 返回 (max_nodes, force_no_limit)
    if 'mobilenet' in name:
        return {'max_nodes': total_nodes + 10, 'k_stages': WORLD_SIZE}, True
    elif 'vit' in name:
        return {'max_nodes': total_nodes + 10, 'k_stages': WORLD_SIZE}, True
    elif 'resnet' in name:
        return {'max_nodes': 150, 'k_stages': WORLD_SIZE}, False
    else: 
        return {'max_nodes': WORLD_SIZE * 10, 'k_stages': WORLD_SIZE}, False

def run_test():
    warning_message()
    
    models = ['resnet50', 'mobilenet_v3_large', 'vit', 'llama']
    # models = ['llama'] # 调试用
    summary_table = []

    for model_name in models:
        print(f"\n{'='*20} 正在处理模型: {model_name} (BS={BATCH_SIZE}) {'='*20}")
        
        # 1. 获取 Ground Truth (真实) Profile
        try:
            # get_model_and_input 会接收 batch_size 参数，从而生成 BS=64 的 Input
            model, inp = get_model_and_input(model_name, batch_size=BATCH_SIZE)
            # Profile 结果会自然反映 BS=64 的计算量
            clean_costs = profile_model(model, inp, device='cuda:0')
        except Exception as e:
            print(f"Profiler Error: {e}")
            print("可能是 BS=64 导致显存不足，或者 int() 转换问题未修复。")
            continue
        
        # 构建真实图 (用于 Evaluator 和 Optimal/SCDP)
        # 注意：GraphPartitioner 也会接收 BS，用于正确计算 Tensor Size (通信量)
        gp_clean = GraphPartitioner(model, clean_costs, WORLD_SIZE, batch_size=BATCH_SIZE)
        full_graph_clean = gp_clean._build()
        total_nodes = len(full_graph_clean.nodes)
        total_latency = sum(n.compute_cost for n in full_graph_clean.nodes.values())
        
        print(f"   [基础统计] 模型: {model_name}")
        print(f"   [基础统计] 总算子数量: {total_nodes}")
        print(f"   [基础统计] 总推理时延: {total_latency:.2f} ms")
        
        # 2. 生成带噪声的 Costs
        costs_noise_30 = inject_noise(clean_costs, 0.3)
        costs_noise_40 = inject_noise(clean_costs, 0.4)
        costs_noise_50 = inject_noise(clean_costs, 0.5)

        evaluator = PlanEvaluator(full_graph_clean, COMM_BETA)
        sim_results = {}

        print(">> 开始运行算法...")

        # === 1. Optimal (Full DP, Clean) ===
        opt_plan, _ = DPPartitioner(full_graph_clean, WORLD_SIZE).solve()
        sim_results['Optimal'] = evaluator.evaluate(opt_plan)

        # === 2. SCDP (Adaptive, Clean) ===
        scdp_params, force_no_limit = get_adaptive_scdp_params(model_name, total_nodes)
        coarsener = Coarsener(full_graph_clean, clean_costs, model_arch=('transformer' if 'vit' in model_name or 'llama' in model_name else 'cnn'))
        coarsened_graph = coarsener.coarsen(
            max_nodes=scdp_params['max_nodes'], 
            k_stages=scdp_params['k_stages'],
            force_no_limit=force_no_limit 
        )
        scdp_plan, _ = DPPartitioner(coarsened_graph, WORLD_SIZE).solve()
        sim_results['SCDP(Clean)'] = evaluator.evaluate(scdp_plan)

        # === 3. Baselines (Clean) ===
        adapipe_c_plan, _ = AdapipePartitioner(gp_clean.traced, clean_costs, WORLD_SIZE).solve()
        sim_results['AdaPipe(Clean)'] = evaluator.evaluate(adapipe_c_plan)
        
        pico_c_plan, _ = PicoPartitioner(gp_clean.traced, clean_costs, WORLD_SIZE).solve()
        sim_results['Pico(Clean)'] = evaluator.evaluate(pico_c_plan)
        
        dagp_c_plan, _ = DagPPartitioner(gp_clean.traced, clean_costs, WORLD_SIZE).solve()
        sim_results['DagP(Clean)'] = evaluator.evaluate(dagp_c_plan)

        # === 4. Baselines (Noisy) ===
        adapipe_n_plan, _ = AdapipePartitioner(gp_clean.traced, costs_noise_30, WORLD_SIZE).solve()
        sim_results['AdaPipe(N30%)'] = evaluator.evaluate(adapipe_n_plan)

        pico_n_plan, _ = PicoPartitioner(gp_clean.traced, costs_noise_40, WORLD_SIZE).solve()
        sim_results['Pico(N40%)'] = evaluator.evaluate(pico_n_plan)

        dagp_n_plan, _ = DagPPartitioner(gp_clean.traced, costs_noise_50, WORLD_SIZE).solve()
        sim_results['DagP(N50%)'] = evaluator.evaluate(dagp_n_plan)

        # --- 打印结果 ---
        print(f"\n   [模型: {model_name}] 流水线阶段详情 (Clean Latency, ms) | BS={BATCH_SIZE}:")
        headers = ["算法(配置)", "S0", "S1", "S2", "S3", "Cycle(Max)", "Imbal"]
        rows = []
        
        order = [
            'Optimal', 'SCDP(Clean)', 
            'AdaPipe(Clean)', 'AdaPipe(N30%)',
            'Pico(Clean)', 'Pico(N40%)',
            'DagP(Clean)', 'DagP(N50%)'
        ]
        
        for algo in order:
            if algo not in sim_results: continue
            res = sim_results[algo]
            stages_lat = [res[i]['total'] for i in range(WORLD_SIZE)]
            bottleneck = max(stages_lat)
            avg_lat = sum(stages_lat) / WORLD_SIZE
            imbalance = bottleneck / avg_lat if avg_lat > 0 else 0
            
            row = [algo]
            row.extend([f"{x:.2f}" for x in stages_lat])
            row.append(f"{bottleneck:.2f}")
            row.append(f"{imbalance:.2f}x")
            rows.append(row)
            
            summary_table.append([model_name, algo, bottleneck, imbalance])

        try:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        except:
            print("\t".join(headers))
            for r in rows: print("\t".join(map(str, r)))

    print(f"\n{'='*20} 最终对比汇总 (Cycle Time) | BS={BATCH_SIZE} {'='*20}")
    print(f"{'Model':<18} | {'Algo(Config)':<15} | {'Cycle(ms)':<10} | {'Imbalance':<10}")
    print("-" * 65)
    for row in summary_table:
        print(f"{row[0]:<18} | {row[1]:<15} | {row[2]:<10.2f} | {row[3]:<10.2f}")

if __name__ == "__main__":
    run_test()
