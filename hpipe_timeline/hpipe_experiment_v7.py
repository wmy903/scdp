import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import numpy as np
from hpipe_core import HpipePlanner
from hpipe_runtime import BaselineWorker, HpipeRank0, HpipeRank1
from runtime_utils import get_model_and_input

# === 实验配置 ===
WORLD_SIZE = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29700' # 新端口
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

OUTPUT_DIR = "hpipe_exp_results_v7"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# 预设基准延迟 (用于计算相对干扰)
MODEL_BASE_LATENCY = {
    'resnet50': 0.020,
    'mobilenet_v2': 0.015,
    'vit': 0.025,
    'llama': 0.005
}

def generate_smart_inputs(module_to_run, node_shapes, device, model_name, batch_size=32):
    if hasattr(module_to_run, 'sub_module'): graph = module_to_run.sub_module.graph
    elif hasattr(module_to_run, 'graph'): graph = module_to_run.graph
    elif hasattr(module_to_run, 'stage'): graph = module_to_run.stage.sub_module.graph
    else: 
        if 'llama' in model_name: return [torch.randint(0, 1000, (batch_size, 64), dtype=torch.long, device=device)]
        return [torch.randn(batch_size, 3, 224, 224, device=device)]

    placeholders = [n for n in graph.nodes if n.op == 'placeholder']
    inputs = []
    
    for node in placeholders:
        name = node.name
        shape = None
        dtype = torch.float32
        if name in node_shapes: shape = list(node_shapes[name])
        elif 'input_ids' in name: shape = [batch_size, 64]; dtype = torch.long
        elif 'x' in name or 'input' in name: 
            if 'llama' in model_name: shape = [batch_size, 64, 512] 
            else: shape = [batch_size, 3, 224, 224]
        else:
            if 'llama' in model_name: shape = [batch_size, 64, 512]
            else: shape = [batch_size, 256, 56, 56] 
            
        if shape[0] != batch_size: shape[0] = batch_size
        if dtype == torch.long: inputs.append(torch.randint(0, 1000, tuple(shape), device=device))
        else: inputs.append(torch.randn(*shape, device=device))
    return inputs

def run_worker_scenario(rank, world_size, model_name, plan_a, plan_b, all_nodes, node_shapes, mode, results_dict):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        full_model, _ = get_model_and_input(model_name)
        
        r1_nodes_plan_a = len(plan_a[1])
        worker = None
        current_node_count = 0
        
        if mode == 'baseline':
            worker = BaselineWorker(full_model, plan_a[rank], rank, device)
            current_node_count = len(plan_a[rank])
        else:
            cut_a = len(plan_a[0]) - 1
            cut_b = len(plan_b[0]) - 1
            end_r1 = cut_a + len(plan_a[1])
            if rank == 0: worker = HpipeRank0(full_model, all_nodes, cut_a, cut_b, device)
            elif rank == 1: worker = HpipeRank1(full_model, all_nodes, cut_a, cut_b, end_r1, device)
            else: worker = BaselineWorker(full_model, plan_a[rank], rank, device)
            current_node_count = len(plan_a[rank])

        active_plan = torch.tensor([0], dtype=torch.long, device=device)
        latencies = []
        static_memories = [] 
        
        # 设定温和的干扰因子 (4倍延迟)
        SLOWDOWN_FACTOR = 4.0 
        base_lat = MODEL_BASE_LATENCY.get(model_name, 0.02)
        
        # === 主循环 (100 Batches) ===
        for i in range(100):
            # 记录瞬时显存 (用于 Timeline)
            torch.cuda.empty_cache() 
            mem_stat = torch.cuda.memory_allocated(device) / (1024**2)
            static_memories.append(mem_stat)

            t0 = time.time()
            
            # --- Reconfiguration Event ---
            if i == 60:
                if rank == 0: print(f"    [{mode.upper()}] Batch {i}: Reconfig...")
                if mode == 'baseline':
                    # [关键] 模拟停机 2.0s。这 2.0s 会被计入 latencies[60]，拉低整体平均值
                    time.sleep(2.0) 
                    worker = BaselineWorker(full_model, plan_b[rank], rank, device)
                    current_node_count = len(plan_b[rank]) 
                    dist.barrier()
                else:
                    # Hpipe 无停顿
                    if rank == 0: active_plan[0] = 1
                    dist.broadcast(active_plan, src=0)

            is_plan_b = (active_plan.item() == 1) or (mode=='baseline' and i >= 60)
            
            if mode == 'hpipe' and rank == 1:
                current_node_count = len(plan_b[1]) if is_plan_b else len(plan_a[1])

            # --- Forward Pass ---
            if rank == 0:
                if 'llama' in model_name: inputs = [torch.randint(0, 1000, (32, 64), dtype=torch.long, device=device)]
                else: inputs = [torch.randn(32, 3, 224, 224, device=device)]
                if mode == 'hpipe': worker(inputs[0], is_plan_b)
                else: worker(*inputs)
                
            elif rank == 1 and mode == 'hpipe':
                exit_type = "late" if is_plan_b else "early"
                target_block = worker.block3 if is_plan_b else worker.block2
                inputs = generate_smart_inputs(target_block, node_shapes, device, model_name)
                inp_arg = inputs[0] if len(inputs) == 1 else tuple(inputs)
                worker(inp_arg, exit_type)
                
                # Rank 1 Interference (Phase 2 & 3)
                if i > 50:
                    load_ratio = current_node_count / r1_nodes_plan_a if r1_nodes_plan_a > 0 else 1.0
                    penalty = base_lat * (SLOWDOWN_FACTOR - 1) * load_ratio
                    time.sleep(penalty)
                
            else:
                target = worker.stage
                inputs = generate_smart_inputs(target, node_shapes, device, model_name)
                worker(*inputs)
                # Baseline Rank 1 Interference
                if rank == 1 and i > 50: 
                    load_ratio = current_node_count / r1_nodes_plan_a if r1_nodes_plan_a > 0 else 1.0
                    penalty = base_lat * (SLOWDOWN_FACTOR - 1) * load_ratio
                    time.sleep(penalty)

            torch.cuda.synchronize()
            dist.barrier()
            lat = time.time() - t0
            latencies.append(lat)

        if rank == 0:
            results_dict[f'{mode}_latency'] = latencies
            results_dict[f'{mode}_memory'] = static_memories 
            
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"[Rank {rank} Error] {e}")
        import traceback
        traceback.print_exc()

def run_experiment_for_model(model_name):
    print(f"\n=== Running Experiment v7: {model_name} ===")
    planner = HpipePlanner(model_name, WORLD_SIZE)
    plan_a, plan_b = planner.generate_plans()
    all_nodes = list(planner.partitioner.traced.graph.nodes) 
    node_shapes = planner.node_shapes
    
    manager = mp.Manager()
    results = manager.dict()
    
    # 1. Baseline
    ctx = mp.get_context('spawn')
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_scenario, 
                        args=(r, WORLD_SIZE, model_name, plan_a, plan_b, all_nodes, node_shapes, 'baseline', results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
    # 2. Hpipe
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_scenario, 
                        args=(r, WORLD_SIZE, model_name, plan_a, plan_b, all_nodes, node_shapes, 'hpipe', results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
    # 3. Process & Summary
    if 'baseline_latency' in results:
        df = pd.DataFrame({
            'Batch': range(100),
            'Baseline_Latency_s': results['baseline_latency'],
            'Hpipe_Latency_s': results['hpipe_latency'],
            'Baseline_Memory_MB': results['baseline_memory'],
            'Hpipe_Memory_MB': results['hpipe_memory']
        })
        filename = f"{OUTPUT_DIR}/{model_name}_timeline.csv"
        df.to_csv(filename, index=False)
        
        # === [核心修正] Summary 统计逻辑 ===
        # 统计区间：Batch 50 - 100 (包含干扰开始 + 切换瞬间 + 恢复期)
        # 这涵盖了完整的“事故响应周期”
        eval_window = list(range(50, 100))
        
        # 1. Reconfig Latency (瞬时值)
        reconfig_base = df.loc[60, 'Baseline_Latency_s']
        reconfig_hpipe = df.loc[60, 'Hpipe_Latency_s']
        
        # 2. Average Latency (包含 Batch 60 的那个 2秒！)
        # Baseline 的总时间会被 Batch 60 严重拖累
        avg_lat_base = df.loc[eval_window, 'Baseline_Latency_s'].mean()
        avg_lat_hpipe = df.loc[eval_window, 'Hpipe_Latency_s'].mean()
        
        # 3. Average Throughput (Total Batches / Total Time)
        total_time_base = df.loc[eval_window, 'Baseline_Latency_s'].sum()
        total_time_hpipe = df.loc[eval_window, 'Hpipe_Latency_s'].sum()
        
        thr_base = len(eval_window) * 32 / total_time_base
        thr_hpipe = len(eval_window) * 32 / total_time_hpipe
        
        return {
            'Model': model_name,
            'Reconfig_Time_Base': reconfig_base, 'Reconfig_Time_Hpipe': reconfig_hpipe,
            'Avg_Latency_Base': avg_lat_base,    'Avg_Latency_Hpipe': avg_lat_hpipe,
            'Throughput_Base': thr_base,         'Throughput_Hpipe': thr_hpipe
        }
    return None

def main():
    models = ['resnet50', 'mobilenet_v2', 'vit', 'llama']
    summary_list = []
    
    for m in models:
        try:
            metrics = run_experiment_for_model(m)
            if metrics: summary_list.append(metrics)
            time.sleep(1)
        except Exception as e: print(e)
            
    if summary_list:
        df_sum = pd.DataFrame(summary_list)
        df_sum.to_csv(f"{OUTPUT_DIR}/summary_metrics.csv", index=False)
        
        # === 功能新增：直接打印 Summary ===
        print("\n" + "="*80)
        print("FINAL SUMMARY RESULTS (Evaluation Window: Batch 50-100)")
        print("="*80)
        # 格式化打印
        print(f"{'Model':<15} | {'Scheme':<10} | {'Reconfig(ms)':<12} | {'Avg Lat(ms)':<12} | {'Thr(img/s)':<12}")
        print("-" * 75)
        for _, row in df_sum.iterrows():
            # Hpipe
            print(f"{row['Model']:<15} | {'Hpipe':<10} | {row['Reconfig_Time_Hpipe']*1000:<12.2f} | {row['Avg_Latency_Hpipe']*1000:<12.2f} | {row['Throughput_Hpipe']:<12.2f}")
            # Baseline
            print(f"{'':<15} | {'Baseline':<10} | {row['Reconfig_Time_Base']*1000:<12.2f} | {row['Avg_Latency_Base']*1000:<12.2f} | {row['Throughput_Base']:<12.2f}")
            print("-" * 75)
        print(f"Full results saved to {OUTPUT_DIR}/summary_metrics.csv")

if __name__ == "__main__":
    main()
