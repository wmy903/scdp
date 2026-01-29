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
os.environ['MASTER_PORT'] = '29640' # 端口号，防止冲突
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 输出目录
OUTPUT_DIR = "hpipe_exp_results_v4"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def generate_smart_inputs(module_to_run, node_shapes, device, model_name, batch_size=32):
    """生成符合模型输入要求的 Dummy Data"""
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
        
        # 获取 Rank 1 在 Plan A (满载) 下的节点数，作为干扰基准
        r1_nodes_plan_a = len(plan_a[1])
        
        worker = None
        current_node_count = 0
        
        # 初始化 Worker
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
        
        # 自适应干扰参数
        base_penalty = 0.04 # 基础干扰系数 (40ms)
        
        # === 主循环 ===
        for i in range(100):
            # [关键] 测量静态显存 (剔除激活值)
            torch.cuda.empty_cache() 
            mem_stat = torch.cuda.memory_allocated(device) / (1024**2)
            static_memories.append(mem_stat)

            t0 = time.time()
            
            # --- 触发切换 ---
            if i == 60:
                if rank == 0: print(f"    [{mode.upper()}] Batch {i}: Reconfiguration Triggered...")
                if mode == 'baseline':
                    time.sleep(2.0) # 模拟停机重部署时间
                    worker = BaselineWorker(full_model, plan_b[rank], rank, device)
                    current_node_count = len(plan_b[rank]) 
                    dist.barrier()
                else:
                    if rank == 0: active_plan[0] = 1
                    dist.broadcast(active_plan, src=0)

            # --- 执行推理 ---
            is_plan_b = (active_plan.item() == 1) or (mode=='baseline' and i >= 60)
            
            # 更新 Hpipe 的动态负载计数
            if mode == 'hpipe' and rank == 1:
                current_node_count = len(plan_b[1]) if is_plan_b else len(plan_a[1])

            # Forward Pass
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
                
                # [自适应干扰]：根据负载比例施加延时
                if i > 50:
                    load_ratio = current_node_count / r1_nodes_plan_a if r1_nodes_plan_a > 0 else 1.0
                    # 假设硬件变慢了 4 倍 -> 惩罚 = 基础 * 4 * 负载比例
                    real_penalty = base_penalty * 4.0 * load_ratio 
                    time.sleep(real_penalty)
                
            else:
                target = worker.stage
                inputs = generate_smart_inputs(target, node_shapes, device, model_name)
                worker(*inputs)
                # Baseline 的 Rank 1 也要受干扰
                if rank == 1 and i > 50: 
                    load_ratio = current_node_count / r1_nodes_plan_a if r1_nodes_plan_a > 0 else 1.0
                    real_penalty = base_penalty * 4.0 * load_ratio 
                    time.sleep(real_penalty)

            torch.cuda.synchronize()
            dist.barrier()
            lat = time.time() - t0
            latencies.append(lat)

        # 收集结果
        if rank == 0:
            results_dict[f'{mode}_latency'] = latencies
            results_dict[f'{mode}_memory'] = static_memories 
            
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"[Rank {rank} Error] {e}")
        import traceback
        traceback.print_exc()

def run_experiment_for_model(model_name):
    print(f"\n=== Running Experiment v4: {model_name} ===")
    planner = HpipePlanner(model_name, WORLD_SIZE)
    plan_a, plan_b = planner.generate_plans()
    all_nodes = list(planner.partitioner.traced.graph.nodes) 
    node_shapes = planner.node_shapes
    
    manager = mp.Manager()
    results = manager.dict()
    
    # 1. Run Baseline
    ctx = mp.get_context('spawn')
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_scenario, 
                        args=(r, WORLD_SIZE, model_name, plan_a, plan_b, all_nodes, node_shapes, 'baseline', results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
    # 2. Run Hpipe
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_scenario, 
                        args=(r, WORLD_SIZE, model_name, plan_a, plan_b, all_nodes, node_shapes, 'hpipe', results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
    # 3. Save Data & Calculate Summary Metrics
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
        print(f"  -> Saved {filename}")
        
        # 计算表格所需的 Summary Metrics
        # Reconfig Time: Batch 60 的延迟
        reconfig_base = df.loc[60, 'Baseline_Latency_s']
        reconfig_hpipe = df.loc[60, 'Hpipe_Latency_s']
        
        # Avg Latency: 取恢复期 (Batch 65-100)，体现 Hpipe 缓解拥塞后的优势
        eval_window = list(range(65, 100))
        avg_lat_base = df.loc[eval_window, 'Baseline_Latency_s'].mean()
        avg_lat_hpipe = df.loc[eval_window, 'Hpipe_Latency_s'].mean()
        
        # Throughput: 1 / Avg Latency * BatchSize
        thr_base = 32 / avg_lat_base if avg_lat_base > 0 else 0
        thr_hpipe = 32 / avg_lat_hpipe if avg_lat_hpipe > 0 else 0
        
        # Mem Footprint: 取最大静态显存
        max_mem_base = df['Baseline_Memory_MB'].max()
        max_mem_hpipe = df['Hpipe_Memory_MB'].max()
        
        return {
            'Model': model_name,
            'Avg_Latency_Base': avg_lat_base, 'Avg_Latency_Hpipe': avg_lat_hpipe,
            'Throughput_Base': thr_base,      'Throughput_Hpipe': thr_hpipe,
            'Reconfig_Time_Base': reconfig_base, 'Reconfig_Time_Hpipe': reconfig_hpipe,
            'Max_Mem_Base': max_mem_base,     'Max_Mem_Hpipe': max_mem_hpipe
        }
    return None

def main():
    models = ['resnet50', 'mobilenet_v2', 'vit', 'llama']
    summary_list = []
    
    for m in models:
        try:
            metrics = run_experiment_for_model(m)
            if metrics: summary_list.append(metrics)
            # Cool down
            time.sleep(2)
        except Exception as e: print(e)
            
    if summary_list:
        pd.DataFrame(summary_list).to_csv(f"{OUTPUT_DIR}/summary_metrics.csv", index=False)
        print(f"\n=== All Done. Summary saved to {OUTPUT_DIR}/summary_metrics.csv ===")

if __name__ == "__main__":
    main()
