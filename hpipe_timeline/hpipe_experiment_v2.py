import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import numpy as np
import shutil
from hpipe_core import HpipePlanner
from hpipe_runtime import BaselineWorker, HpipeRank0, HpipeRank1
from runtime_utils import get_model_and_input

# === Config ===
WORLD_SIZE = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29621' # Port bump
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

# Output Directory
OUTPUT_DIR = "hpipe_exp_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def simulate_interference(rank, batch_idx, intensity_ms=100):
    """Simulate resource contention on Rank 1 between batch 50-60."""
    if rank == 1 and batch_idx > 50:
        time.sleep(intensity_ms / 1000.0)

def generate_smart_inputs(module_to_run, node_shapes, device, model_name, batch_size=32):
    """
    Generate appropriate dummy inputs based on module requirements.
    [FIX] Added model_name to handle Llama vs Vision fallback shapes.
    """
    if hasattr(module_to_run, 'sub_module'): graph = module_to_run.sub_module.graph
    elif hasattr(module_to_run, 'graph'): graph = module_to_run.graph
    elif hasattr(module_to_run, 'stage'): graph = module_to_run.stage.sub_module.graph
    else: 
        # Fallback for generic start
        if 'llama' in model_name:
            return [torch.randint(0, 1000, (batch_size, 64), dtype=torch.long, device=device)]
        return [torch.randn(batch_size, 3, 224, 224, device=device)]

    placeholders = [n for n in graph.nodes if n.op == 'placeholder']
    inputs = []
    
    for node in placeholders:
        name = node.name
        shape = None
        dtype = torch.float32
        
        # 1. Try Lookup
        if name in node_shapes:
            shape = list(node_shapes[name])
        
        # 2. Heuristics based on name
        elif 'input_ids' in name or 'ids' in name:
            shape = [batch_size, 64]
            dtype = torch.long
        elif 'x' in name or 'input' in name: 
            if 'llama' in model_name:
                # Llama hidden state input
                shape = [batch_size, 64, 512] 
            else:
                # Vision input
                shape = [batch_size, 3, 224, 224]
        else:
            # 3. Last Resort Fallback [CRITICAL FIX]
            if 'llama' in model_name:
                # Llama intermediate: [Batch, Seq, Hidden]
                shape = [batch_size, 64, 512]
            else:
                # Vision intermediate: [Batch, Channel, H, W]
                shape = [batch_size, 256, 56, 56] 
            
        if shape[0] != batch_size: shape[0] = batch_size
        
        if dtype == torch.long:
            inputs.append(torch.randint(0, 1000, tuple(shape), device=device))
        else:
            inputs.append(torch.randn(*shape, device=device))
        
    return inputs

def run_worker_scenario(rank, world_size, model_name, plan_a, plan_b, all_nodes, node_shapes, mode, results_dict):
    """
    Unified worker for both Baseline and Hpipe scenarios.
    mode: 'baseline' or 'hpipe'
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        full_model, _ = get_model_and_input(model_name)
        
        # --- Setup ---
        worker = None
        if mode == 'baseline':
            worker = BaselineWorker(full_model, plan_a[rank], rank, device)
        else:
            # Hpipe setup
            cut_a = len(plan_a[0]) - 1
            cut_b = len(plan_b[0]) - 1
            end_r1 = cut_a + len(plan_a[1])
            if rank == 0: worker = HpipeRank0(full_model, all_nodes, cut_a, cut_b, device)
            elif rank == 1: worker = HpipeRank1(full_model, all_nodes, cut_a, cut_b, end_r1, device)
            else: worker = BaselineWorker(full_model, plan_a[rank], rank, device)

        # Sync flags
        active_plan = torch.tensor([0], dtype=torch.long, device=device)
        
        # Data collectors
        latencies = []
        memories = [] # Peak memory per batch
        
        # --- Run Loop ---
        TOTAL_BATCHES = 100
        SWITCH_BATCH = 60
        
        for i in range(TOTAL_BATCHES):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats(device) 
            
            # --- Trigger Logic ---
            if i == SWITCH_BATCH:
                if rank == 0:
                    print(f"    [{mode.upper()}] Batch {i}: Triggering Reconfiguration...")
                
                if mode == 'baseline':
                    # Simulate Stop-the-world
                    time.sleep(2.0) 
                    # Re-deploy Plan B
                    worker = BaselineWorker(full_model, plan_b[rank], rank, device)
                    dist.barrier()
                else:
                    # Hpipe: Broadcast switch signal
                    if rank == 0: active_plan[0] = 1
                    dist.broadcast(active_plan, src=0)

            # --- Execution Logic ---
            is_plan_b = (active_plan.item() == 1) or (mode=='baseline' and i >= SWITCH_BATCH)
            
            # Input Generation & Forward
            if rank == 0:
                # Global Input
                if 'llama' in model_name: 
                    inputs = [torch.randint(0, 1000, (32, 64), dtype=torch.long, device=device)]
                else:
                    inputs = [torch.randn(32, 3, 224, 224, device=device)]
                
                if mode == 'hpipe': worker(inputs[0], is_plan_b)
                else: worker(*inputs)
                
            elif rank == 1 and mode == 'hpipe':
                exit_type = "late" if is_plan_b else "early"
                target_block = worker.block3 if is_plan_b else worker.block2
                
                # [FIX] Pass model_name
                inputs = generate_smart_inputs(target_block, node_shapes, device, model_name)
                
                inp_arg = inputs[0] if len(inputs) == 1 else tuple(inputs)
                worker(inp_arg, exit_type)
                simulate_interference(rank, i)
            else:
                # Generic worker (Baseline R1-R3, Hpipe R2-R3)
                target = worker.stage
                # [FIX] Pass model_name
                inputs = generate_smart_inputs(target, node_shapes, device, model_name)
                worker(*inputs)
                if rank == 1: simulate_interference(rank, i)

            torch.cuda.synchronize()
            dist.barrier()
            
            # Record Metrics
            lat = time.time() - t0
            mem = torch.cuda.max_memory_allocated(device) / (1024**2) # MB
            
            latencies.append(lat)
            memories.append(mem)
        
        # Save results to dict
        if rank == 0:
            results_dict[f'{mode}_latency'] = latencies
            results_dict[f'{mode}_memory'] = memories 
            
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"[Rank {rank} Error] {e}")
        import traceback
        traceback.print_exc()

def run_experiment_for_model(model_name):
    print(f"\n=== Running Experiment: {model_name} ===")
    
    # 1. Plan
    planner = HpipePlanner(model_name, WORLD_SIZE)
    plan_a, plan_b = planner.generate_plans()
    all_nodes = list(planner.partitioner.traced.graph.nodes) 
    node_shapes = planner.node_shapes
    
    manager = mp.Manager()
    results = manager.dict()
    
    # 2. Run Baseline
    ctx = mp.get_context('spawn')
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_scenario, 
                        args=(r, WORLD_SIZE, model_name, plan_a, plan_b, all_nodes, node_shapes, 'baseline', results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
    # 3. Run Hpipe
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_scenario, 
                        args=(r, WORLD_SIZE, model_name, plan_a, plan_b, all_nodes, node_shapes, 'hpipe', results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
    # 4. Save Timeline Data
    if 'baseline_latency' in results and 'hpipe_latency' in results:
        df = pd.DataFrame({
            'Batch': range(100),
            'Baseline_Latency_s': results['baseline_latency'],
            'Hpipe_Latency_s': results['hpipe_latency'],
            'Baseline_Memory_MB': results['baseline_memory'],
            'Hpipe_Memory_MB': results['hpipe_memory']
        })
        filename = f"{OUTPUT_DIR}/{model_name}_timeline.csv"
        df.to_csv(filename, index=False)
        print(f"  -> Saved timeline to {filename}")
        
        # 5. Compute Summary Metrics
        reconfig_base = df.loc[60, 'Baseline_Latency_s']
        reconfig_hpipe = df.loc[60, 'Hpipe_Latency_s']
        
        stable_idx = list(range(0, 50)) + list(range(65, 100))
        avg_lat_base = df.loc[stable_idx, 'Baseline_Latency_s'].mean()
        avg_lat_hpipe = df.loc[stable_idx, 'Hpipe_Latency_s'].mean()
        
        thr_base = 32 / avg_lat_base if avg_lat_base > 0 else 0
        thr_hpipe = 32 / avg_lat_hpipe if avg_lat_hpipe > 0 else 0
        
        max_mem_base = df['Baseline_Memory_MB'].max()
        max_mem_hpipe = df['Hpipe_Memory_MB'].max()
        
        return {
            'Model': model_name,
            'Avg_Latency_Base': avg_lat_base, 'Avg_Latency_Hpipe': avg_lat_hpipe,
            'Throughput_Base': thr_base,      'Throughput_Hpipe': thr_hpipe,
            'Reconfig_Time_Base': reconfig_base, 'Reconfig_Time_Hpipe': reconfig_hpipe,
            'Max_Mem_Base': max_mem_base,     'Max_Mem_Hpipe': max_mem_hpipe
        }
    else:
        print("  -> Experiment Failed (Missing results)")
        return None

def main():
    models = ['resnet50', 'mobilenet_v2', 'vit', 'llama']
    
    summary_list = []
    
    for m in models:
        try:
            metrics = run_experiment_for_model(m)
            if metrics:
                summary_list.append(metrics)
            time.sleep(2)
        except Exception as e:
            print(f"Error running {m}: {e}")
            
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv(f"{OUTPUT_DIR}/summary_metrics.csv", index=False)
        print(f"\n=== All Experiments Completed. Summary saved to {OUTPUT_DIR}/summary_metrics.csv ===")
        print(summary_df)

if __name__ == "__main__":
    main()
