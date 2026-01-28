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

# === Config ===
WORLD_SIZE = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29630' # Port bump
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

OUTPUT_DIR = "hpipe_exp_results_v4"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def generate_smart_inputs(module_to_run, node_shapes, device, model_name, batch_size=32):
    # (Same as before)
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
        
        # --- Metrics for scaling interference ---
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
        static_memories = [] # New Metric
        
        # For Adaptive Interference Calculation
        warmup_latencies = []
        base_penalty = 0.05 # Default fallback
        
        # --- Run Loop ---
        for i in range(100):
            # 1. Measure Static Memory (Before Forward)
            # This captures Weights + Context, ignoring Activations
            torch.cuda.empty_cache() 
            mem_stat = torch.cuda.memory_allocated(device) / (1024**2)
            static_memories.append(mem_stat)

            t0 = time.time()
            
            # --- Trigger Logic ---
            if i == 60:
                if rank == 0: print(f"    [{mode.upper()}] Batch {i}: Reconfiguration...")
                if mode == 'baseline':
                    # Simulate downtime + reload
                    time.sleep(1.0) # Reduced from 2.0 to be kinder
                    worker = BaselineWorker(full_model, plan_b[rank], rank, device)
                    current_node_count = len(plan_b[rank]) 
                    dist.barrier()
                else:
                    if rank == 0: active_plan[0] = 1
                    dist.broadcast(active_plan, src=0)

            # --- Execution Logic ---
            is_plan_b = (active_plan.item() == 1) or (mode=='baseline' and i >= 60)
            
            if mode == 'hpipe' and rank == 1:
                current_node_count = len(plan_b[1]) if is_plan_b else len(plan_a[1])

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
                
                # [ADAPTIVE INTERFERENCE]
                # Apply 4x slowdown based on observed base latency
                if i > 50:
                    load_ratio = current_node_count / r1_nodes_plan_a if r1_nodes_plan_a > 0 else 1.0
                    real_penalty = base_penalty * 4.0 * load_ratio 
                    time.sleep(real_penalty)
                
            else:
                target = worker.stage
                inputs = generate_smart_inputs(target, node_shapes, device, model_name)
                worker(*inputs)
                if rank == 1 and i > 50: 
                    load_ratio = current_node_count / r1_nodes_plan_a if r1_nodes_plan_a > 0 else 1.0
                    real_penalty = base_penalty * 4.0 * load_ratio 
                    time.sleep(real_penalty)

            torch.cuda.synchronize()
            dist.barrier()
            lat = time.time() - t0
            latencies.append(lat)
            
            # Calibrate Base Penalty during warmup (Batch 10-40)
            if 10 <= i < 40 and rank == 1:
                warmup_latencies.append(lat)
            if i == 40 and rank == 1 and warmup_latencies:
                base_penalty = sum(warmup_latencies) / len(warmup_latencies)
                # Cap minimum to avoid almost-zero sleep
                if base_penalty < 0.005: base_penalty = 0.005

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
    
    # Run Baseline
    ctx = mp.get_context('spawn')
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_scenario, 
                        args=(r, WORLD_SIZE, model_name, plan_a, plan_b, all_nodes, node_shapes, 'baseline', results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
    # Run Hpipe
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_scenario, 
                        args=(r, WORLD_SIZE, model_name, plan_a, plan_b, all_nodes, node_shapes, 'hpipe', results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
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
        print(f"Saved {filename}")
    return None

def main():
    models = ['resnet50', 'mobilenet_v2', 'vit', 'llama']
    for m in models:
        try:
            run_experiment_for_model(m)
            time.sleep(2)
        except Exception as e: print(e)

if __name__ == "__main__":
    main()
