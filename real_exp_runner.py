import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import pandas as pd  # 新增 pandas 用于保存 CSV
from datetime import timedelta
from torch.fx.passes.shape_prop import ShapeProp

from scdp_core import GraphPartitioner
from runtime_utils import get_model_and_input, PipelineStage, profile_model

# === Configuration ===
WORLD_SIZE = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29608' # Port incremented to avoid conflict

# === NCCL Settings ===
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_NET_GDR_LEVEL"] = "0"

# === Logger for saving results ===
class ExperimentLogger:
    def __init__(self, filename="experiment_results.csv"):
        self.filename = filename
        self.results = []

    def log(self, model, algo, plan, latencies, cycle_time, total_latency, bubble_rate, throughput):
        entry = {
            "Model": model,
            "Algorithm": algo,
            "Cycle Time (ms)": round(cycle_time, 2),
            "Total Latency (ms)": round(total_latency, 2),
            "Bubble Rate (%)": round(bubble_rate * 100, 2),
            "Throughput (img/s)": round(throughput, 2),
            "Latencies (ms)": str([round(l, 2) for l in latencies]), # Save as string
            "Partition Plan": str(plan)
        }
        self.results.append(entry)
        print(f"    [Logger] Data recorded for {model}-{algo}")

    def save(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.filename, index=False)
        print(f"\n>>> Results saved to {self.filename}")

# Global logger instance (will be passed to run_experiment)
logger = ExperimentLogger()

def profile_worker(model_name, output_queue):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = "cuda:0"
        model, sample_input = get_model_and_input(model_name)
        
        BATCH_SIZE = 32
        if sample_input.dtype == torch.long:
            sample_input = sample_input.repeat(BATCH_SIZE, 1)
        else:
            sample_input = sample_input.repeat(BATCH_SIZE, 1, 1, 1)
            
        print(f"  [Profiler] Profiling {model_name} on GPU 0...")
        costs = profile_model(model, sample_input, device=device)
        output_queue.put(costs)
    except Exception as e:
        print(f"  [Profiler] Error: {e}")
        output_queue.put({})

def run_worker_metrics(rank, world_size, partition_plan, node_shapes, model_name, return_dict):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
        
        full_model, _ = get_model_and_input(model_name)
        my_nodes = partition_plan[rank]
        if not my_nodes:
            dist.barrier()
            dist.destroy_process_group()
            return

        stage_module = PipelineStage(full_model, my_nodes, rank).to(device)
        
        input_nodes = [n for n in stage_module.sub_module.graph.nodes if n.op == 'placeholder']
        input_names = [n.name for n in input_nodes]
        
        BATCH_SIZE = 32
        real_inputs = []
        if rank == 0:
            _, dummy = get_model_and_input(model_name)
            if dummy.dtype == torch.long: real_inputs = [dummy.repeat(BATCH_SIZE, 1).to(device)]
            else: real_inputs = [dummy.repeat(BATCH_SIZE, 1, 1, 1).to(device)]
        else:
            for name in input_names:
                if name in node_shapes:
                    shape = list(node_shapes[name])
                    shape[0] = BATCH_SIZE
                    shape = tuple(shape)
                else:
                    if 'llama' in model_name: shape = (BATCH_SIZE, 64, 512) 
                    else: shape = (BATCH_SIZE, 64, 56, 56)
                
                if 'llama' in model_name and 'ids' in name:
                     real_inputs.append(torch.randint(0, 100, shape, device=device))
                else:
                     real_inputs.append(torch.randn(*shape, device=device))
        
        dist.barrier()
        with torch.no_grad(): stage_module(*real_inputs)
        torch.cuda.synchronize()
        dist.barrier()
        
        latencies = []
        NUM_BATCHES = 20
        for i in range(NUM_BATCHES):
            t0 = time.time()
            with torch.no_grad(): stage_module(*real_inputs)
            torch.cuda.synchronize()
            latencies.append(time.time() - t0)
            dist.barrier()
            
        avg_lat = sum(latencies) / len(latencies)
        return_dict[rank] = avg_lat
        dist.destroy_process_group()

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")

def run_experiment(model_name, algo):
    print(f"\n>>> [Experiment] Model: {model_name}, Algo: {algo}")
    
    # 1. Oracle Profiling
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p_prof = ctx.Process(target=profile_worker, args=(model_name, queue))
    p_prof.start()
    p_prof.join()
    real_costs = queue.get()
    
    if not real_costs:
        print("Profiling failed. Skipping.")
        return

    # 2. Partitioning
    model, sample_input = get_model_and_input(model_name)
    partitioner = GraphPartitioner(model, num_devices=WORLD_SIZE)
    ShapeProp(partitioner.traced).propagate(sample_input)
    
    node_shapes = {}
    for node in partitioner.traced.graph.nodes:
        if 'tensor_meta' in node.meta and hasattr(node.meta['tensor_meta'], 'shape'):
            node_shapes[node.name] = node.meta['tensor_meta'].shape

    if algo == 'scdp':
        plan = partitioner.run_scdp(profile_data=real_costs)
    elif algo == 'optimal':
        plan = partitioner.run_optimal(profile_data=real_costs)
    else:
        plan = partitioner.run_baseline(profile_data=real_costs)
        
    plan_summary = [len(plan[r]) for r in plan]
    print(f"    Plan: {plan_summary}")
    
    # 3. Execution
    manager = mp.Manager()
    return_dict = manager.dict()
    procs = []
    for rank in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_metrics, args=(rank, WORLD_SIZE, plan, node_shapes, model_name, return_dict))
        p.start()
        procs.append(p)
    
    for p in procs: p.join()
    
    # 4. Metrics & Logging
    if len(return_dict) < WORLD_SIZE:
        print("    Error: Missing data from some ranks.")
        return

    lats = [return_dict[r] * 1000 for r in range(WORLD_SIZE)]
    cycle = max(lats)
    total = sum(lats)
    bubble = (cycle * WORLD_SIZE - total) / (cycle * WORLD_SIZE) if cycle > 0 else 0
    thr = 1000.0 / cycle * 32 if cycle > 0 else 0
    
    print(f"    Latencies: {[f'{x:.2f}ms' for x in lats]}")
    print(f"    Cycle Time: {cycle:.2f} ms")
    print(f"    Bubble Rate: {bubble*100:.2f}%")
    print(f"    Throughput: {thr:.2f} img/s")
    
    # Save to global logger
    logger.log(model_name, algo, plan_summary, lats, cycle, total, bubble, thr)

def main():
    models = ['resnet50', 'mobilenet_v2', 'vit', 'llama']
    algos = ['baseline', 'scdp', 'optimal']
    
    print("Starting Experiments with Oracle Profiling & Data Logging...")
    for m in models:
        for a in algos:
            try:
                run_experiment(m, a)
                time.sleep(2)
            except Exception as e:
                print(f"Error: {e}")
    
    # Final Save
    logger.save()

if __name__ == "__main__":
    main()
