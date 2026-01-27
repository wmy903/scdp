import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from datetime import timedelta
from torch.fx.passes.shape_prop import ShapeProp

from scdp_core import GraphPartitioner
from runtime_utils import get_model_and_input, PipelineStage, profile_model

# === Configuration ===
WORLD_SIZE = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29607' # Increment port

# === NCCL Settings ===
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_NET_GDR_LEVEL"] = "0"

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
        
    print(f"    Plan: {[len(plan[r]) for r in plan]}")
    
    # 3. Execution
    manager = mp.Manager()
    return_dict = manager.dict()
    procs = []
    for rank in range(WORLD_SIZE):
        p = ctx.Process(target=run_worker_metrics, args=(rank, WORLD_SIZE, plan, node_shapes, model_name, return_dict))
        p.start()
        procs.append(p)
    
    for p in procs: p.join()
    
    # 4. Metrics
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

def main():
    models = ['resnet50', 'mobilenet_v2', 'vit', 'llama']
    # Added 'optimal' to the list
    algos = ['baseline', 'scdp', 'optimal']
    
    print("Starting Experiments with Oracle Profiling & Optimal Baseline...")
    for m in models:
        for a in algos:
            try:
                run_experiment(m, a)
                time.sleep(2)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
