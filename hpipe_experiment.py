import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import random
from hpipe_core import HpipePlanner
from hpipe_runtime import BaselineWorker, HpipeRank0, HpipeRank1
from runtime_utils import get_model_and_input, PipelineStage

# Config
WORLD_SIZE = 4
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29615' 
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

def simulate_interference(rank, batch_idx, intensity_ms=100):
    if rank == 1 and batch_idx > 50:
        time.sleep(intensity_ms / 1000.0)

def generate_smart_inputs(module_to_run, node_shapes, device, batch_size=32):
    """
    Introspects the FX Graph of the module to find exactly what inputs (placeholders) are needed.
    Lookups up the correct shape in node_shapes.
    Returns a LIST of tensors.
    """
    # module_to_run is likely a PipelineStage or a wrapper containing .sub_module
    if hasattr(module_to_run, 'sub_module'):
        graph = module_to_run.sub_module.graph
    elif hasattr(module_to_run, 'graph'):
        graph = module_to_run.graph
    elif hasattr(module_to_run, 'stage'): # Handle BaselineWorker wrapper
        graph = module_to_run.stage.sub_module.graph
    else:
        # Fallback: Can't inspect
        return [torch.randn(batch_size, 3, 224, 224, device=device)]

    placeholders = [n for n in graph.nodes if n.op == 'placeholder']
    inputs = []
    
    for node in placeholders:
        name = node.name
        shape = None
        
        # 1. Try exact match in shape dictionary
        if name in node_shapes:
            shape = list(node_shapes[name])
        # 2. Fallback for Input
        elif 'x' in name or 'input' in name: 
            shape = [batch_size, 3, 224, 224]
        else:
            # Last resort fallback (should be rare with proper ShapeProp)
            shape = [batch_size, 256, 56, 56] 
            
        # Ensure batch size matches
        if shape[0] != batch_size:
            shape[0] = batch_size
            
        inputs.append(torch.randn(*shape, device=device))
        
    return inputs

def run_baseline_scenario(rank, world_size, model_name, plan_a, plan_b, node_shapes, results_dict):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        full_model, _ = get_model_and_input(model_name)
        
        # --- Phase 1: Deploy Plan A ---
        current_plan = plan_a
        worker = BaselineWorker(full_model, current_plan[rank], rank, device)
        
        latencies = []
        
        for i in range(100):
            t0 = time.time()
            
            # 1. Trigger Redeployment
            if i == 60:
                if rank == 0: print(f"  [Baseline] Batch {i}: DETECTED SLOWDOWN. STOPPING Pipeline...")
                time.sleep(2.0) # Simulate Overhead
                
                if rank == 0: print(f"  [Baseline] Batch {i}: Switching to Plan B...")
                current_plan = plan_b
                worker = BaselineWorker(full_model, current_plan[rank], rank, device)
                dist.barrier()
            
            # 2. Execution
            # [FIX] Always regenerate inputs based on current worker state
            # This handles Rank 1 needing different inputs when Plan changes
            if rank == 0:
                # Rank 0 always takes generic input (image)
                inputs = [torch.randn(32, 3, 224, 224, device=device)]
                if 'llama' in model_name: inputs = [torch.randint(0, 100, (32, 64), device=device)]
                worker(*inputs)
            else:
                # Other ranks simulate receiving data by generating dummy inputs that MATCH the stage requirements
                inputs = generate_smart_inputs(worker.stage, node_shapes, device)
                worker(*inputs)
                simulate_interference(rank, i, intensity_ms=100)
            
            torch.cuda.synchronize()
            dist.barrier()
            latencies.append(time.time() - t0)
            
        if rank == 0: results_dict['baseline'] = latencies
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"[Baseline-R{rank}] Crash: {e}")
        import traceback
        traceback.print_exc()

def run_hpipe_scenario(rank, world_size, model_name, plan_a, plan_b, all_nodes, node_shapes, results_dict):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        full_model, _ = get_model_and_input(model_name)
        
        cut_a = len(plan_a[0]) - 1
        cut_b = len(plan_b[0]) - 1
        end_r1 = cut_a + len(plan_a[1]) 
        
        # --- Build Workers ---
        worker = None
        if rank == 0:
            worker = HpipeRank0(full_model, all_nodes, cut_a, cut_b, device)
        elif rank == 1:
            worker = HpipeRank1(full_model, all_nodes, cut_a, cut_b, end_r1, device)
        else:
            worker = BaselineWorker(full_model, plan_a[rank], rank, device)

        active_plan = torch.tensor([0], dtype=torch.long, device=device)
        latencies = []
        
        for i in range(100):
            t0 = time.time()
            
            if i == 60:
                if rank == 0: 
                    print(f"  [Hpipe] Batch {i}: Instant Handoff Triggered...")
                    active_plan[0] = 1 
                dist.broadcast(active_plan, src=0)
            
            is_plan_b = (active_plan.item() == 1)
            
            if rank == 0:
                # Rank 0 Input
                inputs = [torch.randn(32, 3, 224, 224, device=device)]
                if 'llama' in model_name: inputs = [torch.randint(0, 100, (32, 64), device=device)]
                worker(inputs[0], is_plan_b)
                
            elif rank == 1:
                exit_type = "late" if is_plan_b else "early"
                
                # [FIX] Smartly generate inputs for the SPECIFIC block we are about to run
                if is_plan_b:
                    # Plan B: Rank 0 ran long, Rank 1 starts from Block 3
                    inputs = generate_smart_inputs(worker.block3, node_shapes, device)
                else:
                    # Plan A: Rank 0 ran short, Rank 1 starts from Block 2
                    inputs = generate_smart_inputs(worker.block2, node_shapes, device)
                
                # worker.forward expects 'x' (or tuple), plus exit_type
                # Unpack list to tuple if multiple inputs
                inp_arg = inputs[0] if len(inputs) == 1 else tuple(inputs)
                worker(inp_arg, exit_type)
                
                simulate_interference(rank, i, intensity_ms=100)
            else:
                # R2/R3
                inputs = generate_smart_inputs(worker.stage, node_shapes, device)
                worker(*inputs)
                
            torch.cuda.synchronize()
            dist.barrier()
            latencies.append(time.time() - t0)
            
        if rank == 0: results_dict['hpipe'] = latencies
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"[Hpipe-R{rank}] Crash: {e}")
        import traceback
        traceback.print_exc()

def main():
    model_name = 'resnet50'
    print(f"=== Starting Hpipe vs Baseline Experiment ({model_name}) ===")
    
    # 1. Planning
    planner = HpipePlanner(model_name, WORLD_SIZE)
    plan_a, plan_b = planner.generate_plans()
    all_nodes = planner.partitioner.traced.graph.nodes
    all_node_names = [n.name for n in all_nodes]
    node_shapes = planner.node_shapes
    
    print(f"Plan A (Normal) R0 Size: {len(plan_a[0])}")
    print(f"Plan B (Offload) R0 Size: {len(plan_b[0])}")
    
    manager = mp.Manager()
    results = manager.dict()
    
    # 2. Baseline
    print("\n>>> Running Baseline...")
    ctx = mp.get_context('spawn')
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_baseline_scenario, args=(r, WORLD_SIZE, model_name, plan_a, plan_b, node_shapes, results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
    # 3. Hpipe
    print("\n>>> Running Hpipe...")
    procs = []
    for r in range(WORLD_SIZE):
        p = ctx.Process(target=run_hpipe_scenario, args=(r, WORLD_SIZE, model_name, plan_a, plan_b, list(all_nodes), node_shapes, results))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    
    # 4. Save
    if 'baseline' in results and 'hpipe' in results:
        df = pd.DataFrame({
            'Batch': range(100),
            'Baseline': results['baseline'],
            'Hpipe': results['hpipe']
        })
        df.to_csv('hpipe_experiment_data.csv', index=False)
        print("\nSuccess! Data saved to hpipe_experiment_data.csv")
        
        b = results['baseline']
        h = results['hpipe']
        print(f"Batch 60 (Switch): Base={b[60]*1000:.0f}ms, Hpipe={h[60]*1000:.0f}ms")
    else:
        print("Experiment failed (Results missing).")

if __name__ == "__main__":
    main()
