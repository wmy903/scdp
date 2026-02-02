import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import random
import sys
from tabulate import tabulate
from torch.fx.passes.shape_prop import ShapeProp

from scdp_core import (
    GraphPartitioner, Coarsener, DPPartitioner, 
    AdapipePartitioner, PicoPartitioner, DagPPartitioner, UniformPartitioner
)
from runtime_utils import get_model_and_input, PipelineStage, profile_model, LeafTracer

# === Configuration ===
TARGET_BATCH_SIZE = 32
PROFILE_BATCH_SIZE = 1
WORLD_SIZE = 4
# Low bandwidth setting to emphasize communication cost
COMM_BETA = 1.0 / (1 * 1024**3 / 1000.0) 
os.environ['MASTER_ADDR'] = 'localhost'
os.environ["NCCL_P2P_DISABLE"] = "1" 
os.environ["NCCL_IB_DISABLE"] = "1"

def setup_dist(rank, world_size, port):
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_dist():
    try: dist.destroy_process_group()
    except: pass

def inject_noise(profile_data, noise_ratio):
    noisy_data = {}
    for name, cost in profile_data.items():
        factor = 1.0 + np.random.uniform(-noise_ratio, noise_ratio)
        noisy_data[name] = max(0.001, cost * factor)
    return noisy_data

def get_adaptive_scdp_params(model_name, total_nodes):
    name = model_name.lower()
    if 'mobilenet' in name: return {'max_nodes': total_nodes + 10, 'k_stages': WORLD_SIZE}, True
    elif 'vit' in name: return {'max_nodes': total_nodes + 10, 'k_stages': WORLD_SIZE}, True
    elif 'resnet' in name: return {'max_nodes': 150, 'k_stages': WORLD_SIZE}, False
    else: return {'max_nodes': WORLD_SIZE * 10, 'k_stages': WORLD_SIZE}, False

def run_worker_deployment(rank, world_size, port, plan, model_name, node_shapes, traced_model_cpu, return_dict):
    try:
        setup_dist(rank, world_size, port)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        my_nodes = plan[rank]
        if not my_nodes:
            return_dict[rank] = 0.0
            cleanup_dist()
            return

        try:
            model_arch = 'transformer' if 'vit' in model_name or 'llama' in model_name else 'cnn'
            # batch_size MUST be TARGET_BATCH_SIZE for real execution
            stage_model = PipelineStage(
                traced_model_cpu, my_nodes, rank, world_size, node_shapes, 
                model_arch=model_arch, model_name=model_name, batch_size=TARGET_BATCH_SIZE
            ).to(device)
            stage_model.eval()

            with torch.no_grad():
                for _ in range(3): stage_model()
                torch.cuda.synchronize()
                
                latencies = []
                for _ in range(10):
                    t0 = time.time()
                    stage_model()
                    torch.cuda.synchronize()
                    latencies.append((time.time() - t0) * 1000)
                
                return_dict[rank] = np.median(latencies)

        except RuntimeError as e:
            return_dict[rank] = -999.0 
        except Exception as e:
            return_dict[rank] = -999.0

        cleanup_dist()
        
    except Exception as e:
        return_dict[rank] = -999.0

def calculate_comm_cost_for_plan(plan, algo_graph):
    comm_costs = {}
    node_to_stage = {}
    for stage_id, nodes in plan.items():
        for node_name in nodes: node_to_stage[node_name] = stage_id
    
    for stage_id in range(len(plan)):
        stage_nodes = plan[stage_id]
        comm_bytes = 0.0
        for nid in stage_nodes:
            if nid not in algo_graph.adj: continue
            for neighbor in algo_graph.adj[nid]:
                if neighbor not in node_to_stage: continue
                if node_to_stage[neighbor] == stage_id + 1:
                    edge_key = (nid, neighbor)
                    if edge_key in algo_graph.edges:
                        comm_bytes += algo_graph.edges[edge_key].tensor_bytes
        comm_costs[stage_id] = comm_bytes * COMM_BETA
    return comm_costs

def main():
    print(f"{'='*120}")
    print(f"REAL DEPLOYMENT EXPERIMENT (4 GPUs)")
    print(f"Update: Llama uses COARSE granularity (~100 nodes). Others use ATOMIC (~200-400 nodes).")
    print(f"BS Source: Ours=32, Baselines=1. Execution BS=32.")
    print(f"{'='*120}")
    
    models = ['resnet50', 'mobilenet_v3_large', 'vit', 'llama']
    # models = ['llama'] # Debug
    
    mp.set_start_method('spawn', force=True)
    summary_results = []

    for model_name in models:
        print(f"\nProcessing {model_name}...")
        
        # [MODIFIED] Determine Tracer based on model type
        # Llama -> Coarse (keep blocks/MLP atomic)
        # Others -> Fine (unfold everything)
        is_llama = 'llama' in model_name
        tracer = LeafTracer(coarse_llama=is_llama)
        
        try:
            # A. Profile with Target BS=32 (for Ours)
            print(f"  [Profiler] Profiling with BS={TARGET_BATCH_SIZE}...")
            model_32, inp_32 = get_model_and_input(model_name, batch_size=TARGET_BATCH_SIZE)
            costs_32 = profile_model(model_32, inp_32, device='cuda:0', tracer=tracer)
            
            # B. Profile with Base BS=1 (for Baselines)
            print(f"  [Profiler] Profiling with BS={PROFILE_BATCH_SIZE} (Proxy)...")
            model_1, inp_1 = get_model_and_input(model_name, batch_size=PROFILE_BATCH_SIZE)
            costs_1 = profile_model(model_1, inp_1, device='cuda:0', tracer=tracer)
            
        except Exception as e:
            print(f"  Skipping due to profile error: {e}"); continue

        # --- 2. Graph Construction (Using the SELECTED tracer) ---
        try:
            # Build canonical graph structure
            graph = tracer.trace(model_32)
            traced_canonical = torch.fx.GraphModule(model_32, graph)
            
            # Build Graph for Ours (BS=32 weights)
            gp_32 = GraphPartitioner(traced_canonical, costs_32, WORLD_SIZE, batch_size=TARGET_BATCH_SIZE)
            full_graph_32 = gp_32._build()
            
        except Exception as e:
            print(f"  Graph build error: {e}"); continue
        
        # --- 3. Shape Propagation (BS=32) ---
        try:
            # Must use same tracer to match node names
            model_cpu, inp_cpu = get_model_and_input(model_name, batch_size=TARGET_BATCH_SIZE)
            graph = tracer.trace(model_cpu)
            traced_model_cpu = torch.fx.GraphModule(model_cpu, graph)
            ShapeProp(traced_model_cpu).propagate(inp_cpu)
            
            node_shapes = {}
            for n in traced_model_cpu.graph.nodes:
                if n.op != 'output':
                    tm = n.meta.get('tensor_meta')
                    if tm is not None and hasattr(tm, 'shape'):
                        s = list(tm.shape); s[0] = TARGET_BATCH_SIZE
                        node_shapes[n.name] = tuple(s)
            print(f"  [ShapeProp] Captured shapes for {len(node_shapes)} nodes.")
        except Exception as e:
            print(f"  [Warning] ShapeProp failed: {e}. Using fallback.")
            node_shapes = {}

        # --- 4. Prepare Costs for Baselines ---
        costs_n30 = inject_noise(costs_1, 0.3)
        costs_n40 = inject_noise(costs_1, 0.4)
        costs_n50 = inject_noise(costs_1, 0.5)
        
        plans = {}
        
        # --- 5. Generate Plans ---
        
        # >> Group A: Powered by BS=32 Data <<
        p, _ = DPPartitioner(full_graph_32, WORLD_SIZE).solve(); plans['Optimal'] = p
        
        scdp_params, force_no = get_adaptive_scdp_params(model_name, len(full_graph_32.nodes))
        coarsener = Coarsener(full_graph_32, costs_32, model_arch=('transformer' if 'vit' in model_name or 'llama' in model_name else 'cnn'))
        cg = coarsener.coarsen(max_nodes=scdp_params['max_nodes'], k_stages=scdp_params['k_stages'], force_no_limit=force_no)
        p, _ = DPPartitioner(cg, WORLD_SIZE).solve(); plans['SCDP(Clean)'] = p
        
        p, _ = UniformPartitioner(traced_canonical, costs_32, WORLD_SIZE).solve(); plans['Uniform'] = p
        
        # >> Group B: Powered by BS=1 Data <<
        p, _ = AdapipePartitioner(traced_canonical, costs_1, WORLD_SIZE).solve(); plans['AdaPipe(Clean)'] = p
        p, _ = PicoPartitioner(traced_canonical, costs_1, WORLD_SIZE).solve(); plans['Pico(Clean)'] = p
        p, _ = DagPPartitioner(traced_canonical, costs_1, WORLD_SIZE).solve(); plans['DagP(Clean)'] = p
        
        p, _ = AdapipePartitioner(traced_canonical, costs_n30, WORLD_SIZE).solve(); plans['AdaPipe(N30%)'] = p
        p, _ = PicoPartitioner(traced_canonical, costs_n40, WORLD_SIZE).solve(); plans['Pico(N40%)'] = p
        p, _ = DagPPartitioner(traced_canonical, costs_n50, WORLD_SIZE).solve(); plans['DagP(N50%)'] = p

        # --- 6. Execute ---
        print(f"  -> Executing plans on 4 GPUs (BS={TARGET_BATCH_SIZE})...")
        model_results = []
        
        order = ['Optimal', 'SCDP(Clean)', 'Uniform', 'AdaPipe(Clean)', 'AdaPipe(N30%)', 'Pico(Clean)', 'Pico(N40%)', 'DagP(Clean)', 'DagP(N50%)']
        
        for algo_name in order:
            if algo_name not in plans: continue
            plan = plans[algo_name]
            
            comm_costs = calculate_comm_cost_for_plan(plan, full_graph_32)
            
            manager = mp.Manager(); return_dict = manager.dict()
            port = str(random.randint(20000, 30000))
            procs = []
            for rank in range(WORLD_SIZE):
                p = mp.Process(target=run_worker_deployment, args=(rank, WORLD_SIZE, port, plan, model_name, node_shapes, traced_model_cpu, return_dict))
                p.start(); procs.append(p)
            for p in procs: p.join()
            
            stages = []
            crashed = False
            for r in range(WORLD_SIZE):
                val = return_dict.get(r, -999.0)
                if val == -999.0: crashed = True; break
                stages.append(val + comm_costs[r])
            
            if crashed:
                model_results.append([algo_name, "Fail", "Fail", "Fail", "Fail", "Fail", "Fail", "Fail", "0.0"])
                summary_results.append([model_name, algo_name, "Fail", "0.0"])
                print(f"    {algo_name}: Failed (Invalid Graph)")
            else:
                cycle = max(stages)
                total_lat = sum(stages)
                bubble_rate = (WORLD_SIZE * cycle - total_lat) / (WORLD_SIZE * cycle) if cycle > 0 else 0
                throughput = (TARGET_BATCH_SIZE * 1000) / cycle if cycle > 0 else 0
                
                model_results.append([algo_name] + [f"{x:.1f}" for x in stages] + [f"{cycle:.1f}", f"{total_lat:.1f}", f"{bubble_rate*100:.1f}%", f"{throughput:.1f}"])
                summary_results.append([model_name, algo_name, f"{cycle:.1f}", f"{throughput:.1f}"])
                print(f"    {algo_name}: Cycle={cycle:.1f}ms")

        headers = ["Algo", "S0", "S1", "S2", "S3", "Cycle", "Total", "Bubbles", "Thrpt"]
        print(f"\nResults for {model_name}:")
        print(tabulate(model_results, headers=headers))

    print(f"\n{'='*100}")
    print(f"FINAL DEPLOYMENT SUMMARY")
    print(f"Target BS={TARGET_BATCH_SIZE} | Baseline Profile BS={PROFILE_BATCH_SIZE}")
    print(f"{'Model':<18} | {'Algo':<15} | {'Cycle(ms)':<10} | {'Throughput':<12}")
    print("-" * 65)
    for res in summary_results:
        print(f"{res[0]:<18} | {res[1]:<15} | {res[2]:<10} | {res[3]:<12}")

if __name__ == "__main__":
    main()
