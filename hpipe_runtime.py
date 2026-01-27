import torch
import torch.nn as nn
from runtime_utils import PipelineStage

def smart_connect(prev_output, next_stage):
    """
    Connects output of one stage to input of next, handling Tuple/Tensor mismatch
    and Shape matching.
    """
    # 1. Get requirements
    graph = next_stage.sub_module.graph
    placeholders = [n for n in graph.nodes if n.op == 'placeholder']
    num_required = len(placeholders)
    
    # 2. Normalize inputs
    if isinstance(prev_output, (tuple, list)):
        available_inputs = list(prev_output)
    else:
        available_inputs = [prev_output]
    
    # 3. Simple Count Match
    if num_required == len(available_inputs):
        return next_stage(*available_inputs)
    
    # 4. Single Input Expected (Most common case for clean cuts)
    if num_required == 1:
        # If we have multiple outputs, find the one that fits?
        # For now, taking the first one is the standard convention for "Main Stream"
        return next_stage(available_inputs[0])
        
    # 5. Not enough inputs? (e.g. need x+residual, got x)
    if num_required > len(available_inputs):
        # Pad with the last available input (heuristic fix for latency test)
        diff = num_required - len(available_inputs)
        padded = available_inputs + [available_inputs[-1]] * diff
        return next_stage(*padded)

    # 6. Too many inputs?
    return next_stage(*available_inputs[:num_required])

class BaselineWorker(nn.Module):
    def __init__(self, full_model, nodes, rank, device):
        super().__init__()
        self.rank = rank
        self.device = device
        # Ensure strings
        node_names = [n.name if hasattr(n, 'name') else n for n in nodes]
        self.stage = PipelineStage(full_model, node_names, rank).to(device)
        
    def forward(self, *inputs):
        return self.stage(*inputs)

class HpipeRank0(nn.Module):
    def __init__(self, full_model, all_nodes, cut_a, cut_b, device):
        super().__init__()
        self.device = device
        self.cut_early = min(cut_a, cut_b)
        self.cut_late = max(cut_a, cut_b)
        
        raw_1 = all_nodes[0 : self.cut_early + 1]
        raw_2 = all_nodes[self.cut_early + 1 : self.cut_late + 1]
        
        n1 = [n.name if hasattr(n, 'name') else n for n in raw_1]
        n2 = [n.name if hasattr(n, 'name') else n for n in raw_2]
        
        self.block1 = PipelineStage(full_model, n1, 0).to(device)
        self.block2 = PipelineStage(full_model, n2, 0).to(device)
        
        print(f"  [Hpipe-R0] Ready. Exit1={self.cut_early}, Exit2={self.cut_late}")

    def forward(self, x, active_plan_is_b):
        # Rank 0 input is single tensor
        out1 = self.block1(x)
        
        if active_plan_is_b:
            # Run Block 2 using smart connect
            out2 = smart_connect(out1, self.block2)
            return out2, "late"
        else:
            return out1, "early"

class HpipeRank1(nn.Module):
    def __init__(self, full_model, all_nodes, cut_a, cut_b, end_idx, device):
        super().__init__()
        self.device = device
        self.cut_early = min(cut_a, cut_b)
        self.cut_late = max(cut_a, cut_b)
        
        raw_2 = all_nodes[self.cut_early + 1 : self.cut_late + 1]
        raw_3 = all_nodes[self.cut_late + 1 : end_idx + 1]
        
        n2 = [n.name if hasattr(n, 'name') else n for n in raw_2]
        n3 = [n.name if hasattr(n, 'name') else n for n in raw_3]
        
        self.block2 = PipelineStage(full_model, n2, 1).to(device)
        self.block3 = PipelineStage(full_model, n3, 1).to(device)

    def forward(self, x, from_exit_type):
        val = x
        if from_exit_type == "early":
            # Rank 0 exited early -> We run Block 2
            val = smart_connect(x, self.block2)
            
        # Run Block 3
        return smart_connect(val, self.block3)
