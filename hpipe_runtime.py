import torch
import torch.nn as nn
import time
import torch.distributed as dist
from runtime_utils import PipelineStage

class BaselineWorker(nn.Module):
    def __init__(self, full_model, nodes, rank, device):
        super().__init__()
        self.rank = rank
        self.device = device
        self.stage = PipelineStage(full_model, nodes, rank).to(device)
        
    def forward(self, x):
        # Standard forward
        return self.stage(x)

class HpipeRank0(nn.Module):
    """
    Smart Worker for Rank 0 that supports Dynamic Handoff.
    It holds layers [0...Max_Cut].
    """
    def __init__(self, full_model, all_nodes, cut_a, cut_b, device):
        super().__init__()
        self.device = device
        self.cut_early = min(cut_a, cut_b)
        self.cut_late = max(cut_a, cut_b)
        
        # Block 1: 0 to Early Cut
        nodes_1 = all_nodes[0 : self.cut_early + 1]
        self.block1 = PipelineStage(full_model, nodes_1, 0).to(device)
        
        # Block 2: Early Cut + 1 to Late Cut (The Handoff Delta)
        nodes_2 = all_nodes[self.cut_early + 1 : self.cut_late + 1]
        self.block2 = PipelineStage(full_model, nodes_2, 0).to(device)
        
        print(f"  [Hpipe-R0] Built Dynamic Stage. Exit 1 at {self.cut_early}, Exit 2 at {self.cut_late}")

    def forward(self, x, active_plan_is_b):
        # 1. Always run Block 1
        x = self.block1(x)
        
        # 2. Check Handoff
        # If Plan B (Contention Plan) is active, Rank 0 does MORE work (Block 2)
        # to relieve Rank 1.
        if active_plan_is_b:
            x = self.block2(x)
            # Send x (Late Exit)
            return x, "late"
        else:
            # Plan A (Normal): Early Exit
            # Send x immediately
            return x, "early"

class HpipeRank1(nn.Module):
    """
    Rank 1 needs to handle receiving from either Early or Late exit of Rank 0.
    In a real implementation, this is complex (skipping layers).
    For this experiment demo, we construct Block2 as 'optional' input bypass.
    """
    def __init__(self, full_model, all_nodes, cut_a, cut_b, end_idx, device):
        super().__init__()
        self.device = device
        self.cut_early = min(cut_a, cut_b)
        self.cut_late = max(cut_a, cut_b)
        
        # Block 2 Replica: The part Rank 0 might do, or Rank 1 might do.
        # Nodes: Early+1 to Late
        nodes_2 = all_nodes[self.cut_early + 1 : self.cut_late + 1]
        self.block2 = PipelineStage(full_model, nodes_2, 1).to(device)
        
        # Block 3: Late + 1 to End of Rank 1
        nodes_3 = all_nodes[self.cut_late + 1 : end_idx + 1]
        self.block3 = PipelineStage(full_model, nodes_3, 1).to(device)

    def forward(self, x, from_exit_type):
        # If data came from "early" exit (Rank 0 did less), I must run Block 2
        if from_exit_type == "early":
            x = self.block2(x)
        
        # If data came from "late" exit (Rank 0 did more), I skip Block 2
        # Always run Block 3
        
        # Simulate Interference Here!
        # If this is Rank 1, we might inject delay
        # (This logic will be in the runner loop)
        
        x = self.block3(x)
        return x
