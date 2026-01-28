import torch
import torch.fx
import copy
from scdp_core import GraphPartitioner
from runtime_utils import get_model_and_input
from torch.fx.passes.shape_prop import ShapeProp 

class HpipePlanner:
    def __init__(self, model_name, devices=4):
        self.model_name = model_name
        self.devices = devices
        self.model, self.sample_input = get_model_and_input(model_name)
        
        self.partitioner = GraphPartitioner(self.model, num_devices=devices)
        # Ensure shape prop is run
        try:
            ShapeProp(self.partitioner.traced).propagate(self.sample_input)
        except Exception as e:
            print(f"  [Warn] ShapeProp failed for {model_name}: {e}")
        
        self.node_costs = {}
        self.node_shapes = {}
        self.name_to_node = {}
        self.node_list = [] 
        
        for node in self.partitioner.traced.graph.nodes:
            self.name_to_node[node.name] = node
            self.node_list.append(node)
            self.node_costs[node.name] = self.partitioner._get_cost(node)
            
            if 'tensor_meta' in node.meta and hasattr(node.meta['tensor_meta'], 'shape'):
                self.node_shapes[node.name] = node.meta['tensor_meta'].shape

    def is_clean_cut(self, node_name):
        """Check if a node represents a clean architectural boundary."""
        if node_name not in self.name_to_node: return False
        node = self.name_to_node[node_name]
        
        # 1. Residual Add (ResNet, MobileNet, Transformers)
        is_add = node.op == 'call_function' and ('add' in str(node.target) or 'sum' in node.name)
        
        # 2. ReLU after Add (ResNet)
        is_relu_after_add = False
        if node.op == 'call_module' and 'relu' in node.name:
            if len(node.args) > 0 and isinstance(node.args[0], torch.fx.Node):
                prev = node.args[0]
                if prev.op == 'call_function' and 'add' in str(prev.target):
                    is_relu_after_add = True
        
        # 3. Transformer Specific: often ends with add (skip connection)
        # We might also look for 'mul' in LayerNorm if it's the very last op of a block
        
        return is_add or is_relu_after_add

    def align_to_nearest_cut(self, plan_nodes):
        if not plan_nodes: return plan_nodes
        last_node_name = plan_nodes[-1]
        try:
            global_idx = [n.name for n in self.node_list].index(last_node_name)
        except ValueError:
            return plan_nodes

        best_idx = global_idx
        min_dist = 100
        
        # Search radius
        radius = 15
        start_search = max(0, global_idx - radius)
        end_search = min(len(self.node_list), global_idx + radius)
        
        for i in range(start_search, end_search):
            if self.is_clean_cut(self.node_list[i].name):
                dist = abs(i - global_idx)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        
        if best_idx != global_idx:
            start_name = plan_nodes[0]
            start_idx = [n.name for n in self.node_list].index(start_name)
            return [n.name for n in self.node_list[start_idx : best_idx+1]]
            
        return plan_nodes

    def generate_plans(self):
        # print(f"  [Hpipe] Planning for {self.model_name}...")
        raw_plan_a = self.partitioner.run_scdp() 
        
        # Step 1: Align Plan A
        plan_a = {}
        current_start_idx = 0
        all_names = [n.name for n in self.node_list]
        
        for r in range(self.devices):
            original_nodes = raw_plan_a[r]
            if not original_nodes: 
                plan_a[r] = []
                continue
            
            if r == self.devices - 1:
                plan_a[r] = all_names[current_start_idx:]
                continue
            
            aligned_nodes = self.align_to_nearest_cut(original_nodes)
            length = len(aligned_nodes)
            actual_end = current_start_idx + length
            plan_a[r] = all_names[current_start_idx : actual_end]
            current_start_idx = actual_end

        # Step 2: Generate Plan B (Shift Workload)
        plan_b = copy.deepcopy(plan_a)
        r1_nodes = plan_a[1]
        shift_idx = -1
        
        # Find clean cut in R1 to shift
        for i, name in enumerate(r1_nodes):
            if i > 5 and self.is_clean_cut(name): 
                shift_idx = i
                break
        
        # Fallback if no clean cut found (e.g. small models)
        if shift_idx == -1 and len(r1_nodes) > 5:
            shift_idx = 5

        if shift_idx > 0:
            transfer = r1_nodes[:shift_idx+1]
            plan_b[0].extend(transfer)
            plan_b[1] = r1_nodes[shift_idx+1:]
            # print(f"  [Hpipe] Plan B Shift: {len(transfer)} nodes moved.")
        
        return plan_a, plan_b
