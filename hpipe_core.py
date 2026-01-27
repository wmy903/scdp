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
        
        # 1. 运行 Shape Propagation (必须在收集 info 之前)
        ShapeProp(self.partitioner.traced).propagate(self.sample_input)
        
        # 2. 初始化类属性 (确保 self.node_shapes 在这里被定义)
        self.node_costs = {}
        self.node_shapes = {} # [Critical Fix] Explicitly define this
        self.name_to_node = {}
        self.node_list = [] 
        
        # 3. 收集信息
        for node in self.partitioner.traced.graph.nodes:
            self.name_to_node[node.name] = node
            self.node_list.append(node)
            self.node_costs[node.name] = self.partitioner._get_cost(node)
            
            # 提取 Shape
            if 'tensor_meta' in node.meta and hasattr(node.meta['tensor_meta'], 'shape'):
                self.node_shapes[node.name] = node.meta['tensor_meta'].shape

    def is_clean_cut(self, node_name):
        """Check if a node represents a clean architectural boundary."""
        if node_name not in self.name_to_node: return False
        node = self.name_to_node[node_name]
        
        # Condition 1: Element-wise add (End of Residual Block)
        is_add = node.op == 'call_function' and ('add' in str(node.target) or 'sum' in node.name)
        
        # Condition 2: ReLU immediately after add
        is_relu_after_add = False
        if node.op == 'call_module' and 'relu' in node.name:
            if len(node.args) > 0 and isinstance(node.args[0], torch.fx.Node):
                prev = node.args[0]
                if prev.op == 'call_function' and 'add' in str(prev.target):
                    is_relu_after_add = True
                    
        return is_add or is_relu_after_add

    def align_to_nearest_cut(self, plan_nodes):
        """Snap split point to the nearest clean cut."""
        if not plan_nodes: return plan_nodes
        
        last_node_name = plan_nodes[-1]
        try:
            global_idx = [n.name for n in self.node_list].index(last_node_name)
        except ValueError:
            return plan_nodes

        best_idx = global_idx
        min_dist = 100
        
        # Search radius: 15 nodes (covers a full Bottleneck block)
        for i in range(max(0, global_idx - 15), min(len(self.node_list), global_idx + 15)):
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
        print("  [Hpipe] Generating Normal Plan A...")
        raw_plan_a = self.partitioner.run_scdp() 
        
        # [Step 1] Align Plan A to Clean Cuts
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

        print("  [Hpipe] Generating Contention Plan B...")
        plan_b = copy.deepcopy(plan_a)
        
        # [Step 2] Shift workload: Move ONE full residual block from R1 to R0
        r1_nodes = plan_a[1]
        shift_idx = -1
        
        # Find the first clean cut inside Rank 1 to act as the shift point
        for i, name in enumerate(r1_nodes):
            # Skip first few nodes to make sure we move something substantial
            if i > 5 and self.is_clean_cut(name): 
                shift_idx = i
                break
        
        if shift_idx > 0:
            transfer = r1_nodes[:shift_idx+1]
            plan_b[0].extend(transfer)
            plan_b[1] = r1_nodes[shift_idx+1:]
            print(f"  [Hpipe] Plan B Shift: Moved {len(transfer)} nodes (Clean Block).")
        else:
            print("  [Hpipe] Warning: Could not shift cleanly.")
            
        return plan_a, plan_b
