import torch
import torch.fx
import networkx as nx
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

# === 1. Data Structures ===

@dataclass
class AlgoNode:
    id: str
    compute_cost: float = 0.0
    memory_footprint: float = 0.0
    contained_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.contained_names:
            self.contained_names = [self.id]

@dataclass
class AlgoEdge:
    u: str
    v: str
    data_size: float = 0.0

class AlgoGraph:
    def __init__(self):
        self.nodes: Dict[str, AlgoNode] = {}
        self.edges: Dict[Tuple[str, str], AlgoEdge] = {}
        self.adj: Dict[str, List[str]] = {}
        self.rev_adj: Dict[str, List[str]] = {}

    def add_node(self, node: AlgoNode):
        self.nodes[node.id] = node
        if node.id not in self.adj: self.adj[node.id] = []
        if node.id not in self.rev_adj: self.rev_adj[node.id] = []

    def add_edge(self, u: str, v: str, data_size: float):
        if (u, v) in self.edges:
            self.edges[(u, v)].data_size += data_size
        else:
            self.edges[(u, v)] = AlgoEdge(u, v, data_size)
            self.adj[u].append(v)
            self.rev_adj[v].append(u)

    def remove_node(self, nid: str):
        if nid not in self.nodes: return
        for neighbor in list(self.adj[nid]):
            del self.edges[(nid, neighbor)]
            self.rev_adj[neighbor].remove(nid)
        for source in list(self.rev_adj[nid]):
            del self.edges[(source, nid)]
            self.adj[source].remove(nid)
        del self.nodes[nid]
        del self.adj[nid]
        del self.rev_adj[nid]

    def to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for nid in self.nodes:
            G.add_node(nid, weight=self.nodes[nid].compute_cost)
        for (u, v), edge in self.edges.items():
            G.add_edge(u, v, weight=edge.data_size)
        return G

# === 2. Coarsening (Heuristic for Scalability) ===

class Coarsener:
    def __init__(self, graph: AlgoGraph):
        self.graph = copy.deepcopy(graph)

    def check_cycle_after_merge(self, u: str, v: str) -> bool:
        G = self.graph.to_networkx()
        if G.has_edge(u, v): G.remove_edge(u, v)
        return nx.has_path(G, u, v)

    def merge_nodes(self, u: str, v: str):
        node_u = self.graph.nodes[u]
        node_v = self.graph.nodes[v]
        node_u.compute_cost += node_v.compute_cost
        node_u.memory_footprint = max(node_u.memory_footprint, node_v.memory_footprint)
        node_u.contained_names.extend(node_v.contained_names)
        
        for neighbor in list(self.graph.adj[v]):
            if neighbor == u: continue
            data = self.graph.edges[(v, neighbor)].data_size
            self.graph.add_edge(u, neighbor, data)
        for source in list(self.graph.rev_adj[v]):
            if source == u: continue
            data = self.graph.edges[(source, v)].data_size
            self.graph.add_edge(source, u, data)
        self.graph.remove_node(v)

    def coarsen(self, target_size: int) -> AlgoGraph:
        max_rounds = 10
        for _ in range(max_rounds):
            if len(self.graph.nodes) <= target_size: break
            edges = list(self.graph.edges.values())
            edges.sort(key=lambda e: e.data_size, reverse=True)
            merged_any = False
            for edge in edges:
                u, v = edge.u, edge.v
                if not self.check_cycle_after_merge(u, v):
                    self.merge_nodes(u, v)
                    merged_any = True
                    break 
            if not merged_any: break
        return self.graph

# === 3. DP Partitioner (Exact Optimization) ===

class DPPartitioner:
    def __init__(self, graph: AlgoGraph, num_devices: int):
        self.graph = graph
        self.K = num_devices
        
        # Linearize graph
        try:
            self.topo_order = list(nx.topological_sort(self.graph.to_networkx()))
        except:
            self.topo_order = list(self.graph.nodes.keys())
        
        self.nodes = [self.graph.nodes[nid] for nid in self.topo_order]
        self.M = len(self.nodes)
        
        # Pre-compute prefix sums for compute cost
        self.prefix_comp = [0.0] * (self.M + 1)
        for i in range(self.M):
            self.prefix_comp[i+1] = self.prefix_comp[i] + self.nodes[i].compute_cost

    def get_communication_cost(self, start_idx: int, end_idx: int) -> float:
        """
        Calculate precise communication cost for a stage covering nodes[start_idx:end_idx].
        Comm Cost = Sum of data sizes of edges going OUT of this stage.
        """
        # Bandwidth constant (ms / MB)
        # Assuming PCIe/NVLink ~10GB/s effective => 0.1 ms/MB
        # Tuning this is important. Let's assume a fast interconnect.
        BETA = 0.05 
        
        stage_nodes = set(self.topo_order[i] for i in range(start_idx, end_idx))
        total_data = 0.0
        
        for u_id in stage_nodes:
            # Check outgoing edges
            if u_id in self.graph.adj:
                for v_id in self.graph.adj[u_id]:
                    # If v is NOT in this stage, it's a cut edge
                    if v_id not in stage_nodes:
                        total_data += self.graph.edges[(u_id, v_id)].data_size
        
        return total_data * BETA

    def solve(self):
        # dp[k][i] = min max_latency using k devices for first i blocks
        dp = np.full((self.K, self.M + 1), float('inf'))
        split = np.zeros((self.K, self.M + 1), dtype=int)
        
        # --- Initialization (Device 0) ---
        for i in range(1, self.M + 1):
            comp = self.prefix_comp[i]
            # Device 0 always sends data out if i < M
            comm = self.get_communication_cost(0, i) if i < self.M else 0
            dp[0][i] = comp + comm

        # --- DP Loop ---
        for k in range(1, self.K):
            for i in range(1, self.M + 1):
                # Try all valid split points j
                # Rank k takes nodes [j, i)
                start_search = k # Ensure at least 1 node per previous rank
                end_search = i   # Ensure at least 1 node for current rank
                
                for j in range(start_search, end_search):
                    prev_max = dp[k-1][j]
                    if prev_max == float('inf'): continue
                    
                    # Current stage cost
                    curr_comp = self.prefix_comp[i] - self.prefix_comp[j]
                    
                    # Comm cost is 0 for the last device (Pipeline Sink)
                    is_last_stage = (k == self.K - 1) and (i == self.M)
                    curr_comm = 0.0 if is_last_stage else self.get_communication_cost(j, i)
                    
                    curr_latency = curr_comp + curr_comm
                    bottleneck = max(prev_max, curr_latency)
                    
                    if bottleneck < dp[k][i]:
                        dp[k][i] = bottleneck
                        split[k][i] = j

        # --- Backtrack ---
        partition_map = {} 
        curr = self.M
        for k in range(self.K - 1, -1, -1):
            start = split[k][curr]
            names = []
            for node in self.nodes[start:curr]:
                names.extend(node.contained_names)
            partition_map[k] = names
            curr = start
            
        return partition_map

# === 4. Partitioning Interface ===

class GraphPartitioner:
    def __init__(self, model: torch.nn.Module, num_devices=4):
        self.model = model
        self.devices = num_devices
        self.traced = torch.fx.symbolic_trace(model)
        
    def _get_cost(self, node: torch.fx.Node, profile_data: Dict[str, float] = None) -> float:
        # 1. Oracle Data
        if profile_data and node.name in profile_data:
            return profile_data[node.name]
        
        # 2. Heuristic Fallback
        if 'tensor_meta' not in node.meta: return 0.001
        meta = node.meta['tensor_meta']
        if not hasattr(meta, 'shape'): return 0.001
        out_shape = list(meta.shape)
        if not out_shape: return 0.001
        out_shape[0] = 32
        
        num_elements = np.prod(out_shape)
        # Simple FLOPs proxy
        compute_ops = 0.0
        if node.op == 'call_module': compute_ops = num_elements * 100 
        elif node.op == 'call_function' and 'matmul' in str(node.target): compute_ops = num_elements * 128
        
        return (compute_ops / 20e9) + 0.01

    def _build_graph(self, profile_data):
        g = AlgoGraph()
        for node in self.traced.graph.nodes:
            cost = self._get_cost(node, profile_data)
            dsize = 0.0
            if 'tensor_meta' in node.meta and hasattr(node.meta['tensor_meta'], 'shape'):
                # MB size
                dsize = np.prod(node.meta['tensor_meta'].shape) * 4 / (1024**2) 
            g.add_node(AlgoNode(node.name, cost, dsize))
            
        for node in self.traced.graph.nodes:
            for user in node.users:
                dsize = g.nodes[node.name].memory_footprint
                g.add_edge(node.name, user.name, dsize)
        return g

    def run_optimal(self, profile_data: Dict[str, float] = None):
        """
        [New] Optimal Partitioning Algorithm.
        Directly runs DP on the full, un-coarsened graph with precise costs.
        """
        raw_graph = self._build_graph(profile_data)
        # Skip coarsening, run DP directly
        partitioner = DPPartitioner(raw_graph, self.devices)
        return partitioner.solve()

    def run_scdp(self, profile_data: Dict[str, float] = None):
        """
        SCDP: Coarsening + DP.
        Scalable approach for very large graphs.
        """
        raw_graph = self._build_graph(profile_data)
        
        # Skip coarsening if graph is small (optimization)
        if len(raw_graph.nodes) < 500:
            coarsened = raw_graph
        else:
            coarsener = Coarsener(raw_graph)
            coarsened = coarsener.coarsen(target_size=self.devices * 8)
        
        partitioner = DPPartitioner(coarsened, self.devices)
        return partitioner.solve()

    def run_baseline(self, profile_data: Dict[str, float] = None):
        """Greedy Baseline"""
        nodes_cost = []
        total_cost = 0.0
        
        for node in self.traced.graph.nodes:
            if node.op in ['call_module', 'call_function', 'call_method']:
                c = self._get_cost(node, profile_data)
                nodes_cost.append((node.name, c))
                total_cost += c
        
        avg_cost = total_cost / self.devices
        partition_map = {i: [] for i in range(self.devices)}
        curr_rank = 0
        curr_sum = 0.0
        
        for i, (name, cost) in enumerate(nodes_cost):
            partition_map[curr_rank].append(name)
            curr_sum += cost
            
            if curr_rank < self.devices - 1:
                nodes_left = len(nodes_cost) - (i + 1)
                ranks_left = (self.devices - 1) - curr_rank
                
                must_switch = (nodes_left <= ranks_left)
                greedy_switch = (curr_sum >= avg_cost)
                
                if must_switch or (greedy_switch and nodes_left > ranks_left):
                    curr_rank += 1
                    curr_sum = 0.0
            
        return partition_map
