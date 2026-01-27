import torch
import torch.fx
import networkx as nx
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class AlgoNode:
    id: str
    compute_cost: float = 0.0
    memory_footprint: float = 0.0
    contained_names: List[str] = field(default_factory=list)
    def __post_init__(self):
        if not self.contained_names: self.contained_names = [self.id]

@dataclass
class AlgoEdge:
    u: str; v: str; data_size: float = 0.0

class AlgoGraph:
    def __init__(self):
        self.nodes = {}; self.edges = {}; self.adj = {}; self.rev_adj = {}
    def add_node(self, node):
        self.nodes[node.id] = node
        if node.id not in self.adj: self.adj[node.id] = []
        if node.id not in self.rev_adj: self.rev_adj[node.id] = []
    def add_edge(self, u, v, data):
        if (u,v) in self.edges: self.edges[(u,v)].data_size += data
        else:
            self.edges[(u,v)] = AlgoEdge(u, v, data)
            self.adj[u].append(v); self.rev_adj[v].append(u)
    def remove_node(self, nid):
        if nid not in self.nodes: return
        for n in list(self.adj[nid]): del self.edges[(nid,n)]; self.rev_adj[n].remove(nid)
        for s in list(self.rev_adj[nid]): del self.edges[(s,nid)]; self.adj[s].remove(nid)
        del self.nodes[nid]; del self.adj[nid]; del self.rev_adj[nid]
    def to_networkx(self):
        G = nx.DiGraph()
        for n in self.nodes.values(): G.add_node(n.id, weight=n.compute_cost)
        for e in self.edges.values(): G.add_edge(e.u, e.v, weight=e.data_size)
        return G

class Coarsener:
    def __init__(self, graph): self.graph = copy.deepcopy(graph)
    def check_cycle(self, u, v):
        G = self.graph.to_networkx()
        if G.has_edge(u,v): G.remove_edge(u,v)
        return nx.has_path(G, u, v)
    def merge(self, u, v):
        nu, nv = self.graph.nodes[u], self.graph.nodes[v]
        nu.compute_cost += nv.compute_cost
        nu.memory_footprint = max(nu.memory_footprint, nv.memory_footprint)
        nu.contained_names.extend(nv.contained_names)
        for n in list(self.graph.adj[v]):
            if n!=u: self.graph.add_edge(u, n, self.graph.edges[(v,n)].data_size)
        for s in list(self.graph.rev_adj[v]):
            if s!=u: self.graph.add_edge(s, u, self.graph.edges[(s,v)].data_size)
        self.graph.remove_node(v)
    def coarsen(self, target):
        if len(self.graph.nodes) <= target: return self.graph
        for _ in range(10):
            if len(self.graph.nodes) <= target: break
            edges = sorted(self.graph.edges.values(), key=lambda e: e.data_size, reverse=True)
            merged = False
            for e in edges:
                if not self.check_cycle(e.u, e.v):
                    self.merge(e.u, e.v); merged = True; break
            if not merged: break
        return self.graph

class DPPartitioner:
    def __init__(self, graph, K):
        self.graph = graph; self.K = K
        try: self.topo = list(nx.topological_sort(graph.to_networkx()))
        except: self.topo = list(graph.nodes.keys())
        self.nodes = [graph.nodes[nid] for nid in self.topo]; self.M = len(self.nodes)
    def solve(self):
        dp = np.full((self.K, self.M+1), float('inf'))
        split = np.zeros((self.K, self.M+1), dtype=int)
        prefix = [0.0]*(self.M+1)
        for i in range(self.M): prefix[i+1] = prefix[i] + self.nodes[i].compute_cost
        for i in range(1, self.M+1): dp[0][i] = prefix[i]
        for k in range(1, self.K):
            for i in range(1, self.M+1):
                for j in range(k, i):
                    prev = dp[k-1][j]
                    if prev == float('inf'): continue
                    # Comm Cost Penalty
                    curr = max(prev, prefix[i]-prefix[j] + 0.005)
                    if curr < dp[k][i]: dp[k][i] = curr; split[k][i] = j
        plan = {}
        curr = self.M
        for k in range(self.K-1, -1, -1):
            start = split[k][curr]
            names = []
            for n in self.nodes[start:curr]: names.extend(n.contained_names)
            plan[k] = names; curr = start
        return plan

class GraphPartitioner:
    def __init__(self, model, num_devices=4):
        self.traced = torch.fx.symbolic_trace(model); self.devices = num_devices
    
    def _get_cost(self, node, profile):
        if profile and node.name in profile: return profile[node.name]
        return 0.01
        
    def run_scdp(self, profile_data=None):
        g = AlgoGraph()
        for n in self.traced.graph.nodes:
            c = self._get_cost(n, profile_data)
            d = np.prod(n.meta['tensor_meta'].shape)*4/1e6 if 'tensor_meta' in n.meta and hasattr(n.meta['tensor_meta'], 'shape') else 0
            g.add_node(AlgoNode(n.name, c, d))
        for n in self.traced.graph.nodes:
            for u in n.users: g.add_edge(n.name, u.name, g.nodes[n.name].memory_footprint)
        
        # Smart Coarsening Bypass
        if len(g.nodes) < 500: coarsened = g
        else: coarsened = Coarsener(g).coarsen(self.devices*8)
        
        return DPPartitioner(coarsened, self.devices).solve()

    def run_baseline(self, profile_data=None):
        nodes = [(n.name, self._get_cost(n, profile_data)) for n in self.traced.graph.nodes if n.op in ['call_module','call_function','call_method']]
        avg = sum(c for _,c in nodes)/self.devices
        plan = {i:[] for i in range(self.devices)}
        curr = 0; s = 0
        for i, (n, c) in enumerate(nodes):
            plan[curr].append(n); s+=c
            rem_n = len(nodes)-(i+1); rem_r = self.devices-1-curr
            if curr < self.devices-1 and (rem_n <= rem_r or (s >= avg and rem_n > rem_r)):
                curr+=1; s=0
        return plan
