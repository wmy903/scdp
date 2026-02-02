import torch
import torch.fx
import networkx as nx
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

@dataclass
class AlgoNode:
    id: str; compute_cost: float = 0.0; contained_names: List[str] = field(default_factory=list)
    def __post_init__(self):
        if not self.contained_names: self.contained_names = [self.id]
@dataclass
class AlgoEdge: u: str; v: str; tensor_bytes: float = 0.0 

class AlgoGraph:
    def __init__(self):
        self.nodes = {}; self.edges = {}; self.adj = {}; self.rev_adj = {}
    def add_node(self, node):
        self.nodes[node.id] = node
        if node.id not in self.adj: self.adj[node.id] = set(); self.rev_adj[node.id] = set()
    def remove_node(self, nid):
        if nid in self.nodes:
            del self.nodes[nid]; 
            if nid in self.adj: del self.adj[nid]
            if nid in self.rev_adj: del self.rev_adj[nid]
    def add_edge(self, u, v, size_bytes):
        if u not in self.nodes or v not in self.nodes: return
        if (u, v) in self.edges: self.edges[(u, v)].tensor_bytes += size_bytes
        else: self.edges[(u, v)] = AlgoEdge(u, v, size_bytes); self.adj[u].add(v); self.rev_adj[v].add(u)
    def remove_edge(self, u, v):
        if (u, v) in self.edges: del self.edges[(u, v)]
        if u in self.adj: self.adj[u].discard(v)
        if v in self.rev_adj: self.rev_adj[v].discard(u)

class Coarsener:
    def __init__(self, graph, profile, model_arch='cnn'):
        self.graph = graph; self.profile = profile; self.model_arch = model_arch
        self.total = sum(n.compute_cost for n in graph.nodes.values())
    
    def coarsen(self, max_nodes=60, k_stages=4, force_no_limit=False):
        curr = copy.deepcopy(self.graph)
        if force_no_limit: limit = -1.0 
        else:
            ideal = self.total / k_stages
            limit = ideal * 0.95 if self.model_arch == 'transformer' else ideal * 1.5 
        changed = True
        while changed and limit > 0:
            changed = False
            try:
                g = nx.DiGraph(); 
                for n in curr.nodes: g.add_node(n)
                for u,v in curr.edges: g.add_edge(u,v)
                topo = list(nx.topological_sort(g))
            except: topo = list(curr.nodes.keys())
            for i in range(len(topo)-1):
                u = topo[i]; 
                if u not in curr.nodes: continue
                succs = list(curr.adj.get(u, []))
                if len(succs)==1:
                    v = succs[0]
                    if v not in curr.nodes: continue
                    preds = list(curr.rev_adj.get(v, []))
                    if len(preds)==1 and preds[0]==u:
                        if (curr.nodes[u].compute_cost + curr.nodes[v].compute_cost) < limit:
                            self._merge(curr, u, v); changed = True
        while len(curr.nodes) > max_nodes:
            cands = list(curr.edges.values()); 
            if not cands: break
            cands.sort(key=lambda e: e.tensor_bytes, reverse=True)
            merged = False
            for e in cands:
                if e.u not in curr.nodes or e.v not in curr.nodes: continue
                should_merge = False
                cost_sum = curr.nodes[e.u].compute_cost + curr.nodes[e.v].compute_cost
                if limit > 0 and cost_sum < limit: should_merge = True
                elif limit < 0: should_merge = True 
                if should_merge:
                    self._merge(curr, e.u, e.v); merged = True; break
            if not merged: break
        return curr

    def _merge(self, g, u, v):
        un, vn = g.nodes[u], g.nodes[v]
        un.compute_cost += vn.compute_cost; un.contained_names.extend(vn.contained_names)
        for p in list(g.rev_adj[v]):
            if p!=u and (p,v) in g.edges: 
                w = g.edges[(p,v)].tensor_bytes; g.remove_edge(p,v); g.add_edge(p,u,w)
        for s in list(g.adj[v]):
            if s!=u and (v,s) in g.edges:
                w = g.edges[(v,s)].tensor_bytes; g.remove_edge(v,s); g.add_edge(u,s,w)
        g.remove_edge(u,v); g.remove_edge(v,u); g.remove_node(v)

class DPPartitioner:
    def __init__(self, graph, k):
        self.graph = graph; self.K = k; self.sorted_nodes = self._topo()
        self.comm_beta = 1.0 / (10 * 1024**3 / 1000.0)
    def _topo(self):
        g = nx.DiGraph(); 
        for n in self.graph.nodes: g.add_node(n)
        for u,v in self.graph.edges: g.add_edge(u,v)
        try: return list(nx.topological_sort(g))
        except: return list(self.graph.nodes.keys())
    def solve(self):
        N = len(self.sorted_nodes); K = self.K
        nodes = [self.graph.nodes[n] for n in self.sorted_nodes]
        pre = [0.0]*(N+1)
        for i in range(N): pre[i+1] = pre[i] + nodes[i].compute_cost
        comm_costs = [0.0]*N
        for i in range(N-1):
            u, v = nodes[i].id, nodes[i+1].id
            if (u,v) in self.graph.edges: comm_costs[i] = self.graph.edges[(u,v)].tensor_bytes * self.comm_beta
        def get_cost(s, e):
            return (pre[e] - pre[s]) + (comm_costs[e-1] if e < N else 0.0)
        dp = np.full((K+1, N+1), float('inf')); par = np.zeros((K+1, N+1), dtype=int); dp[0][0] = 0
        for k in range(1, K+1):
            for i in range(1, N+1):
                for j in range(i):
                    val = max(dp[k-1][j], get_cost(j, i))
                    if val < dp[k][i]: dp[k][i] = val; par[k][i] = j
        plan = {r:[] for r in range(K)}; stats = {r:{} for r in range(K)}; curr = N
        for k in range(K, 0, -1):
            prev = par[k][curr]
            stats[k-1] = {'compute': pre[curr]-pre[prev], 'comm': comm_costs[curr-1] if curr < N else 0.0}
            for idx in range(prev, curr): plan[k-1].extend(nodes[idx].contained_names)
            curr = prev
        return plan, stats

# === 改进：均匀分割算法 ===
class UniformPartitioner:
    def __init__(self, m, p, d):
        self.nodes=[n.name for n in m.graph.nodes if n.op in ['call_module','call_function','call_method']]
        self.costs=[p.get(n,0.0) for n in self.nodes]; self.d=d

    def solve(self):
        total_cost = sum(self.costs)
        target = total_cost / self.d
        plan = {i: [] for i in range(self.d)}
        stage = 0
        current_stage_cost = 0.0
        
        for n, c in zip(self.nodes, self.costs):
            # 只有当：1. 还没到最后一个 stage
            # 2. 加上当前节点后的总和 超过了 target
            # 3. 且 加上后的误差 比 仅仅保留现在的误差 更大（说明应该切了）
            # 才进行切分
            if stage < self.d - 1:
                cost_with_node = current_stage_cost + c
                if cost_with_node > target:
                    # 比较：是在这里切（归入下一级），还是加进去再切？
                    diff_exclude = abs(current_stage_cost - target)
                    diff_include = abs(cost_with_node - target)
                    if diff_include > diff_exclude and current_stage_cost > 0:
                        stage += 1
                        current_stage_cost = 0.0
            
            plan[stage].append(n)
            current_stage_cost += c
            
        return plan, None

class AdapipePartitioner:
    def __init__(self, m, p, d):
        self.nodes=[n.name for n in m.graph.nodes if n.op in ['call_module','call_function','call_method']]
        self.costs=[p.get(n,0.0) for n in self.nodes]; self.d=d
    def solve(self):
        l,h=max(self.costs) if self.costs else 0, sum(self.costs)
        cuts=[]; 
        for _ in range(30):
            t=(l+h)/2; s=0; st=1; c=[]; ok=True
            for i,v in enumerate(self.costs):
                if s+v>t: st+=1; s=v; c.append(i); 
                else: s+=v
                if st>self.d: ok=False; break
            if ok: cuts=c; h=t
            else: l=t
        b=[0]+cuts+[len(self.nodes)]; 
        while len(b)<self.d+1: b.insert(-1, b[-1])
        return {r: self.nodes[b[r]:b[r+1]] for r in range(self.d)}, None

class DagPPartitioner:
    def __init__(self, m, p, d):
        self.nodes=[n.name for n in m.graph.nodes if n.op in ['call_module','call_function','call_method']]
        self.costs=[p.get(n,0.0) for n in self.nodes]; self.d=d
    def solve(self):
        tgt=sum(self.costs)/self.d; plan={i:[] for i in range(self.d)}; r,s=0,0.0
        for n,c in zip(self.nodes, self.costs):
            if s+c>tgt and r<self.d-1: r+=1; s=0.0
            plan[r].append(n); s+=c
        return plan, None

class PicoPartitioner:
    def __init__(self, m, p, d):
        self.nodes=[n.name for n in m.graph.nodes if n.op in ['call_module','call_function','call_method']]
        self.costs=[p.get(n,0.0) for n in self.nodes]; self.d=d
    def _rec(self, s, e, k):
        if k==1: return [e]
        if (e-s) < k: return list(range(s+1, s+k+1))
        tot=sum(self.costs[s:e]); tgt=tot*(k//2/k); cur=0; spl=e; md=float('inf')
        for i in range(s,e):
            cur+=self.costs[i]; d=abs(cur-tgt)
            if d<md: md=d; spl=i+1
            else: break
        return self._rec(s, spl, k//2) + self._rec(spl, e, k-k//2)
    def solve(self):
        c=self._rec(0, len(self.nodes), self.d); b=sorted(list(set([0]+c)))
        while len(b)<self.d+1: b.append(b[-1])
        return {r: self.nodes[b[r]:b[r+1]] for r in range(self.d)}, None

class GraphPartitioner:
    def __init__(self, m, p, d, batch_size=32): 
        self.traced=torch.fx.symbolic_trace(m); self.profile_data=p; self.devices=d; self.bs=batch_size
    def _build(self):
        g=AlgoGraph(); valid=['call_module','call_function','call_method']
        for n in self.traced.graph.nodes:
            if n.op in valid:
                g.add_node(AlgoNode(n.name, compute_cost=self.profile_data.get(n.name,0.0)))
        for n in self.traced.graph.nodes:
            if n.op in valid:
                sz=0.0; tm=n.meta.get('tensor_meta')
                if tm and hasattr(tm, 'shape'): 
                    s=list(tm.shape); 
                    if s: s[0]=self.bs
                    sz=np.prod(s)*4.0
                elif tm and isinstance(tm,(tuple,list)) and len(tm)>0 and hasattr(tm[0],'shape'):
                    s=list(tm[0].shape); 
                    if s: s[0]=self.bs
                    sz=np.prod(s)*4.0
                for u in n.users:
                    if u.op in valid: g.add_edge(n.name, u.name, sz)
        return g
    def get_partition_plan(self, enable_coarsening=True, model_arch='cnn'):
        if not enable_coarsening: return DPPartitioner(self._build(), self.devices).solve()
        coarsener = Coarsener(self._build(), self.profile_data, model_arch)
        return DPPartitioner(coarsener.coarsen(self.devices*10, self.devices), self.devices).solve()
