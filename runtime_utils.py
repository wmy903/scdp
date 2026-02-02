import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
import torchvision.models as models
import numpy as np
import time
from torch.fx import Tracer

# === 1. Models ===
class MiniViT(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        self.l0 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
        self.l1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
        self.l2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
        self.l3 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
        self.l4 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
        self.l5 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
        self.l6 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
        self.l7 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
        self.head = nn.Linear(embed_dim, 1000)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.l0(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        x = self.l4(x); x = self.l5(x); x = self.l6(x); x = self.l7(x)
        x = x[:, 0]
        return self.head(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x): return self._norm(x.float()).type_as(x) * self.weight

class LlamaMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x): return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.input_layernorm = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.post_attention_layernorm = RMSNorm(dim)
        hidden_dim = int(2 * (4 * dim) / 3) 
        self.mlp = LlamaMLP(dim, hidden_dim)
    def forward(self, x):
        res = x; x = self.input_layernorm(x)
        attn_out, _ = self.self_attn(x, x, x); x = res + attn_out
        res = x; x = self.post_attention_layernorm(x); x = self.mlp(x)
        return res + x

class MiniLlama(nn.Module):
    def __init__(self, vocab_size=10000, dim=1024, num_heads=8, num_layers=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([LlamaBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers: x = layer(x)
        x = self.norm(x); return self.lm_head(x)

def get_model_and_input(model_name: str, batch_size=32):
    name = model_name.lower()
    if 'resnet101' in name:
        model = models.resnet101(weights=None); inp = torch.randn(batch_size, 3, 224, 224)
    elif 'resnet' in name:
        model = models.resnet50(weights=None); inp = torch.randn(batch_size, 3, 224, 224)
    elif 'mobilenet' in name:
        model = models.mobilenet_v3_large(weights=None); inp = torch.randn(batch_size, 3, 224, 224)
    elif 'vit' in name:
        model = MiniViT(); inp = torch.randn(batch_size, 3, 224, 224)
    elif 'llama' in name:
        model = MiniLlama(dim=1024, num_layers=16); inp = torch.randint(0, 1000, (batch_size, 64), dtype=torch.long)
    else: raise ValueError(f"Unknown model: {model_name}")
    return model, inp

# === 2. Configurable Leaf Tracer ===
class LeafTracer(Tracer):
    def __init__(self, coarse_llama=False):
        super().__init__()
        self.coarse_llama = coarse_llama

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # 1. Standard PyTorch leaves (Conv, Linear, etc.)
        is_leaf = super().is_leaf_module(m, module_qualified_name)
        
        # 2. Logic for Llama Coarsening
        if self.coarse_llama:
            # If coarse_llama is True, we WANT LlamaMLP and Attention to be atomic (leaves)
            if isinstance(m, (LlamaMLP, nn.MultiheadAttention)):
                return True
            # But we still want to look INSIDE LlamaBlock (so it's NOT a leaf)
            if isinstance(m, LlamaBlock):
                return False
            # For other models/layers, we stick to default (or force unfold if needed)
            
        # 3. Logic for General Fine-Grained Unfolding (ResNet, etc.)
        # If it's a container, force unfold it to see inside (unless it's one of our special coarse leaves)
        if isinstance(m, (nn.Sequential, models.resnet.Bottleneck, models.resnet.BasicBlock, LlamaBlock)):
            return False 
            
        return is_leaf

# === 3. Overhead-Aware Profiler ===
class OffloadProfiler(torch.fx.Interpreter):
    def __init__(self, module, device='cuda:0'):
        super().__init__(module)
        self.device = device
        self.node_costs = {}
        self.module.to('cpu')
        self.overhead = self._measure_overhead() 

    def _measure_overhead(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latencies = []
        for _ in range(100):
            torch.cuda.synchronize()
            start.record()
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
        overhead = np.percentile(latencies, 10)
        print(f"  [Profiler] Calibrated System Overhead: {overhead:.4f} ms per op")
        return overhead

    def run_node(self, n):
        if n.op in ['call_module', 'call_function', 'call_method']:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            
            def to_device(obj):
                if isinstance(obj, torch.Tensor): return obj.to(self.device)
                if isinstance(obj, (tuple, list)): return type(obj)(to_device(x) for x in obj)
                return obj
            
            try:
                gpu_args = to_device(args)
                gpu_kwargs = to_device(kwargs)
                
                target_mod = None
                if n.op == 'call_module':
                    target_mod = self.module.get_submodule(n.target)
                    target_mod.to(self.device)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                if n.op == 'call_module':
                    _ = target_mod(*gpu_args, **gpu_kwargs)
                
                torch.cuda.synchronize()
                start.record()
                
                if n.op == 'call_module':
                    output = target_mod(*gpu_args, **gpu_kwargs)
                elif n.op == 'call_function':
                    output = n.target(*gpu_args, **gpu_kwargs)
                elif n.op == 'call_method':
                    output = getattr(gpu_args[0], n.target)(*gpu_args[1:], **gpu_kwargs)
                
                end.record()
                torch.cuda.synchronize()
                
                raw_time = start.elapsed_time(end)
                self.node_costs[n.name] = max(0.001, raw_time - self.overhead)
            
            except Exception as e:
                self.node_costs[n.name] = 0.001
                output = torch.zeros(1)

            def to_cpu(obj):
                if isinstance(obj, torch.Tensor): return obj.detach().cpu()
                if isinstance(obj, (tuple, list)): return type(obj)(to_cpu(x) for x in obj)
                return obj
            
            cpu_output = to_cpu(output)
            if target_mod: target_mod.to('cpu')
            del gpu_args, gpu_kwargs, output
            return cpu_output
        else:
            return super().run_node(n)

def profile_model(model: nn.Module, sample_input: torch.Tensor, device='cuda:0', tracer=None):
    model = model.cpu(); sample_input = sample_input.cpu()
    
    # [FIX] Use Custom Tracer if provided, else default fine-grained
    if tracer is None:
        tracer = LeafTracer(coarse_llama=False)
        
    graph = tracer.trace(model)
    traced = torch.fx.GraphModule(model, graph)
    
    print(f"  [Profiler] Starting Profiling (Nodes: {len(traced.graph.nodes)})...")
    profiler = OffloadProfiler(traced, device)
    with torch.no_grad(): profiler.run(sample_input)
    return profiler.node_costs

# === 4. Pipeline Runtime ===
class PipelineStage(nn.Module):
    def __init__(self, module_or_traced, node_names: list, rank: int, world_size: int = 1, node_shapes: dict = None, model_arch: str = 'cnn', model_name: str = '', batch_size: int = 32):
        super().__init__()
        self.rank = rank; self.world_size = world_size
        self.node_shapes = node_shapes if node_shapes is not None else {}
        self.node_names = set(node_names); self.model_arch = model_arch; self.model_name = model_name
        self.batch_size = batch_size
        
        # Ensure we work on a graph module
        # Note: We assume module_or_traced is ALREADY traced with the correct granularity by the caller
        if not isinstance(module_or_traced, torch.fx.GraphModule):
             tracer = LeafTracer(coarse_llama=('llama' in model_name))
             graph = tracer.trace(module_or_traced)
             module_or_traced = torch.fx.GraphModule(module_or_traced, graph)
             
        self.source_module = module_or_traced
        self.sub_module = self._extract_subgraph(self.source_module)
        self.sub_module.eval(); self._infer_missing_shapes_bfs()

    def _extract_subgraph(self, traced_model):
        new_graph = torch.fx.Graph(); env = {}
        for node in traced_model.graph.nodes:
            if node.name in self.node_names:
                for input_node in node.all_input_nodes:
                    if input_node.name not in self.node_names:
                        if input_node not in env:
                            if input_node.op == 'get_attr': env[input_node] = new_graph.node_copy(input_node, lambda x: env.get(x, x))
                            else:
                                new_node = new_graph.placeholder(input_node.name)
                                if hasattr(input_node, 'meta'): new_node.meta = input_node.meta.copy()
                                env[input_node] = new_node
        for node in traced_model.graph.nodes:
            if node.name in self.node_names: env[node] = new_graph.node_copy(node, lambda x: env.get(x, x))
        outputs = []
        for node in traced_model.graph.nodes:
            if node.name in self.node_names:
                for user in node.users:
                    if user.name not in self.node_names:
                        if env[node] not in outputs: outputs.append(env[node]); break
        for node in traced_model.graph.nodes:
            if node.op == 'output':
                def recurse(n):
                    if isinstance(n, torch.fx.Node) and n.name in self.node_names:
                        if env[n] not in outputs: outputs.append(env[n])
                    elif isinstance(n, (tuple, list)):
                        for item in n: recurse(item)
                recurse(node.args)
        if len(outputs) == 1: new_graph.output(outputs[0])
        elif len(outputs) > 1: new_graph.output(tuple(outputs))
        else: new_graph.output(tuple())
        return torch.fx.GraphModule(traced_model, new_graph)

    def _infer_missing_shapes_bfs(self):
        pass

    def forward(self, *inputs):
        dummy_args = []; device = next(self.parameters()).device
        placeholders = [n for n in self.sub_module.graph.nodes if n.op == 'placeholder']
        input_list = list(inputs) if len(inputs) > 0 and inputs[0] is not None else []
        for i, node in enumerate(placeholders):
            if i < len(input_list): dummy_args.append(input_list[i]); continue
            shape = self.node_shapes.get(node.name, None)
            if shape is None:
                if 'vit' in self.model_name: shape = (self.batch_size, 197, 384)
                elif 'llama' in self.model_name: shape = (self.batch_size, 64, 1024)
                else: shape = (self.batch_size, 64, 56, 56)
            if shape:
                s = list(shape); s[0] = self.batch_size
                if 'llama' in self.model_name and len(s)==2:
                     dummy_args.append(torch.randint(0, 1000, tuple(s), device=device, dtype=torch.long))
                else:
                     dummy_args.append(torch.randn(tuple(s), device=device))
            else: dummy_args.append(torch.randn(self.batch_size, 64, 56, 56, device=device))
        return self.sub_module(*dummy_args)
