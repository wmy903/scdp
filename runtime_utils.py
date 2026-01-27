import torch
import torch.nn as nn
import torch.fx
import torchvision.models as models
import numpy as np

# === 1. Trace-Friendly Models ===

class MiniViT(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, num_layers=8):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, 1000)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0]
        x = self.head(x)
        return x

class MiniLlamaBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        res = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x)
        x = res + attn_out
        res = x
        x = self.ln2(x)
        mlp_out = self.mlp(x)
        return res + mlp_out

class MiniLlama(nn.Module):
    def __init__(self, vocab_size=10000, dim=512, num_heads=8, num_layers=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MiniLlamaBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x

# === 2. Model Loader ===

def get_model_and_input(model_name: str):
    name = model_name.lower()
    if 'resnet' in name:
        model = models.resnet50(weights=None)
        inp = torch.randn(1, 3, 224, 224)
    elif 'mobilenet' in name:
        model = models.mobilenet_v2(weights=None)
        inp = torch.randn(1, 3, 224, 224)
    elif 'vit' in name:
        model = MiniViT()
        inp = torch.randn(1, 3, 224, 224)
    elif 'llama' in name:
        model = MiniLlama()
        inp = torch.randint(0, 1000, (1, 64), dtype=torch.long)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, inp

# === 3. Real Profiler ===

class FXProfiler(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.node_costs = {}

    def run_node(self, n):
        # 仅测量计算节点
        if n.op in ['call_module', 'call_function', 'call_method']:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = super().run_node(n)
            end_event.record()
            torch.cuda.synchronize()
            self.node_costs[n.name] = start_event.elapsed_time(end_event)
            return result
        else:
            self.node_costs[n.name] = 0.0
            return super().run_node(n)

def profile_model(model: nn.Module, sample_input: torch.Tensor, device='cuda:0'):
    model = model.to(device)
    sample_input = sample_input.to(device)
    traced = torch.fx.symbolic_trace(model)
    
    print("  [Profiler] Warming up...")
    with torch.no_grad():
        for _ in range(5): traced(sample_input)
    torch.cuda.synchronize()
    
    print("  [Profiler] Running per-node measurement...")
    profiler = FXProfiler(traced)
    with torch.no_grad():
        profiler.run(sample_input)
    
    return profiler.node_costs

# === 4. Pipeline Runtime (Fixed Parameter Replication) ===

class PipelineStage(nn.Module):
    def __init__(self, full_model: torch.nn.Module, node_names: list, rank: int):
        super().__init__()
        self.rank = rank
        self.node_names = set(node_names)
        self.sub_module = self._extract_subgraph(full_model)
        self.sub_module.eval() 

    def _extract_subgraph(self, full_model):
        traced = torch.fx.symbolic_trace(full_model)
        new_graph = torch.fx.Graph()
        env = {}
        
        # 1. Handle Inputs (Placeholders vs Parameters)
        for node in traced.graph.nodes:
            if node.name in self.node_names:
                for input_node in node.all_input_nodes:
                    if input_node.name not in self.node_names:
                        # 依赖外部节点
                        if input_node not in env:
                            # [FIX] 如果是参数(get_attr)，直接复制节点，不要做成 Placeholder！
                            if input_node.op == 'get_attr':
                                env[input_node] = new_graph.node_copy(input_node, lambda x: env.get(x, x))
                            else:
                                # 真正的激活值输入 -> Placeholder
                                env[input_node] = new_graph.placeholder(input_node.name)
        
        # 2. Copy Internal Nodes
        for node in traced.graph.nodes:
            if node.name in self.node_names:
                env[node] = new_graph.node_copy(node, lambda x: env.get(x, x))
        
        # 3. Handle Outputs
        outputs = []
        for node in traced.graph.nodes:
            if node.name in self.node_names:
                for user in node.users:
                    if user.name not in self.node_names:
                        if env[node] not in outputs: outputs.append(env[node])
                        break
        for node in traced.graph.nodes:
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

        return torch.fx.GraphModule(full_model, new_graph)

    def forward(self, *inputs):
        return self.sub_module(*inputs)
