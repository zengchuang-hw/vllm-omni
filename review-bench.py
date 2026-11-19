import json
import sys

import torch
from vllm_omni.model_executor.models.fish_omni.sampling import FusedFrameSampler

case = sys.argv[1]
width = 64 if case == "base" else 4096
rows = 8
device = torch.device("cuda")
torch.manual_seed(9157)
sam = FusedFrameSampler(max_top_k=width, num_codebooks=10, max_rows=rows, device=device)
main = torch.randn(rows, 4096, device=device, dtype=torch.bfloat16)
heads = torch.randn(9, rows, 4096, device=device, dtype=torch.bfloat16)
ids = torch.arange(4096, device=device)
history = torch.randint(0, 4096, (rows, 16), device=device)
temp = torch.ones(rows, device=device)
pk = torch.full(
    (rows,), 30 if case == "base" else 4096, device=device, dtype=torch.long
)
pp = torch.full((rows,), 0.9 if case == "base" else 1.0, device=device)
noise = sam.noise(rows, device)


def draw():
    sam.main_tokens(main, ids, history, temp, pk, pp, noise, 0, 4096)
    for i in range(9):
        sam.code(heads[i], temp, pk, pp, noise, i)


for _ in range(6):
    draw()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    draw()
for _ in range(6):
    g.replay()
torch.cuda.synchronize()
a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
a.record()
for _ in range(100):
    g.replay()
b.record()
b.synchronize()
print(
    json.dumps(
        dict(
            case=case,
            batch=rows,
            width=width,
            frames=100,
            mean_us=a.elapsed_time(b) * 10,
            device=torch.cuda.get_device_name(),
            torch=torch.__version__,
        )
    ),
    flush=True,
)
