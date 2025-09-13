import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import time

softmax_cuda = load(
    name="softmax_cuda",
    sources=["softmax.cu"],
    extra_cuda_cflags=["-O3", "-arch=sm_86"]
)

def benchmark(fn, inp, n_warmup=10, n_iter=100):
    # Warmup
    for _ in range(n_warmup):
        fn(inp)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iter):
        out = fn(inp)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / n_iter
    return elapsed_ms, out

if __name__ == "__main__":
    torch.manual_seed(0)
    n = 512
    x = torch.randn(n, device='cuda', dtype=torch.float32)

    # Benchmark PyTorch softmax
    pt_time, pt_out = benchmark(lambda t: F.softmax(t, dim=0), x)
    print(f"PyTorch softmax: {pt_time:.3f} ms")

    # Benchmark custom CUDA softmax
    custom_time, custom_out = benchmark(lambda t: softmax_cuda.softmax_cuda(t), x)
    print(f"Custom softmax: {custom_time:.3f} ms")

    # Check numerical closeness
    close = torch.allclose(pt_out, custom_out, atol=1e-6)
    print(f"Close: {close}")
    # print(pt_out)
    # print(custom_out)
