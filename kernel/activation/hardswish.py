import argparse
import torch
import time 
from torch.utils.cpp_extension import load
from typing import Optional
from functools import partial
import torch.nn.functional as F

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--check_diff', action='store_true', help='Wheather check correctness or not')

lib = load(name='hardswish_lib',
           sources=['hardswish.cu'],
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ], 
           extra_cflags=['std=c++17'])

def check_result(torch_result: torch.Tensor, gpu_result: torch.Tensor, diff: float):
    if torch.allclose(torch_result, gpu_result, atol=diff):
        return "test pass"
    else:
        return "test fail"


def run_benchmark(perf_func: callable, tag: str, x: torch.Tensor,
                  out: Optional[torch.Tensor] = None, check_diff: bool = False, 
                  diff: float = 1e-6, golden: Optional[torch.Tensor] = None):
    warmup = 10
    iters = 1000
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)

    # profiling begin    
    torch.cuda.synchronize()
    start = time.time()

    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    # profiling end
    
    total_time = (end - start) * 1000 # 单位(ms)
    mean_time = total_time / iters
    
    out_info = f"out_{tag}"
    res = "perf only"
    if check_diff: res = check_result(golden, out, diff)
    print(f"{out_info:>16}: {res}, time:{mean_time:.8f}ms, ")
    return out, mean_time

def torch_hardswish(x, out=None):
    if out is None:
        return F.hardswish(x)
    else:
        out.copy_(F.hardswish(x))
        return out

Ss = [1, 1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

args = parser.parse_args()

for (S, K) in SKs:
    # check wheather compare kernel result with torch result
    check_diff = args.check_diff
    diff = 1e-6

    print("-" * 50)
    print(" " * 20 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    y_torch = torch.zeros_like(x).cuda().float().contiguous()

    run_benchmark(torch_hardswish, "f32_th", x, y_torch)
    run_benchmark(lib.hardswish_f32,      "f32",    x, y, check_diff, diff, y_torch)
    run_benchmark(lib.hardswish_f32x4,    "f32x4",  x, y, check_diff, diff, y_torch)

    print("-" * 50)
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    y_f16_torch = y.half().contiguous()
    run_benchmark(torch_hardswish,          "f16_th",    x_f16, y_f16_torch)
    run_benchmark(lib.hardswish_f16,        "f16",       x_f16, y_f16, check_diff, diff, y_f16_torch)
    run_benchmark(lib.hardswish_f16x2,      "f16x2",     x_f16, y_f16, check_diff, diff, y_f16_torch)
    run_benchmark(lib.hardswish_f16x8,      "f16x8",     x_f16, y_f16, check_diff, diff, y_f16_torch)
    run_benchmark(lib.hardswish_f16x8_pack, "f16x8pack", x_f16, y_f16, check_diff, diff, y_f16_torch)

    print("-" * 50)