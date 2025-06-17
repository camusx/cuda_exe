import argparse
import torch
import time 
from torch.utils.cpp_extension import load
from typing import Optional
from functools import partial

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--check_diff', action='store_true', help='Wheather check correctness or not')

lib = load(name='elementwise_lib',
           sources=['elementwise.cu'],
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

def run_benchmark(perf_func: callable, tag: str, a: torch.Tensor, b: torch.Tensor, 
                  out: Optional[torch.Tensor] = None, check_diff: bool = False, 
                  diff: float = 1e-6, golden: Optional[torch.Tensor] = None):
    warmup = 10
    iters = 1000
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    # profiling begin    
    torch.cuda.synchronize()
    start = time.time()

    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
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
    a = torch.randn((S, K)).cuda().float().contiguous()
    b = torch.randn((S, K)).cuda().float().contiguous()
    c = torch.zeros_like(a).cuda().float().contiguous()
    c_torch = torch.zeros_like(a).cuda().float().contiguous()

    run_benchmark(partial(torch.add, out=c_torch), "f32_torch", a, b)
    run_benchmark(lib.elementwise_add_f32,         "f32",       a, b, c, check_diff, diff, c_torch)
    run_benchmark(lib.elementwise_add_f32x4,       "f32x4",     a, b, c, check_diff, diff, c_torch)

    print("-" * 50)
    a_f16 = a.half().contiguous()
    b_f16 = b.half().contiguous()
    c_torch = c.half().contiguous()
    c_gpu = c.half().contiguous()
    
    run_benchmark(partial(torch.add, out=c_torch), "f16_torch", a_f16, b_f16)
    run_benchmark(lib.elementwise_add_f16,         "f16",       a_f16, b_f16, c_gpu, check_diff, diff, c_torch)
    run_benchmark(lib.elementwise_add_f16x2,       "f16x2",     a_f16, b_f16, c_gpu, check_diff, diff, c_torch)
    run_benchmark(lib.elementwise_add_f16x8,       "f16x8",     a_f16, b_f16, c_gpu, check_diff, diff, c_torch)
    run_benchmark(lib.elementwise_add_f16x8_pack,  "f16x8pack", a_f16, b_f16, c_gpu, check_diff, diff, c_torch)

    print("-" * 50)