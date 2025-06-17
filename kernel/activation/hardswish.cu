/*
https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/functional.py#L2403
torch.nn.Hardswish(lambd=0.5)
               【  0                 if x <= -3  】
HardSwish(x) = 【  x                 if x >= +3  】
               【  x * (x + 3) / 6   otherwise   】
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include "../data_base.h"

#define BLOCK_SIZE 256

// hardswish implement for one float element
__device__ __forceinline__ float hardswish_float(float x) {
    if (x <= -3.f) {
        return 0.f;
    } else if (x >= 3.f) {
        return 3.f;
    } else {
        return x * (x + 3.f) / 6.f;
    }
}

// hardswish implement for one half element
__device__ __forceinline__ half hardswish_half(half x) {
    if (x <= __float2half(-3.f)) {
        return 0;
    } else if (x >= __float2half(3.f)) {
        return x;
    } else {
        return x * (x + __float2half(3.f)) / __float2half(6.f);
    }
}

__global__ void hardswish_f32_kernel(float *x, float *y, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) y[idx] = hardswish_float(x[idx]);
}

__global__ void hardswish_f32x4_kernel(float *x, float *y, int len) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < len) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = hardswish_float(reg_x.x);
        reg_y.y = hardswish_float(reg_x.y);
        reg_y.z = hardswish_float(reg_x.z);
        reg_y.w = hardswish_float(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

__global__ void hardswish_f16_kernel(half *x, half *y, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) y[idx] = hardswish_half(x[idx]);
}

__global__ void hardswish_f16x2_kernel(half *x, half *y, int len) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < len) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = hardswish_half(reg_x.x);
        reg_y.y = hardswish_half(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

__global__ void hardswish_f16x8_kernel(half *x, half *y, int len) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = hardswish_half(reg_x_0.x);
    reg_y_0.y = hardswish_half(reg_x_0.y);
    reg_y_1.x = hardswish_half(reg_x_1.x);
    reg_y_1.y = hardswish_half(reg_x_1.y);
    reg_y_2.x = hardswish_half(reg_x_2.x);
    reg_y_2.y = hardswish_half(reg_x_2.y);
    reg_y_3.x = hardswish_half(reg_x_3.x);
    reg_y_3.y = hardswish_half(reg_x_3.y);
    if ((idx + 0) < len) { HALF2(y[idx + 0]) = reg_y_0; }
    if ((idx + 2) < len) { HALF2(y[idx + 2]) = reg_y_1; }
    if ((idx + 4) < len) { HALF2(y[idx + 4]) = reg_y_2; }
    if ((idx + 6) < len) { HALF2(y[idx + 6]) = reg_y_3; }
}

__global__ void hardswish_f16x8_pack_kernel(half *x, half *y, int len) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        pack_y[i] = hardswish_half(pack_x[i]);
    }
    if ((idx + 7) < len) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }
}

// PyTorch 绑定代码
#define TORCH_BINDING_HARDSWISH(packed_type, th_type, element_type, pack_num)                                          \
    void hardswish_##packed_type(torch::Tensor x, torch::Tensor y) {                                                   \
        CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                                                         \
        CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                                                         \
        const int ndim = x.dim();                                                                                      \
        if (ndim != 2) {                                                                                               \
            int N = 1;                                                                                                 \
            for (int i = 0; i < ndim; ++i) {                                                                           \
                N *= x.size(i);                                                                                        \
            }                                                                                                          \
            dim3 block(BLOCK_SIZE / pack_num);                                                                         \
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);                                                              \
            hardswish_##packed_type##_kernel<<<grid, block>>>(                                                         \
                reinterpret_cast<element_type *>(x.data_ptr()), reinterpret_cast<element_type *>(y.data_ptr()), N);    \
        } else {                                                                                                       \
            const int S = x.size(0);                                                                                   \
            const int K = x.size(1);                                                                                   \
            const int N = S * K;                                                                                       \
            if ((K / (pack_num)) <= 1024) {                                                                            \
                dim3 block(K / pack_num);                                                                              \
                dim3 grid(S);                                                                                          \
                hardswish_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),      \
                                                                  reinterpret_cast<element_type *>(y.data_ptr()),      \
                                                                  N);                                                  \
            } else {                                                                                                   \
                int N = 1;                                                                                             \
                for (int i = 0; i < ndim; ++i) {                                                                       \
                    N *= x.size(i);                                                                                    \
                }                                                                                                      \
                dim3 block(BLOCK_SIZE / pack_num);                                                                     \
                dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);                                                          \
                hardswish_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),      \
                                                                  reinterpret_cast<element_type *>(y.data_ptr()),      \
                                                                  N);                                                  \
            }                                                                                                          \
        }                                                                                                              \
    }

// clang-format off
TORCH_BINDING_HARDSWISH(f32,        torch::kFloat32, float, 1)
TORCH_BINDING_HARDSWISH(f32x4,      torch::kFloat32, float, 4)
TORCH_BINDING_HARDSWISH(f16,        torch::kHalf,    half,  1)
TORCH_BINDING_HARDSWISH(f16x2,      torch::kHalf,    half,  2)
TORCH_BINDING_HARDSWISH(f16x8,      torch::kHalf,    half,  8)
TORCH_BINDING_HARDSWISH(f16x8_pack, torch::kHalf,    half,  8)
// clang-format on

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(hardswish_f32)
    TORCH_BINDING_COMMON_EXTENSION(hardswish_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(hardswish_f16)
    TORCH_BINDING_COMMON_EXTENSION(hardswish_f16x2)
    TORCH_BINDING_COMMON_EXTENSION(hardswish_f16x8)
    TORCH_BINDING_COMMON_EXTENSION(hardswish_f16x8_pack)
}