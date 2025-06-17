/*
https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/activation.py#L740
torch.nn.functional.hardswish(input, inplace=False)
               【  x,  if x > lamdb  】
HardShrink(x) =【  x,  if x > lamdb  】
               【  0   otherwise     】
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include "../data_base.h"

#define BLOCK_SIZE 256
#define LAMBD 0.5f

// hardshrink implement for one float element
__device__ __forceinline__ float hardshrink_float(float x) {
    if (x > LAMBD || x < -LAMBD) {
        return x;
    } else {
        return 0.f;
    }
}

// hardshrink implement for one half element
__device__ __forceinline__ half hardshrink_half(half x) {
    if (x > __float2half(LAMBD) || x < __float2half(-LAMBD)) {
        return x;
    } else {
        return __float2half(0.f);
    }
}

__global__ void hardshrink_f32_kernel(float *x, float *y, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) y[idx] = hardshrink_float(x[idx]);
}

__global__ void hardshrink_f32x4_kernel(float *x, float *y, int len) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < len) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = hardshrink_float(reg_x.x);
        reg_y.y = hardshrink_float(reg_x.y);
        reg_y.z = hardshrink_float(reg_x.z);
        reg_y.w = hardshrink_float(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

__global__ void hardshrink_f16_kernel(half *x, half *y, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) y[idx] = hardshrink_half(x[idx]);
}

__global__ void hardshrink_f16x2_kernel(half *x, half *y, int len) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < len) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = hardshrink_half(reg_x.x);
        reg_y.y = hardshrink_half(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

__global__ void hardshrink_f16x8_kernel(half *x, half *y, int len) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = hardshrink_half(reg_x_0.x);
    reg_y_0.y = hardshrink_half(reg_x_0.y);
    reg_y_1.x = hardshrink_half(reg_x_1.x);
    reg_y_1.y = hardshrink_half(reg_x_1.y);
    reg_y_2.x = hardshrink_half(reg_x_2.x);
    reg_y_2.y = hardshrink_half(reg_x_2.y);
    reg_y_3.x = hardshrink_half(reg_x_3.x);
    reg_y_3.y = hardshrink_half(reg_x_3.y);
    if ((idx + 0) < len) { HALF2(y[idx + 0]) = reg_y_0; }
    if ((idx + 2) < len) { HALF2(y[idx + 2]) = reg_y_1; }
    if ((idx + 4) < len) { HALF2(y[idx + 4]) = reg_y_2; }
    if ((idx + 6) < len) { HALF2(y[idx + 6]) = reg_y_3; }
}

__global__ void hardshrink_f16x8_pack_kernel(half *x, half *y, int len) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        pack_y[i] = hardshrink_half(pack_x[i]);
    }
    if ((idx + 7) < len) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }
}

// PyTorch 绑定代码
#define TORCH_BINDING_HARDSHRINK(packed_type, th_type, element_type, pack_num)                                         \
    void hardshrink_##packed_type(torch::Tensor x, torch::Tensor y) {                                                  \
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
            hardshrink_##packed_type##_kernel<<<grid, block>>>(                                                        \
                reinterpret_cast<element_type *>(x.data_ptr()), reinterpret_cast<element_type *>(y.data_ptr()), N);    \
        } else {                                                                                                       \
            const int S = x.size(0);                                                                                   \
            const int K = x.size(1);                                                                                   \
            const int N = S * K;                                                                                       \
            if ((K / (pack_num)) <= 1024) {                                                                            \
                dim3 block(K / pack_num);                                                                              \
                dim3 grid(S);                                                                                          \
                hardshrink_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),     \
                                                                   reinterpret_cast<element_type *>(y.data_ptr()),     \
                                                                   N);                                                 \
            } else {                                                                                                   \
                int N = 1;                                                                                             \
                for (int i = 0; i < ndim; ++i) {                                                                       \
                    N *= x.size(i);                                                                                    \
                }                                                                                                      \
                dim3 block(BLOCK_SIZE / pack_num);                                                                     \
                dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);                                                          \
                hardshrink_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),     \
                                                                   reinterpret_cast<element_type *>(y.data_ptr()),     \
                                                                   N);                                                 \
            }                                                                                                          \
        }                                                                                                              \
    }

// clang-format off
TORCH_BINDING_HARDSHRINK(f32,        torch::kFloat32, float, 1)
TORCH_BINDING_HARDSHRINK(f32x4,      torch::kFloat32, float, 4)
TORCH_BINDING_HARDSHRINK(f16,        torch::kHalf,    half,  1)
TORCH_BINDING_HARDSHRINK(f16x2,      torch::kHalf,    half,  2)
TORCH_BINDING_HARDSHRINK(f16x8,      torch::kHalf,    half,  8)
TORCH_BINDING_HARDSHRINK(f16x8_pack, torch::kHalf,    half,  8)
// clang-format on

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(hardshrink_f32)
    TORCH_BINDING_COMMON_EXTENSION(hardshrink_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16)
    TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16x2)
    TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16x8)
    TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16x8_pack)
}