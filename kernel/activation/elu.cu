/*
torch.nn.ELU(alpha=1.0, inplace=False)
ELU(x) =【  x,                      if x >  0  】
        【  alpha * (exp(x) - 1)    if x <= 0  】
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include "../data_base.h"

#define BLOCK_SIZE 256
#define ALPHA 1.0f

// elu implement for one float element
__device__ __forceinline__ float elu(float x) { return x > 0.f ? x : ALPHA * (expf(x) - 1.f); }

// elu implement for one half element
__device__ __forceinline__ half elu_half(half x) {
    return __hgt(x, __float2half(0.f)) ? x : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.f)));
}

__global__ void elu_f32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = elu(x[idx]);
}

__global__ void elu_f32x4_kernel(float *x, float *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = elu(reg_x.x);
        reg_y.y = elu(reg_x.y);
        reg_y.z = elu(reg_x.z);
        reg_y.w = elu(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

__global__ void elu_f16_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = elu_half(x[idx]);
}

__global__ void elu_f16x2_kernel(half *x, half *y, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = elu_half(reg_x.x);
        reg_y.y = elu_half(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

__global__ void elu_f16x8_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = elu_half(reg_x_0.x);
    reg_y_0.y = elu_half(reg_x_0.y);
    reg_y_1.x = elu_half(reg_x_1.x);
    reg_y_1.y = elu_half(reg_x_1.y);
    reg_y_2.x = elu_half(reg_x_2.x);
    reg_y_2.y = elu_half(reg_x_2.y);
    reg_y_3.x = elu_half(reg_x_3.x);
    reg_y_3.y = elu_half(reg_x_3.y);
    if ((idx + 0) < N) { HALF2(y[idx + 0]) = reg_y_0; }
    if ((idx + 2) < N) { HALF2(y[idx + 2]) = reg_y_1; }
    if ((idx + 4) < N) { HALF2(y[idx + 4]) = reg_y_2; }
    if ((idx + 6) < N) { HALF2(y[idx + 6]) = reg_y_3; }
}

__global__ void elu_f16x8_pack_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        pack_y[i] = elu_half(pack_x[i]);
    }
    if ((idx + 7) < N) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }
}

// PyTorch 绑定代码
#define TORCH_BINDING_ELU(packed_type, th_type, element_type, pack_num)                                                \
    void elu_##packed_type(torch::Tensor x, torch::Tensor y) {                                                         \
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
            elu_##packed_type##_kernel<<<grid, block>>>(                                                               \
                reinterpret_cast<element_type *>(x.data_ptr()), reinterpret_cast<element_type *>(y.data_ptr()), N);    \
        } else {                                                                                                       \
            const int S = x.size(0);                                                                                   \
            const int K = x.size(1);                                                                                   \
            const int N = S * K;                                                                                       \
            if ((K / (pack_num)) <= 1024) {                                                                            \
                dim3 block(K / pack_num);                                                                              \
                dim3 grid(S);                                                                                          \
                elu_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),            \
                                                            reinterpret_cast<element_type *>(y.data_ptr()),            \
                                                            N);                                                        \
            } else {                                                                                                   \
                int N = 1;                                                                                             \
                for (int i = 0; i < ndim; ++i) {                                                                       \
                    N *= x.size(i);                                                                                    \
                }                                                                                                      \
                dim3 block(BLOCK_SIZE / pack_num);                                                                     \
                dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);                                                          \
                elu_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),            \
                                                            reinterpret_cast<element_type *>(y.data_ptr()),            \
                                                            N);                                                        \
            }                                                                                                          \
        }                                                                                                              \
    }

// clang-format off
TORCH_BINDING_ELU(f32,        torch::kFloat32, float, 1)
TORCH_BINDING_ELU(f32x4,      torch::kFloat32, float, 4)
TORCH_BINDING_ELU(f16,        torch::kHalf,    half,  1)
TORCH_BINDING_ELU(f16x2,      torch::kHalf,    half,  2)
TORCH_BINDING_ELU(f16x8,      torch::kHalf,    half,  8)
TORCH_BINDING_ELU(f16x8_pack, torch::kHalf,    half,  8)
// clang-format on

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(elu_f32)
    TORCH_BINDING_COMMON_EXTENSION(elu_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(elu_f16)
    TORCH_BINDING_COMMON_EXTENSION(elu_f16x2)
    TORCH_BINDING_COMMON_EXTENSION(elu_f16x8)
    TORCH_BINDING_COMMON_EXTENSION(elu_f16x8_pack)
}