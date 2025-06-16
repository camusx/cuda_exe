#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

#define BLOCK_SIZE 256
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void ew_add_f32(float *a, float *b, float *c, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) c[idx] = a[idx] + b[idx];
}

__global__ void ew_add_f32x4(float *a, float *b, float *c, int len) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < len) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c;
    }
}

__global__ void ew_add_f16(half *a, half *b, half *c, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) c[idx] = __hadd(a[idx], b[idx]);
}

__global__ void ew_add_f16x2(half *a, half *b, half *c, int len) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < len) {
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c;
        reg_c.x = __hadd(reg_a.x, reg_b.x);
        reg_c.y = __hadd(reg_a.y, reg_b.y);
        HALF2(c[idx]) = reg_c;
    }
}

__global__ void ew_add_f16x8(half *a, half *b, half *c, int len) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_a_0 = HALF2(a[idx + 0]);
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);
    half2 reg_b_0 = HALF2(b[idx + 0]);
    half2 reg_b_1 = HALF2(b[idx + 2]);
    half2 reg_b_2 = HALF2(b[idx + 4]);
    half2 reg_b_3 = HALF2(b[idx + 6]);
    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
    reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
    reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
    reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
    reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
    reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
    reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
    reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
    reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
    if ((idx + 0) < len) { HALF2(c[idx + 0]) = reg_c_0; }
    if ((idx + 2) < len) { HALF2(c[idx + 2]) = reg_c_1; }
    if ((idx + 4) < len) { HALF2(c[idx + 4]) = reg_c_2; }
    if ((idx + 6) < len) { HALF2(c[idx + 6]) = reg_c_3; }
}

__global__ void ew_add_f16x8_pack(half *a, half *b, half *c, int len) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    // temporary register(memory), .local space in ptx, addressable
    half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.
    // reinterpret as float4 and load 128 bits in 1 memory issue.
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits

#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        // __hadd2 for half2 x 4
        HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }
    // reinterpret as float4 and store 128 bits in 1 memory issue.
    if ((idx + 7) < len) { LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]); }
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, torch_type)                                                                        \
    if (((T).options().dtype() != (torch_type))) {                                                                     \
        std::cout << "Tensor Info:" << (T).options() << std::endl;                                                     \
        throw std::runtime_error("values must be " #torch_type);                                                       \
    }

#define TORCH_BINDING_ELEM_ADD(packed_type, torch_type, element_type, pack_num)                                        \
    void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                            \
        CHECK_TORCH_TENSOR_DTYPE(a, (torch_type))                                                                      \
        CHECK_TORCH_TENSOR_DTYPE(b, (torch_type))                                                                      \
        CHECK_TORCH_TENSOR_DTYPE(c, (torch_type))                                                                      \
        const int ndim = a.dim();                                                                                      \
        if (ndim != 2) {                                                                                               \
            int N = 1;                                                                                                 \
            for (int i = 0; i < ndim; ++i) {                                                                           \
                N *= a.size(i);                                                                                        \
            }                                                                                                          \
            dim3 block(BLOCK_SIZE / pack_num);                                                                         \
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);                                                              \
            ew_add_##packed_type<<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),                      \
                                                  reinterpret_cast<element_type *>(b.data_ptr()),                      \
                                                  reinterpret_cast<element_type *>(c.data_ptr()),                      \
                                                  N);                                                                  \
        } else {                                                                                                       \
            const int S = a.size(0);                                                                                   \
            const int K = a.size(1);                                                                                   \
            const int N = S * K;                                                                                       \
            if ((K / pack_num) <= 1024) {                                                                              \
                dim3 block(K / pack_num);                                                                              \
                dim3 grid(S);                                                                                          \
                ew_add_##packed_type<<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),                  \
                                                      reinterpret_cast<element_type *>(b.data_ptr()),                  \
                                                      reinterpret_cast<element_type *>(c.data_ptr()),                  \
                                                      N);                                                              \
            } else {                                                                                                   \
                int N = 1;                                                                                             \
                for (int i = 0; i < ndim; ++i) {                                                                       \
                    N *= a.size(i);                                                                                    \
                }                                                                                                      \
                dim3 block(BLOCK_SIZE / pack_num);                                                                     \
                dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);                                                          \
                ew_add_##packed_type<<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),                  \
                                                      reinterpret_cast<element_type *>(b.data_ptr()),                  \
                                                      reinterpret_cast<element_type *>(c.data_ptr()),                  \
                                                      N);                                                              \
            }                                                                                                          \
        }                                                                                                              \
    }

// clang-format off
TORCH_BINDING_ELEM_ADD(f32,         torch::kFloat32,    float,    1)
TORCH_BINDING_ELEM_ADD(f32x4,       torch::kFloat32,    float,    4)
TORCH_BINDING_ELEM_ADD(f16,         torch::kHalf,       half,     1)
TORCH_BINDING_ELEM_ADD(f16x2,       torch::kHalf,       half,     2)
TORCH_BINDING_ELEM_ADD(f16x8,       torch::kHalf,       half,     8)
TORCH_BINDING_ELEM_ADD(f16x8_pack,  torch::kHalf,       half,     8)
// clang-format on

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}