#pragma once

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, torch_type)                                                                        \
    if (((T).options().dtype() != (torch_type))) {                                                                     \
        std::cout << "Tensor Info:" << (T).options() << std::endl;                                                     \
        throw std::runtime_error("values must be " #torch_type);                                                       \
    }
// --------------------- PyTorch bindings for custom kernel -----------------------