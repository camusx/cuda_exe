# 只编译Hopper架构,不指定默认编译所有架构,耗时较长: Ampere, Hopper,
export TORCH_CUDA_ARCH_LIST=Hopper
python3 hardswish.py --check_diff