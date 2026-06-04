import torch
import triton
import triton.language as tl

"""
Model -> Pytorch/Tensorflow/JAX/Tinygrad
|
|
Graph -> XLA HLO/Torch Inductor/TVM
|
|
Kernel -> CUDA/HIP/OpenCL/Triton
|
|
Hardware -> GPU/CPU/TPU/ASIC

docs: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
"""

DEVICE = triton.runtime.driver.active.get_active_torch_device()


class KernelCalls:
    @staticmethod
    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,  # size of the vector.
        BLOCK_SIZE=tl.constexpr,  # number of elements each program should process.
    ):
        pid = tl.program_id(axis=0)
