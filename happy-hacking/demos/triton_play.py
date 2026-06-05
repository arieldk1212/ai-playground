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

----------------------------

Docs: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
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
        # Can have multiple programs
        pid = tl.program_id(axis=0)

        # We process inputs that are offsets from the initial data.
        # For instance if we have a vector of length 256 and a block size of 64,
        # the programs would each access the elements: [0:64, 64:128, 128:192, 192:256]
        # The offsets is a list of pointers!
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # Create mask to guard memory operations against out-of-bounds accesses.
        mask: bool = offsets < n_elements

        # Load x and y out of dram, masking out any extra elements in case the input is not
        # a multiple of block size.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y

        # Write x and y back to dram
        tl.store(output_ptr + offsets, output, mask=mask)
