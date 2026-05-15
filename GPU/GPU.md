# Studies

1. More compute cores : around 144 SM, in CPU its mostly 6 cores
2. cpu has higher clock rate (times 2).

* FLOPs (Floating point operations per second) per cycle - CPU: 1, GPU: 1024

* at the hardware CPU has ALU, GPU has Tensor Core

## Memory Hierachy

There are 3 types of main memories inside the gpu:

  1. Register Memory
  2. Shared Memory
  3. Device Memory (HBM) - VRAM

  A memory basically means that its programmer manageable for the user

There are 2 types of cache inside the gpu:
  Done Automatic

  1. L1 cache
  2. L2 cache - on chip - 2 partitions! stores items in HBM!

  Both of them cache items for the HBM
  L1 cache and shared memory both share the save device (DRAM)

## 3 GPU Compute Units

  1. GPC - Graphics Processing Cluster -> 8 of them, its how they scale - the hardware.
  2. Thread Block Cluster -> The api for the GPC, how to use it.
  3. SM -> mostly 18 SM inside a GPC, each one has own shared memory and L1 cache.
  4. Quadrant -> 4 Quadrants inside SM, each one has its own register memory.

## Quadrant Deep-Dive

1. Wrap Schedualer -> Issues Instructions, stress relief, commander of the compute units, slow! one instruction at a time.. therefore the Tensor Core exists.
2. FP32 -> Cuda cores, its same as ALU (adder or subtractor.. scalar ops)
3. Tensor Core 4th Generation - Parallalizes all the workers, it makes the compute fast, can launch one instruction to compute an entire matrices multiplication -> current can compute 256x256x16 (blackwell architecture - <https://www.nvidia.com/en-eu/data-center/technologies/blackwell-architecture/>)

## Writing Kernels

### Quick Recap of the GPU Memory

1. Register Memory (RMEM) - Thread: Has private registers, synchronization - SMEM
2. L1 cache / Shared Memory (SMEM) - a group of up to 1024 threads, guaranteed to be concurrently scheduled on a single SM, multiple thread blocks are independently scheduled across SMs syncchronozation.
3. Device Memory (DMEM) - thread block cluster - a group of up to 8 thread blocks, guaranteed to concurrently scheduled on a GPC synchronozation.
4. L2 cache - group of clusters or thread blocks synchronization. L2 or GMEM.
5. Device Memory (DMEM).

* Check out code snippet - link in prac.cuh.
