# CUDA Performance Optimization

This repository is a collection of CUDA performance optimization studies spanning from operator-level kernels to application-level acceleration on modern NVIDIA GPUs.

Rather than providing a single highly specialized implementation, this project aims to demonstrate **systematic, profiling-driven optimization methodologies**, spanning from kernel-level tuning to system-level execution and data movement optimization.

---

## Project Goals

- Showcase common and reusable CUDA optimization techniques
- Demonstrate how performance bottlenecks evolve across different layers:
  - Kernel execution
  - Memory hierarchy
  - Host-Device interaction
  - Execution scheduling and overlap
- Emphasize **why** a given optimization works, not just **what** was applied
- Serve as a practical reference for real-world CUDA performance engineering

---

## Current Contents

### Vector Addition (`operators/vectorAdd/`)

A complete optimization walkthrough for a memory-bound vector addition operator, including:

- Baseline naive implementation
- Kernel-level optimizations:
  - Explicit memory semantics (`const __restrict__`)
  - Vectorized memory access (`float4`)
- Host–Device interaction optimizations:
  - Pinned (page-locked) memory
  - Asynchronous transfers
- System-level execution optimization:
  - Multi-stream pipelining
  - Overlap of data transfer and computation
- Performance analysis using:
  - Nsight Compute
  - Nsight Systems
- End-to-end execution time evaluation

This example serves as a minimal yet representative case study of CUDA performance optimization across multiple layers.

### General Matrix Multiplication (GEMM) (`operators/gemm/`)

A systematic, execution-centric optimization study of the GEMM operator on modern NVIDIA GPUs, focusing on **where performance bottlenecks actually come from** and **why certain optimizations work or fail**.

This module walks through the full optimization trajectory of a GEMM kernel, starting from a naive CUDA implementation and progressively introducing more advanced techniques, while carefully tracking how bottlenecks shift at each stage.

Key contents include:

* Baseline naive GEMM implementation and performance gap analysis against cuBLAS
* Execution-centric bottleneck identification using Nsight Compute:

  * Compute utilization
  * Warp stall analysis
  * Occupancy vs. memory latency
* Block-level optimizations:

  * Shared memory tiling
  * Register blocking
  * Trade-offs between data reuse, instruction pressure, and occupancy
* Warp-level exploration:

  * Register-level communication (`__shfl_sync`)
  * Structural limitations of warp-only designs for large GEMM problems
* Tensor Core programming with WMMA:

  * Integration into a block–warp–thread hierarchical design
  * Analysis of why Tensor Cores remain underutilized
  * Identification of load–compute serialization and shared memory bank conflicts
* Critical evaluation of common optimization ideas:

  * K-dimension double buffering
  * Shared memory swizzling
  * Increasing per-warp compute density
  * Why these approaches are fundamentally constrained under the WMMA programming model
* Architectural perspective:

  * `cp.async + ldmatrix + mma.sync` as the software-level peak on Ampere
  * TMA + WGMMA on Hopper as a shift from thread scheduling to tile description
* Methodological conclusion:

  * Identification of the practical optimization boundary of hand-written CUDA C++ kernels
  * Motivation for transitioning to framework-level solutions (CUTLASS, cuBLASLt, Triton)

This GEMM study emphasizes that performance optimization is not a linear accumulation of tricks, but a process of **bottleneck migration and constraint tightening**, and illustrates where low-level kernel tuning ends and higher-level operator composition begins.


---

## Future Work

This repository will be continuously extended with additional operator-level and application-level optimization studies, including but not limited to:

### Operator-Level Optimizations
- GEMM
- FlashAttention
- Reduction and scan operators
- Fused and composite kernels

### Application-Level & Algorithm-Level Optimizations
- Large Language Model (LLM) inference acceleration
- Time-series processing algorithms
- End-to-end GPU pipeline optimization
- Memory and execution scheduling strategies at scale

These future additions will build upon the same profiling-driven methodology demonstrated in the current operator examples.

---

## Philosophy

The goal of this project is not to create a universal, production-ready operator library, but to document **how performance bottlenecks are identified, analyzed, and resolved** on modern GPUs.

By focusing on clarity, reproducibility, and reasoning, this repository aims to bridge the gap between CUDA programming tutorials and large-scale performance engineering frameworks.

---

## Environment

- NVIDIA GPUs (RTX 5060 Ti)
- CUDA Toolkit 13.1
- Profiling tools:
  - Nsight Compute
  - Nsight Systems

> Note: While most experiments in this repository are conducted on an RTX 5060 Ti, the optimization principles and analysis methodology are hardware-agnostic and applicable to data center GPUs such as NVIDIA A100 / H200.