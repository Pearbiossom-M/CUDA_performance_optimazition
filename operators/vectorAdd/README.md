# Vector Addition Optimization Report

## 1. Background & Problem Definition

1. **Operator Description**
   - Vector addition: C[i] = A[i] + B[i]
2. **Application Scenario**
   - This is the most fundamental yet memory-bound operator commonly seen in deep learning
3. **Hardware & Software Environment**
   - GPU Model: RTX 5060 Ti
   - CUDA version: 13.1
   - Compilation flags: Default `-O3`

## 2. Naive Implementation & Baseline Performance

1. **Naive kernel code is shown below, full code available in `naive.cu`**

   ```c++
   template <typename T>
   __global__ void vectorAdd_naive(const T* d_a, const T* d_b, T* d_c, size_t N) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < N) {
           d_c[idx] = d_a[idx] + d_b[idx];
       }   
   }
   ```

2. **Theoretical Analysis**

   Assuming vector length is N, with each element being a 4-byte `float` type.

   - **Compute Floating-Point Operations (FLOPs)**

     Each element requires one addition:

     Total operations = **N FLOPs**

   - **Memory Access Volume (Bytes)**

     Each element requires:

     - Read A[i]: 4 Bytes
     - Read B[i]: 4 Bytes
     - Write C[i]: 4 Bytes

     Total memory access = **12N Bytes**

   - **Arithmetic Intensity (FLOPS / Byte)**

     Arithmetic intensity = N FLOPs / 12N bytes = 1/12 ≈ **0.083 FLOPs/byte**

     This extremely low value indicates:

     - **Memory-bound**: Computation is far less than data movement
     - **GPU actual utilization**: Only 0.083 operations per byte, causing severe compute resource waste

3. **Key Performance Metrics from Nsight Compute**

   | Metric                      | Value |
   | :-------------------------- | :---- |
   | Duration [ms]               | 4.02  |
   | Compute (SM) Throughput [%] | 9.66  |
   | Memory Throughput [%]       | 89.88 |
   | DRAM Throughput [%]         | 89.88 |
   | L2 Cache Throughput [%]     | 24.29 |
   | Achieved Occupancy [%]      | 80.26 |

4. **Analysis**

   The naive version achieves nearly 90% memory bandwidth utilization, primarily due to fully coalesced memory access—each warp's 32 threads access contiguous 128-byte data, which is automatically coalesced by the hardware into a single memory transaction. The kernel's minimal computation (single FMA instruction) causes it to quickly hit the memory wall, making it a typical bandwidth-saturated workload. Based on the Roofline model, with the memory bound already evident, subsequent optimizations will focus on precise improvements in instruction efficiency and system-level parallelism.

## 3. Kernel-Level Optimization

### 1. Compiler-Friendly Memory Semantics

1. Using `const __restrict__` explicitly expresses read-only and no-alias semantics, providing optimization opportunities for potential load reordering, instruction merging, and cache path selection. Although modern NVCC with optimization level `-O3` can automatically infer equivalent information, `const __restrict__` remains the strongest and most reliable semantic declaration in complex kernels.

2. **Modified kernel:**

   ```c++
   template <typename T>
   __global__ void vectorAdd_restrict(const T* __restrict__ d_a, const T* __restrict__ d_b, T* __restrict__ d_c, size_t N) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < N) {
           d_c[idx] = d_a[idx] + d_b[idx];
       }   
   }
   ```

3. **Key Performance Metrics from Nsight Compute**

   | Metric                      | Value |
   | :-------------------------- | :---- |
   | Duration [ms]               | 4.02  |
   | Compute (SM) Throughput [%] | 9.67  |
   | Memory Throughput [%]       | 89.84 |
   | DRAM Throughput [%]         | 89.84 |
   | L2 Cache Throughput [%]     | 24.29 |
   | Achieved Occupancy [%]      | 80.22 |

4. **Analysis**

   Experimental results show that using `const __restrict__` yields performance virtually identical to the naive implementation. This is expected on modern CUDA architectures.

   From a performance perspective, this kernel exhibits typical memory-bandwidth-bound behavior: each thread performs only two global memory loads, one floating-point addition, and one global store, resulting in extremely low compute intensity. Without loop structures, data reuse, cross-instruction dependencies, or potential store-load conflicts, even conservative pointer alias assumptions won't introduce additional synchronization or redundant memory operations. Consequently, strengthening no-alias semantics via `__restrict__` doesn't yield significant gains for this operator.

   At the SASS instruction level, introducing `const __restrict__` changes load instructions from `LDG.E` to `LDG.E.CONSTANT`, demonstrating that the compiler leverages stronger read-only semantics. In Volta and later architectures, global memory accesses share a unified L1 data cache. Different load instructions vary primarily in consistency models and dependency constraints; they don't guarantee physically separate caches or reduced latency. The `LDG.E.CONSTANT` advantage lies in relaxed memory visibility constraints, enabling more aggressive instruction scheduling.

   In this kernel's context, the lack of intra-loop data reuse and dependencies prevents these semantic optimizations from improving execution efficiency. The instruction change yields no significant performance difference, confirming that `__restrict__` benefits are highly scenario-dependent. Its impact is more pronounced in kernels with higher arithmetic intensity, complex memory patterns, or alias ambiguity—not in bandwidth-bound vector operations.

### 2. `float4` Vectorized Loads

1. Using `float4` vectorized loads explicitly fetches data from device memory, completing coalesced memory access for four float scalars with a single instruction to maximize GPU memory transaction efficiency and access throughput.

2. **Modified kernel:**

   ```c++
   __global__ void vectorAdd_float4(const float* __restrict__ d_a, const float* __restrict__ d_b, float* __restrict__ d_c, size_t N) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < N / 4) {
           const float4* d_a4 = reinterpret_cast<const float4*>(d_a);
           const float4* d_b4 = reinterpret_cast<const float4*>(d_b);
           float4* d_c4 = reinterpret_cast<float4*>(d_c);
           
           float4 va = d_a4[idx];
           float4 vb = d_b4[idx];
           d_c4[idx] = make_float4(
               va.x + vb.x,
               va.y + vb.y,
               va.z + vb.z,
               va.w + vb.w
           );
       }   
   }
   
   template <typename T>
   __global__ void vectorAdd_tail(const T* __restrict__ d_a, const T* __restrict__ d_b, T* __restrict__ d_c, size_t N, size_t start) {
       int idx = start + threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < N) {
           d_c[idx] = d_a[idx] + d_b[idx];
       }   
   }
   ```

3. **Key Performance Metrics from Nsight Compute**

   | Metric                      | Value |
   | :-------------------------- | :---- |
   | Duration [ms]               | 4.00  |
   | Compute (SM) Throughput [%] | 2.43  |
   | Memory Throughput [%]       | 90.46 |
   | DRAM Throughput [%]         | 90.46 |
   | L2 Cache Throughput [%]     | 24.40 |
   | Achieved Occupancy [%]      | 81.03 |

4. **Analysis**

   While `float4` vectorization didn't significantly reduce kernel execution time (4.02 ms → 4.00 ms), its underlying effects are evident in microarchitecture metrics. Compute (SM) Throughput [%] dropped from 9.67% to 2.43%—not due to degraded compute efficiency, but a fundamental shift in bottlenecks. In the original implementation, discrete loads caused significant instruction issue and address calculation overhead during memory wait periods, artificially inflating "compute throughput". After switching to `float4`, the instruction changed from `LDG.E.CONSTANT` to `LDG.E.128.CONSTANT`, increasing data transfer per instruction by 4× and drastically reducing total instruction count. SM cycles dedicated to actual computation naturally decreased, revealing the kernel's pure memory-bound nature.

   In this operator, since the access pattern is already fully coalesced and bandwidth-limited, vectorization didn't bring additional benefits. However, it still holds potential advantages for more complex operators (e.g., multi-operator fusion, non-aligned accesses, or higher arithmetic intensity).

## 4. Host-Device Communication Optimization

**Baseline**: Kernel-level optimizations are complete (`const __restrict__` explicit semantics + `float4` 128-bit vectorized loads). Kernel-side metrics show Compute (SM) Throughput [%] reduced to 2.43%, memory throughput saturated at 90.46%, and kernel execution time stable at 4.00 ms. No further optimization space remains within the kernel; the bottleneck has completely shifted to the Host-Device communication layer. This section focuses on system-level Host-Device communication optimization.

### 1. Pinned Memory (Page-Locked Memory)

1. Using `cudaMallocHost` allocates page-locked memory on the host side. By eliminating runtime page locking/unlocking, the CUDA driver maps host virtual addresses to stable physical page frames, enabling GPU DMA engines to initiate asynchronous PCIe transfers without CPU involvement. Simultaneously, page-locked memory is a necessary condition for `cudaMemcpyAsync` to achieve true asynchronous semantics: only when source/destination memory is pinned can the CUDA runtime guarantee that copy requests are submitted non-blocking and overlap with kernel execution across different streams. This characteristic provides the system-level prerequisite for subsequent pipeline scheduling based on multi-stream.

2. **Using `cudaMallocHost` on host side and `cudaFreeHost` for deallocation:**

   ```c++
   float *a, *b, *c;
   CHECK_CUDA(cudaMallocHost((void**)&a, N * sizeof(float)));
   CHECK_CUDA(cudaMallocHost((void**)&b, N * sizeof(float)));
   CHECK_CUDA(cudaMallocHost((void**)&c, N * sizeof(float)));
   // ...
   CHECK_CUDA(cudaFreeHost(a));
   CHECK_CUDA(cudaFreeHost(b));
   CHECK_CUDA(cudaFreeHost(c));
   ```

3. **Key Performance Metrics**

   | Metric                                           | Value         |
   | :----------------------------------------------- | :------------ |
   | Duration [ms]                                    | 4.00          |
   | Compute (SM) Throughput [%]                      | 2.43          |
   | Memory Throughput [%]                            | 90.28         |
   | DRAM Throughput [%]                              | 90.28         |
   | L2 Cache Throughput [%]                          | 24.34         |
   | Achieved Occupancy [%]                           | 80.82         |
   | Total Execution Time [ms] - naive                | 310.244537 ms |
   | Total Execution Time [ms] - `const __restrict__` | 310.173401 ms |
   | Total Execution Time [ms] - float4               | 310.333466 ms |
   | Total Execution Time [ms] - pinned memory        | 140.640869 ms |

4. **Analysis**

   Pinned memory optimizes the Host-Device data path, with benefits appearing in the transfer phase rather than kernel computation. Therefore, kernel-specific metrics (SM / DRAM / L2) captured by Nsight Compute remain consistent before and after this optimization, as expected.

   Nsight Compute focuses on hardware utilization during kernel execution and doesn't include the PCIe transfer process of `cudaMemcpyAsync`, thus cannot directly reflect pinned memory benefits on the Host-Device data path. To verify concrete benefits, **Nsight Systems** was used to compare H2D transfer times between pageable and pinned memory. For a 512 MB dataset, using pinned memory reduced transfer time from 70.519 ms to 42.717 ms, with effective bandwidth increasing from ~7.1 GB/s to 11.7 GB/s—closer to the achievable PCIe bus bandwidth limit.

   To evaluate overall application impact, CUDA events timed the complete execution flow including Host-Device data loading, kernel computation, and result writeback. Results show kernel-level optimizations (`const __restrict__` and `float4` vectorization) have negligible impact on Total Execution Time, which remains stable at ~310 ms—indicating that application performance is dominated by Host-Device data transfers. In contrast, introducing pinned memory significantly reduces Total Execution Time to 140.64 ms, confirming that pinned memory as a system-level optimization has a decisive impact on end-to-end performance.

   Simultaneously, pinned memory provides the necessary prerequisite for system-level parallel scheduling: only with pinned memory can `cudaMemcpyAsync` achieve true non-blocking DMA transfers, enabling Host-Device data copies to overlap with kernel execution in different streams. This characteristic manifests as further end-to-end execution time reduction in the next section's dual-stream pipeline optimization.

### 2. Dual-Stream Concurrency

1. Building upon the page-locked memory configuration, we introduce a dual-stream scheduling strategy to improve parallelism between Host-Device data transfer and computation. In the single-stream execution model, even with `cudaMemcpyAsync`, data transfers and kernel execution must complete sequentially within the same stream, exposing PCIe transfer latency entirely on the critical path and limiting overall efficiency.

   By partitioning input data into multiple batches and assigning H2D transfers, kernel execution, and D2H transfers of different batches to two independent CUDA streams, the GPU's copy engine and compute engine can work in parallel. This optimization doesn't change individual kernel performance but shortens end-to-end execution time by hiding Host-Device data transfer overhead.

   **Note**: Multi-stream concurrency depends on the true asynchronous DMA capability provided by page-locked memory. Only with pinned memory can `cudaMemcpyAsync` submit non-blocking requests that overlap with kernel execution, making pinned memory a prerequisite for dual-stream pipelining.

2. **Implementation with dual streams + `cudaMemcpyAsync`:**

   ```c++
   template <typename T>
   void launchVectorAdd(const T* h_a, const T* h_b, T* h_c, size_t N, const int numStream) {
       constexpr size_t chunkSize = 1 << 22; 
       int numChunk = (N + chunkSize - 1) / chunkSize;
   
       cudaStream_t stream[numStream];
       T *dev_a[numStream], *dev_b[numStream], *dev_c[numStream]; 
       for (int i = 0; i < numStream; ++i) {
           CHECK_CUDA(cudaMalloc((void**)&dev_a[i], chunkSize * sizeof(T)));
           CHECK_CUDA(cudaMalloc((void**)&dev_b[i], chunkSize * sizeof(T)));
           CHECK_CUDA(cudaMalloc((void**)&dev_c[i], chunkSize * sizeof(T)));
           CHECK_CUDA(cudaStreamCreate(&stream[i]));
       }
       
       for (int i = 0; i < numChunk; ++i) {
           int sid = i % numStream;      
           size_t offset = i * chunkSize;
           size_t currentChunkSize = min(chunkSize, N - offset);
   
           CHECK_CUDA(cudaMemcpyAsync(dev_a[sid], h_a + offset, currentChunkSize * sizeof(T), cudaMemcpyHostToDevice, stream[sid]));
           CHECK_CUDA(cudaMemcpyAsync(dev_b[sid], h_b + offset, currentChunkSize * sizeof(T), cudaMemcpyHostToDevice, stream[sid]));
   
           const int threadsPerBlock = 256;
           size_t N4 = (currentChunkSize / 4) * 4;
           int blocksPerGrid = (N4 / 4 + threadsPerBlock - 1) / threadsPerBlock;
           vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream[sid]>>>(dev_a[sid], dev_b[sid], dev_c[sid], currentChunkSize);
           if (currentChunkSize != N4) {
               vectorAdd_tail<<<1, 32, 0, stream[sid]>>>(dev_a[sid], dev_b[sid], dev_c[sid], currentChunkSize, N4);
           }
   
           CHECK_CUDA(cudaMemcpyAsync(h_c + offset, dev_c[sid], currentChunkSize * sizeof(T), cudaMemcpyDeviceToHost, stream[sid]));       
       } 
   
       for (int i = 0; i < numStream; ++i) {
           CHECK_CUDA(cudaStreamSynchronize(stream[i])); 
       }
       for (int i = 0; i < numStream; ++i) {
           CHECK_CUDA(cudaFree(dev_a[i]));
           CHECK_CUDA(cudaFree(dev_b[i]));
           CHECK_CUDA(cudaFree(dev_c[i]));
           CHECK_CUDA(cudaStreamDestroy(stream[i])); 
       }  
   }
   ```

3. **Key Performance Metrics**

   | Metric                                           | Value         |
   | :----------------------------------------------- | :------------ |
   | Duration [μs]                                    | 83.14         |
   | Compute (SM) Throughput [%]                      | 3.68          |
   | Memory Throughput [%]                            | 94.43         |
   | DRAM Throughput [%]                              | 94.43         |
   | L2 Cache Throughput [%]                          | 36.98         |
   | Achieved Occupancy [%]                           | 89.46         |
   | Total Execution Time [ms] - naive                | 310.244537 ms |
   | Total Execution Time [ms] - `const __restrict__` | 310.173401 ms |
   | Total Execution Time [ms] - float4               | 310.333466 ms |
   | Total Execution Time [ms] - pinned memory        | 140.640869 ms |
   | Total Execution Time [ms] - dual stream          | 119.945602 ms |

4. **Analysis**

   Building upon pinned memory, the dual-stream mechanism further pipelines Host-Device data transfers with kernel computation. Specifically, the original data is partitioned into multiple chunks, with H2D copy, kernel execution, and D2H writeback alternating across two different CUDA streams, thereby achieving overlap between data transfer and computation phases.

   From Nsight Compute's kernel-level metrics, dual-stream doesn't change kernel computation logic. However, since each kernel only processes a portion of the original data, its single execution time (Duration) significantly shortens to ~83 μs. Note that this metric reflects only a single kernel instance and cannot be directly compared with the 4 ms single-kernel case. After accumulating execution times of all sub-kernels, total kernel execution time is ~2.66 ms, still lower than the original single-kernel execution time.

   This indicates that chunked execution not only changes kernel scheduling but also improves actual execution conditions. Due to significantly reduced working set size per kernel, memory access shows higher L2 cache throughput utilization and DRAM throughput utilization. Correspondingly, SM throughput increases from 2.43% to 3.68%, and Achieved Occupancy rises to 89.46%. This demonstrates that under chunked conditions, the GPU can more effectively hide memory access latency, thereby reducing total kernel execution time.

   From an application perspective, dual-stream's main benefit lies in further end-to-end execution time reduction. While kernel-level optimizations have negligible impact on Total Execution Time, introducing pinned memory already reduced it from ~310 ms to 140.64 ms. Building upon this, the dual-stream mechanism further reduces Total Execution Time to 119.95 ms. This shows that dual-stream effectively reduces idle time in sequential execution phases by overlapping Host-Device data transfer with kernel computation, improving overall execution efficiency.

   In summary, dual-stream doesn't improve performance by accelerating individual kernels but by changing system-level execution patterns through chunked scheduling and compute-transfer overlap, enabling more complete utilization of GPU resources and PCIe data paths.

## 5. Summary and Outlook

This report examines the vector addition operator on the CUDA platform, systematically demonstrating a multi-level optimization process from kernel-level to system-level around its performance bottlenecks in practical applications. By progressively introducing compiler-friendly memory semantics (`const __restrict__`), vectorized data loads (`float4`), page-locked memory (pinned memory), and multi-stream-based transfer-compute overlap scheduling, this report constructs a complete performance evolution path with quantitative analysis of each stage using Nsight Compute and Nsight Systems.

Experimental results show that for this vector addition operator, performance bottlenecks are not limited to kernel computation or memory access efficiency but progressively shift to Host-Device data transfer and system-level scheduling as optimizations advance. Kernel-level optimizations mainly confirm that the operator has approached its memory access ceiling, while page-locked memory and dual-stream mechanisms significantly reduce Host-Device data transfer overhead, delivering major performance gains at the application level. This process demonstrates that effective CUDA performance optimization requires identifying bottlenecks at the correct level and selecting matching optimization techniques.

Further performance improvements remain possible. For example, designing specialized kernels for different data types, scales, and hardware platforms, or introducing parameterized kernels with auto-tuning mechanisms to select optimal configurations at runtime. However, such optimizations typically require significantly increased code complexity and gradually evolve into library-level engineering problems (e.g., general operator libraries or automatic code generation frameworks), whose design goals exceed this report's objective to "demonstrate typical operator optimization methodology." Therefore, related directions were not discussed in depth.

Overall, this work aims to systematically demonstrate common and universally meaningful technical paths in CUDA performance optimization through a simple yet representative operator, rather than building an optimal implementation for all scenarios. It is hoped that this report's analysis process and methodology can serve as a reference for performance tuning of practical CUDA programs.