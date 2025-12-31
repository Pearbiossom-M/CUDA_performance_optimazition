<div align="center">

# Vector Addition Optimization Report

中文 | [English](README.md) 

</div>

---

## 1. Background & Problem Definition

1. 算子描述
   - 向量加法：C[i] = A[i] + B[i]
2. 使用场景
   * 这是深度学习中最基础但最常见的 memory-bound 算子
3. 硬件 & 软件环境
   * GPU型号：RTX 5060 Ti 
   * CUDA version：13.1
   * 编译参数：默认`-O3`

## 2. Naive 实现 & Baseline 性能

1. naive kernel代码如下所示，完整代码见于 naive.cu

   ```c++
   template <typename T>
   __global__ void vectorAdd_naive(const T* d_a, const T* d_b, T* d_c, size_t N) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < N) {
           d_c[idx] = d_a[idx] + d_b[idx];
       }   
   }
   ```

2. 理论分析

   假设向量长度为 N，每个元素是 4 字节的 `float` 类型。

   * 计算浮点运算次数（FLOPs）

     * 每个元素的计算需要一次加法：

       总运算量 = **N FLOPs**

   * 计算内存访问量（Bytes）

     每个元素需要：

     - 读取A[i]：4 Bytes

     - 读取B[i]：4 Bytes

     - 写入C[i]：4 Bytes

     总内存访问量 = **12N Bytes**

   * 计算算术强度（FLOPS / Byte）

     算术强度 = N FLOPs / 12N bytes = 1/12 ≈ **0.083 FLOPs/byte**

     0.083这个值非常低，说明：

     * 内存受限（Memory-bound）：计算量远小于数据搬运量
     * GPU 实际利用率：每字节仅产生 0.083 次运算，导致算力严重浪费

3. 使用Nsight Compute获取关键性能指标

   | metric                      | value |
   | :-------------------------- | ----- |
   | Duration [ms]               | 4.02  |
   | Compute (SM) Throughput [%] | 9.66  |
   | Memory Throughput [%]       | 89.88 |
   | DRAM Throughput [%]         | 89.88 |
   | L2 Cache Throughput [%]     | 24.29 |
   | Achieved Occupancy [%]      | 80.26 |

4. 解读

   naive版本的带宽利用率接近90%，主要归因于100%合并内存访问——每个warp的32个线程连续访问128字节数据，硬件自动合并为单条内存事务（memory transaction）。计算的极简性（单条FMA指令）使kernel迅速触及内存墙，成为典型的带宽饱和型负载。基于Roofline模型，在memory bound已显现的情况下，后续优化将聚焦于指令效率与系统级并行度的精准提升。

## 3. Kernel 级优化

### 1. 编译器友好的内存语义

1. 使用 `const __restrict__` 显式表达只读与无别名语义，为潜在的 load 重排、指令合并以及缓存路径选择提供优化空间。尽管在现代NVCC + 优化级别`-O3`下，编译器已能自动推断等价信息然后自动优化，但在复杂 kernel 中 `const __restrict__` 仍然是最强、最可靠的语义声明。

2. 修改后的kernel如下：

   ```c++
   template <typename T>
   __global__ void vectorAdd_restrict(const T* __restrict__ d_a, const T* __restrict__ d_b, T* __restrict__ d_c, size_t N) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < N) {
           d_c[idx] = d_a[idx] + d_b[idx];
       }   
   }
   ```

3. 使用Nsight Compute获取关键性能指标

   | metric                      | value |
   | :-------------------------- | ----- |
   | Duration [ms]               | 4.02  |
   | Compute (SM) Throughput [%] | 9.67  |
   | Memory Throughput [%]       | 89.84 |
   | DRAM Throughput [%]         | 89.84 |
   | L2 Cache Throughput [%]     | 24.29 |
   | Achieved Occupancy [%]      | 80.22 |

4. 解读

   实验结果表明，使用 `const __restrict__` 修饰指针的矢量加法 kernel，其性能与未使用显式别名限定符的 naive 实现几乎相同。这一现象在现代 CUDA 架构下是符合预期的。

   * 从性能特征上看，该 kernel 具有典型的内存带宽受限（memory-bandwidth-bound）行为：每个线程仅执行两次全局内存加载、一次浮点加法和一次全局存储操作，计算强度极低。同时，内核中不存在循环结构、数据复用、跨指令依赖或潜在的 store–load 冲突。在此情况下，即使编译器在未使用 `__restrict__` 时采用较为保守的指针别名假设，也不会引入额外的同步、序列化或冗余访存操作。因此，通过 `__restrict__` 强化指针的无别名语义，并未在该算子中转化为可观的性能提升。
   * SASS 指令层面，引入 `const __restrict__` 后，生成的加载指令由 `LDG.E` 变为 `LDG.E.CONSTANT`，表明编译器能够利用更强的只读与无别名语义。在 Volta 及之后的 CUDA 架构中，全局内存访问共享统一的 L1 数据缓存结构，不同形式的加载指令主要在一致性模型与依赖约束上存在差异，并不意味着使用物理上独立的缓存或必然降低访存延迟。`LDG.E.CONSTANT` 所提供的优势在于放宽内存可见性与依赖管理约束，从而允许编译器和硬件进行更激进的指令调度。

   * 在本实验所采用的矢量算子中，由于缺乏循环内数据复用、指令级依赖以及潜在的内存冲突，上述语义层面的优化并未转化为实际的执行效率提升。因此，指令形式的变化并未在性能指标上体现出显著差异。该结果表明，`__restrict__` 的性能收益具有明显的场景依赖性，其效果更可能在算术强度更高、内存访问模式更复杂、或存在显著别名歧义的 kernel 中体现，而非在此类极简且完全带宽受限的矢量操作中。

### 2. `float4` 向量化加载

1. 使用 `float4` 向量化加载显存数据，通过单次指令完成 4 个 float 型标量的合并式内存访问，最大化 GPU 的内存事务带宽利用率与内存访问吞吐量。

2. 修改后的kernel如下：

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
   
3. 使用Nsight Compute获取关键性能指标

   | metric                      | value |
   | :-------------------------- | ----- |
   | Duration [ms]               | 4.00  |
   | Compute (SM) Throughput [%] | 2.43  |
   | Memory Throughput [%]       | 90.46 |
   | DRAM Throughput [%]         | 90.46 |
   | L2 Cache Throughput [%]     | 24.40 |
   | Achieved Occupancy [%]      | 81.03 |

4. 解读

   `float4`向量化加载虽未在`kernel`层面执行时间取得显著收益（4.02 ms → 4.00 ms），但其底层优化效果已在微架构指标中清晰显现。Compute (SM) Throughput [%] 从 9.67% 骤降至 2.43%，此降幅并非计算效率恶化，而是根本性的性能瓶颈转移信号——原始实现中因离散加载导致大量指令发射与地址计算开销，使得 SM 在内存等待期间被无效指令占据，表现为虚高的“计算吞吐量”；改用 `float4`向量化加载后，指令`LDG.E.CONSTANT` 转变为`LDG.E.128.CONSTANT` ，单指令搬运数据量提升 4 倍，指令总数与发射压力锐减，SM 真正用于实际运算的周期占比自然回落，暴露出该 kernel 纯内存受限（Memory-Bound）的本质。
   
   在本算子中，由于访问模式已完全合并且受限于带宽上限，向量化未带来额外收益，但在更复杂算子（如多算子融合、非对齐访问或更高算术强度）中仍具有潜在优势。

## 4. Host–Device Interaction Optimization

前置调优基线：已完成 Kernel 级优化（`const __restrict__` 显式语义声明 + `float4` 128bit 向量化加载）。Kernel 内部的 Compute (SM) Throughput [%] 降至 2.43%、内存吞吐饱和至 90.46%，Kernel 执行耗时稳定在 4.00ms，Kernel 侧已无性能挖掘空间，性能瓶颈完全转移至Host-Device 交互层，因此开展本章节的系统级 Host-Device 交互优化。

### 1. `pinned memory`

1. 使用 `cudaMallocHost` 分配主机端页锁定内存，通过避免运行期的页面锁定与解除操作，使 CUDA 驱动能够直接将主机虚拟地址映射为稳定的物理页集合，从而支持 GPU DMA 引擎发起不经 CPU 参与的异步 PCIe 数据传输。同时，页锁定内存是 `cudaMemcpyAsync` 实现真正异步语义的必要条件：仅当源/目的内存为 pinned memory 时，CUDA runtime 才能保证拷贝请求以非阻塞方式提交，并与 kernel 执行在不同 stream 中实现计算–传输重叠。这一特性为后续基于多 stream 的 pipeline 调度提供了系统级前提。

2. 在Host侧使用`cudaMallocHost`分配内存，使用`cudaFreeHost`释放内存，代码如下：

   ```c++
   float *a, *b, *c;
   CHECK_CUDA(cudaMallocHost((void**)&a, N * sizeof(float)));
   CHECK_CUDA(cudaMallocHost((void**)&b, N * sizeof(float)));
   CHECK_CUDA(cudaMallocHost((void**)&c, N * sizeof(float)));
   ...
   CHECK_CUDA(cudaFreeHost(a));
   CHECK_CUDA(cudaFreeHost(b));
   CHECK_CUDA(cudaFreeHost(c));
   ```

3. 采集关键性能指标

   | metric                                          | value         |
   | :---------------------------------------------- | ------------- |
   | Duration [ms]                                   | 4.00          |
   | Compute (SM) Throughput [%]                     | 2.43          |
   | Memory Throughput [%]                           | 90.28         |
   | DRAM Throughput [%]                             | 90.28         |
   | L2 Cache Throughput [%]                         | 24.34         |
   | Achieved Occupancy [%]                          | 80.82         |
   | Total Execution Time [ms]——naive                | 310.244537 ms |
   | Total Execution Time [ms]——`const __restrict__` | 310.173401 ms |
   | Total Execution Time [ms]——float4               | 310.333466 ms |
   | Total Execution Time [ms]——pinned memory        | 140.640869 ms |
   
4. 解读

   页锁定内存属于 Host–Device 数据通路优化，其收益体现在数据传输阶段而非 kernel 计算阶段。因此，Nsight Compute 所采集的 SM / DRAM / L2 等 kernel 内部指标在该优化前后保持一致，符合预期。
   
   Nsight Compute 主要关注 kernel 执行期间的硬件利用率，不包含 `cudaMemcpyAsync` 的 PCIe 传输过程，因此无法直接反映 pinned memory 在 Host–Device 数据传输路径上的性能收益。为验证 pinned memory 的具体收益，使用 **Nsight System** 对比了 pageable 和 pinned memory的 H2D 传输耗时，当数据规模为 512MB 时，使用 pinned memory 后，传输时间由 70.519ms 降为 42.717ms，对应的有效传输带宽由约 7.1 GB/s 提升至 11.7 GB/s，更接近 PCIe 总线的可达带宽上限，这表明 pinned memory 能够有效提升 Host–Device 传输性能。
   
   进一步地，为评估 pinned memory 对应用整体性能的影响，本文使用 CUDA event 对 Host–Device 数据加载、kernel 计算以及结果回写的完整执行流程进行端到端计时。实验结果显示，kernel 级优化（如 `const __restrict__` 及 float4 向量化加载）对 Total Execution Time 的影响可以忽略，整体执行时间稳定在约 310 ms，说明在该工作负载下应用性能主要受 Host–Device 数据传输所主导。相比之下，引入 pinned memory 后，Total Execution Time 显著下降至 140.64 ms，验证了 pinned memory 作为系统级优化手段对端到端性能具有决定性影响。
   
   同时， pinned memory 还为后续的系统级并行调度提供了必要前提：只有在 pinned memory 条件下，`cudaMemcpyAsync` 才能实现真正的非阻塞 DMA 传输，从而使 Host–Device 数据拷贝与 kernel 执行在不同 stream 中重叠。这一特性将在下一节双 Stream pipeline 优化中体现为端到端执行时间的进一步缩短。
   

### 2. `double streams`

1. 在完成页锁定内存配置后，进一步引入双 stream（double streams）调度策略，以提升 Host–Device 数据传输与计算的并行度。在单 stream 执行模型下，即便使用 `cudaMemcpyAsync` 接口，数据传输与 kernel 执行仍需在同一 stream 中按顺序完成，导致 PCIe 传输延迟完全暴露在关键路径上，限制了整体执行效率。

   通过将输入数据划分为多个批次，并将不同批次的 H2D 传输、kernel 执行及 D2H 传输分配至两个独立的 CUDA stream 中，GPU 的 copy engine 与 compute engine 得以并行工作，从而在系统层面实现计算与数据传输的重叠。该优化并不改变单个 kernel 的计算性能，而是通过隐藏 Host–Device 数据传输开销，缩短程序的端到端执行时间。

   需要指出的是，多 stream 并发调度依赖于页锁定内存所提供的真正异步 DMA 传输能力。仅当 Host 内存为 pinned memory 时，`cudaMemcpyAsync` 才能以非阻塞方式提交并与 kernel 执行重叠，因此页锁定内存是实现 double streams pipeline 的必要前提。

2. 使用double streams + `cudaMemcpyAsync`，代码如下：

   ```c++
   template <typename T>
   void launchVectorAdd(const T* h_a, const T* h_b, T* h_c, size_t N, const int numStream) {
       constexpr size_t chunkSize = 1 << 22; 
       int numChunk = (N + chunkSize -1) / chunkSize;
   
       cudaStream_t stream[numStream];
       T *dev_a[numStream], *dev_b[numStream], *dev_c[numStream]; 
       for (int i=0; i<numStream; ++i) {
           CHECK_CUDA(cudaMalloc((void**)&dev_a[i], chunkSize * sizeof(T)));
           CHECK_CUDA(cudaMalloc((void**)&dev_b[i], chunkSize * sizeof(T)));
           CHECK_CUDA(cudaMalloc((void**)&dev_c[i], chunkSize * sizeof(T)));
           CHECK_CUDA(cudaStreamCreate(&stream[i]));
       }
       
       for (int i=0; i<numChunk; ++i) {
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
   
       for (int i=0; i<numStream; ++i) {
           CHECK_CUDA(cudaStreamSynchronize(stream[i])); 
       }
       for (int i=0; i<numStream; ++i) {
           CHECK_CUDA(cudaFree(dev_a[i]));
           CHECK_CUDA(cudaFree(dev_b[i]));
           CHECK_CUDA(cudaFree(dev_c[i]));
           CHECK_CUDA(cudaStreamDestroy(stream[i])); 
       }  
   }
   ```

3. 采集关键性能指标

   | metric                                          | value         |
   | :---------------------------------------------- | ------------- |
   | Duration [us]                                   | 83.14         |
   | Compute (SM) Throughput [%]                     | 3.68          |
   | Memory Throughput [%]                           | 94.43         |
   | DRAM Throughput [%]                             | 94.43         |
   | L2 Cache Throughput [%]                         | 36.98         |
   | Achieved Occupancy [%]                          | 89.46         |
   | Total Execution Time [ms]——naive                | 310.244537 ms |
   | Total Execution Time [ms]——`const __restrict__` | 310.173401 ms |
   | Total Execution Time [ms]——float4               | 310.333466 ms |
   | Total Execution Time [ms]——pinned memory        | 140.640869 ms |
   | Total Execution Time [ms]——double stream        | 119.945602 ms |

4. 解读

   在引入 pinned memory 的基础上，进一步采用双 stream 机制对 Host–Device 数据传输与 kernel 计算进行流水线化调度。具体而言，将原始数据划分为多个子块，并在不同 CUDA stream 中交替执行 H2D 拷贝、kernel 计算及 D2H 回写，从而实现数据传输与计算阶段的重叠。
   
   从 Nsight Compute 的 kernel 级指标来看，double stream 并未改变 kernel 的计算逻辑，但由于每个 kernel 仅处理原始数据的一部分，其单次执行时间（Duration）显著缩短至约 83 μs。需要指出的是，该指标仅反映单个 kernel 实例的执行时间，不能直接与 single kernel 情况下的 4 ms 进行对比。将所有子 kernel 的执行时间累加后，总 kernel 执行时间约为 2.66 ms，仍低于原始单 kernel 的执行时间。
   
   这一现象表明，分块执行不仅改变了 kernel 的调度方式，也改善了 kernel 的实际执行条件。由于单个 kernel 的工作集规模显著减小，其内存访问表现出更高的 L2 Cache 吞吐利用率和更高的 DRAM 吞吐利用率，对应地，SM 吞吐率由 2.43% 提升至 3.68%，Achieved Occupancy 提升至 89.46%。这说明在分块条件下，GPU 能够更有效地隐藏内存访问延迟，从而缩短总 kernel 执行时间。
   
   从应用级角度来看，double stream 的主要收益体现在端到端执行时间的进一步缩短。尽管 kernel 级优化（如 `const __restrict__` 及 float4 向量化加载）对 Total Execution Time 的影响可以忽略，引入 pinned memory 后端到端执行时间已由约 310 ms 降低至 140.64 ms，而在此基础上采用双 stream 机制后，Total Execution Time 进一步下降至 119.95 ms。这表明 double stream 通过重叠 Host–Device 数据传输与 kernel 计算，有效减少了串行执行阶段的空闲时间，提升了整体执行效率。
   
   综上，double stream 并非通过加速单个 kernel 来提升性能，而是通过分块调度与计算–传输重叠改变了系统级执行模式，使 GPU 资源与 PCIe 数据通路得到更充分的利用。

## 5. 总结与展望

本文以 CUDA 平台上的矢量加法算子为研究对象，围绕其在实际应用中的性能瓶颈，系统性地展示了从 kernel 级到系统级的多层次优化过程。通过逐步引入编译器友好的内存语义（`const __restrict__`）、向量化数据加载（float4）、页锁定内存（pinned memory）以及基于多 stream 的传输–计算重叠调度，本文构建了一条完整的性能演化路径，并在每个阶段结合 Nsight Compute 与 Nsight Systems 对性能变化进行了定量分析。

实验结果表明，对于该矢量加法算子，其性能瓶颈并不局限于 kernel 内部计算或访存效率，而是随着优化的推进逐步转移至 Host–Device 数据传输与系统级调度层面。Kernel 级优化主要用于确认算子已接近其内存访问上限，而页锁定内存与双 stream 机制则显著降低了 Host–Device 数据传输开销，从而在应用层面取得了主要性能收益。这一过程表明，有效的 CUDA 性能优化需要在正确的层级识别瓶颈，并选择与之匹配的优化手段。

进一步的性能提升仍然是可能的。例如，可以针对不同数据类型、数据规模及硬件平台设计专用 kernel，或引入参数化 kernel 与自动调优机制，在运行期选择最优配置。然而，此类优化通常需要显著增加代码复杂度，并逐步演化为库级工程问题（如通用算子库或自动代码生成框架），其设计目标已超出本文“展示典型算子优化方法论”的范围。因此，本文未对相关方向展开深入讨论。

总体而言，本文的工作旨在通过一个简单而具有代表性的算子，系统性地展示 CUDA 性能优化中常见且具有普适意义的技术路径，而非构建面向所有场景的最优实现。希望本文的分析过程与方法论能够为实际 CUDA 程序的性能调优提供参考。



