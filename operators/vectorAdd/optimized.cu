/* -----------------------------------------------------------------
 * @file   optimized.cu
 * @brief  optimized vector addition
 * ---------------------------------------------------------------
 */
#include<cuda_runtime.h>
#include<cstdio>
#include<cstdlib>
#include<cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA Error] %s:%d in %s: %s (code: %d)\n", \
                     __FILE__, __LINE__, __func__, cudaGetErrorString(err), err); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void vectorAdd(const float* __restrict__ d_a, const float* __restrict__ d_b, float* __restrict__ d_c, size_t N) {
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
template __global__ void vectorAdd_tail<float> (const float* __restrict__, const float* __restrict__, float* __restrict__, size_t, size_t);

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
template void launchVectorAdd<float> (const float*, const float*, float*, size_t, const int);

int main() {
    size_t N = (1 << 27) + 11;
    printf("N = %zu\n", N);
    const int numStream = 2;

    float *a, *b, *c;
    CHECK_CUDA(cudaMallocHost((void**)&a, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&b, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&c, N * sizeof(float)));
    for (size_t i=0; i<N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start)); 

    launchVectorAdd(a, b, c, N, numStream);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
    printf("Total execution time (including H2D, kernel, D2H) = %f ms\n", elapsedMs);

    bool pass = true;
    const float tolerance = 1e-6f;
    for (size_t i=0; i<N; ++i) {
        float true_value = float(i) + 2.0f * float(i);
        if (std::abs(c[i] - true_value) > tolerance) {
            printf("Error in c[%zu] = %f, while true value = %f\n", i, c[i], true_value);
            pass = false;
        }
    }

    if (pass) {
        printf("test pass!\n");
    } else {
        printf("test failed!\n");
    }

    CHECK_CUDA(cudaFreeHost(a));
    CHECK_CUDA(cudaFreeHost(b));
    CHECK_CUDA(cudaFreeHost(c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}