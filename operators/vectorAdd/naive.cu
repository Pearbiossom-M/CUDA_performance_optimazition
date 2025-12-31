/* -----------------------------------------------------------------
 * @file   naive.cu
 * @brief  naive vector addition
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

template <typename T>
__global__ void vectorAdd_naive(const T* d_a, const T* d_b, T* d_c, size_t N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_c[idx] = d_a[idx] + d_b[idx];
    }   
}
template __global__ void vectorAdd_naive<float> (const float*, const float*, float*, size_t);

template <typename T>
void launchVectorAdd(const T* h_a, const T* h_b, T* h_c, size_t N) {
    T *dev_a, *dev_b, *dev_c; 
    CHECK_CUDA(cudaMalloc((void**)&dev_a, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&dev_b, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&dev_c, N * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(dev_a, h_a, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, h_b, N * sizeof(T), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd_naive<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
    CHECK_CUDA(cudaGetLastError()); // 检查 launch 错误
    CHECK_CUDA(cudaDeviceSynchronize()); // 检查 runtime 错误

    CHECK_CUDA(cudaMemcpy(h_c, dev_c, N * sizeof(T), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dev_a));
    CHECK_CUDA(cudaFree(dev_b));
    CHECK_CUDA(cudaFree(dev_c));
}
template void launchVectorAdd<float> (const float*, const float*, float*, size_t);

int main() {
    size_t N = (1 << 27) + 11;
    printf("N = %zu\n", N);

    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];
    for (size_t i=0; i<N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start, 0));  // 0 表示 default stream，可选

    launchVectorAdd(a, b, c, N);

    CHECK_CUDA(cudaEventRecord(stop, 0));
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

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}