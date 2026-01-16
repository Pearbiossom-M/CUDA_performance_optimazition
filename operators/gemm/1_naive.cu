/* -----------------------------------------------------------------
 * @file   naive.cu
 * @brief  naive GEMM
 * ---------------------------------------------------------------
 */
#include<cuda_runtime.h>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cuda_bf16.h>
#include<cublas_v2.h>
#include<cuda.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA Error] %s:%d in %s: %s (code: %d)\n", \
                     __FILE__, __LINE__, __func__, cudaGetErrorString(err), err); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

constexpr int WARMUP = 10;
constexpr int ITERATION = 50;

double tflops(int M, int N, int K, double ms) {
    double flops = 2.0 * M * N * K;
    return flops / (ms * 1e9);
}

__global__ void gemm_naive(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    float* C,
    const int M, const int N, const int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f, a, b;
        for (int kk = 0; kk < K; ++kk) {
            a = __bfloat162float(A[row * K + kk]);
            b = __bfloat162float(B[kk * N + col]);
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

void launchGEMM(
    const __nv_bfloat16* h_A, 
    const __nv_bfloat16* h_B, 
    float* h_C, float* h_C_ref,
    const int M, const int N, const int K 
) {
    // memcpy: host to device
    size_t sizeA = M * K * sizeof(__nv_bfloat16);
    size_t sizeB = K * N * sizeof(__nv_bfloat16);
    size_t sizeC = M * N * sizeof(float);

    __nv_bfloat16 *d_A, *d_B;
    float *d_C, *d_C_ref;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));
    CHECK_CUDA(cudaMalloc((void**)&d_C_ref, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // warm-up
    for (int i=0; i<WARMUP; ++i) {
        gemm_naive<<<blocksPerGrid, threadsPerBlock>>>(
            d_A, d_B, d_C, M, N, K
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // time
    CHECK_CUDA(cudaEventRecord(start));
    for (int i=0; i<ITERATION; ++i) {
        gemm_naive<<<blocksPerGrid, threadsPerBlock>>>(
            d_A, d_B, d_C, M, N, K
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
    float naive_ms = elapsedMs / ITERATION;

    printf("Naive GEMM:\n");
    printf("  Time: %.3f ms\n", naive_ms);
    printf("  TFLOPS: %.2f\n", tflops(M, N, K, naive_ms));

    // compute reference
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f;
    float beta  = 0.0f;
    CHECK_CUBLAS(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            M, N, K,
            &alpha,
            d_B, CUDA_R_16BF, N,
            d_A, CUDA_R_16BF, K,
            &beta,
            d_C_ref, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // memcpy: device to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, sizeC, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cublasDestroy(handle);
}

int main() {
    int M = 4096;
    int N = 4096;
    int K = 4096;

    printf("GEMM: M=%d, N=%d, K=%d\n", M, N, K);

    __nv_bfloat16* A = new __nv_bfloat16[M * K];
    __nv_bfloat16* B = new __nv_bfloat16[K * N];
    float* C = new float[M * N];
    float* C_ref = new float[M * N];

    // assignment
    //for (int i = 0; i < M * K; ++i) A[i] = __float2bfloat16(float(i % 3));
    //for (int i = 0; i < K * N; ++i) B[i] = __float2bfloat16(float(i % 3));
    for (int i = 0; i < M * K; ++i) A[i] = __int2bfloat16_rn(i % 3);
    for (int i = 0; i < K * N; ++i) B[i] = __int2bfloat16_rn(i % 3);

    // launch gemm_naive
    launchGEMM(A, B, C, C_ref, M, N, K);

    // correctness check
    bool pass = true;
    const float tol = 1e-4f;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C[i] - C_ref[i]) > tol) {
            int row = i / N;
            int col = i % N;
            printf("Error at [%d, %d]: %f vs %f\n", row, col, C[i], C_ref[i]);
            pass = false;
            break;
        }
    }

    printf(pass ? "test pass!\n" : "test failed!\n");

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;
    return 0;
}

// nvcc -arch=native ./naive.cu -lcublas -o ./naive && ./naive