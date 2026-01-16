/* -----------------------------------------------------------------
 * @file   shared_regBlock.cu
 * @brief  shared_regBlock GEMM
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

double tflops(int M, int N, int K, double ms) {
    double flops = 2.0 * M * N * K;
    return flops / (ms * 1e9);
}

// Global variable
constexpr int WARMUP = 10;
constexpr int ITERATION = 50;
// block shape
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;
// the number of elements that each thread needs to move
constexpr int movePerThreadAs = BM * BK / (BM/4 * BN/4);
constexpr int movePerThreadBs = BK * BN / (BM/4 * BN/4);

__global__ void gemm(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    float* C,
    const int M, const int N, const int K
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ __nv_bfloat16 As[BK][BM];
    __shared__ __nv_bfloat16 Bs[BK][BN];

    float sum[4][4] = {0.0f};
    const int rowAs = tid / (BM / movePerThreadAs);
    const int colAs = tid % (BM / movePerThreadAs);
    const int rowBs = tid / (BN / movePerThreadBs);
    const int colBs = tid % (BN / movePerThreadBs);
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load As and Bs
        // Transpose A for the low bank conflict in the compute stage
        for (int i=0; i<movePerThreadAs; ++i) {
            if (blockIdx.y * blockDim.y + colAs * movePerThreadAs + i < M && k0 + rowAs < K) {
                As[rowAs][colAs*movePerThreadAs + i] = \
                A[(blockIdx.y * blockDim.y + colAs * movePerThreadAs + i)*K + k0+rowAs];
                // A[blockIdx.y * blockDim.y + colAs * movePerThreadAs + i][k0+rowAs];
            } else {
                As[rowAs][colAs*movePerThreadAs + i] = __float2bfloat16(0.0f);
            }            
        }  

        for (int i=0; i<movePerThreadBs; ++i) {
            if (k0 + rowBs < K && blockIdx.x*blockDim.x+colBs*movePerThreadBs + i < N) {
                Bs[rowBs][colBs*movePerThreadBs + i] = \
                B[(k0 + rowBs)*N + blockIdx.x*blockDim.x+colBs*movePerThreadBs + i];
                // B[k0 + rowBs][blockIdx.x*blockDim.x+colBs*movePerThreadBs + i];
            } else {
                Bs[rowBs][colBs*movePerThreadBs + i] = __float2bfloat16(0.0f);
            }
        } 
        
        __syncthreads();

        // iterate
        const int rowBase = threadIdx.y * 4;
        const int colBase = threadIdx.x * 4;
        for (int kk = 0; kk < BK; ++kk) {
            float a[4] = {
                __bfloat162float(As[kk][rowBase]),
                __bfloat162float(As[kk][rowBase + 1]),
                __bfloat162float(As[kk][rowBase + 2]),
                __bfloat162float(As[kk][rowBase + 3])
            };
            float b[4] = {
                __bfloat162float(Bs[kk][colBase]),
                __bfloat162float(Bs[kk][colBase + 1]),
                __bfloat162float(Bs[kk][colBase + 2]),
                __bfloat162float(Bs[kk][colBase + 3])
            };
            
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    sum[i][j] += a[i] * b[j];
                }
            } 
        }
        __syncthreads();           
    }
    // write to C
    int row = blockIdx.y * BM + threadIdx.y * 4;
    int col = blockIdx.x * BN + threadIdx.x * 4;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + col + j] = sum[i][j];
            }           
        }
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

    dim3 threadsPerBlock(BN / 4, BM / 4);
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // warm-up
    for (int i=0; i<WARMUP; ++i) {
        gemm<<<blocksPerGrid, threadsPerBlock>>>(
            d_A, d_B, d_C, M, N, K
        );
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // time
    CHECK_CUDA(cudaEventRecord(start));
    for (int i=0; i<ITERATION; ++i) {
        gemm<<<blocksPerGrid, threadsPerBlock>>>(
            d_A, d_B, d_C, M, N, K
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
    float naive_ms = elapsedMs / ITERATION;

    printf("Shared & register GEMM:\n");
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

// nvcc -arch=native ./shared_regBlock.cu -lcublas -o ./shared_regBlock && ./shared_regBlock