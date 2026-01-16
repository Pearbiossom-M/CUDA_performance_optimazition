/* -----------------------------------------------------------------
 * @file   tensor_core_WMMA.cu
 * @brief  tensor_core_WMMA GEMM
 * ---------------------------------------------------------------
 */
#include<cuda_runtime.h>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cuda_bf16.h>
#include<cublas_v2.h>
#include<cuda.h>
#include<mma.h>
using namespace nvcuda;

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
// Block tile
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
// Warp tile
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

__global__ void gemm(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    float* C,
    const int M, const int N, const int K
) {
    // block-level tile
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // warp identification inside block
    int warpId = threadIdx.x / 32;

    // 4*4 warps per block
    int warpRow = warpId / (BN/WN);
    int warpCol = warpId % (BN/WN);

    // shared memory
    __shared__ __nv_bfloat16 As[BM][BK];
    __shared__ __nv_bfloat16 Bs[BK][BN];

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // iterate
    for (int kk = 0; kk < K; kk += BK) {
        // load A tile
        for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
            int i = idx / BK;
            int j = idx % BK;
            As[i][j] = A[(blockRow + i) * K + kk + j];
        }

        // load B tile
        for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
            int i = idx / BN;
            int j = idx % BN;
            Bs[i][j] = B[(kk + i) * N + (blockCol + j)];
        }

        __syncthreads();

        // load fragments from shared memory
        wmma::load_matrix_sync(
            a_frag,
            &As[warpRow * WM][0],
            BK
        );

        wmma::load_matrix_sync(
            b_frag,
            &Bs[0][warpCol * WN],
            BK
        );

        // Tensor Core MMA
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // write back
    wmma::store_matrix_sync(
        &C[(blockRow + warpRow * WM) * N + blockCol + warpCol * WN],
        c_frag,
        N,
        wmma::mem_row_major
    );
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

    constexpr int threadsPerBlock = (BM / WM) * (BN / WN) * 32;
    dim3 blocksPerGrid(N / BN, M / BM);

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

    printf("Tencor Core WMMA GEMM:\n");
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

// nvcc -arch=native ./tensor_core_WMMA.cu -lcublas -o ./tensor_core_WMMA && ./tensor_core_WMMA