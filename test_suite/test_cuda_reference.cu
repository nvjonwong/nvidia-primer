/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_cuda_reference.cpp
 * @brief Pure CUDA reference implementations for performance comparison
 * 
 * These tests implement the same operations as the Kokkos tests but using
 * raw CUDA. This allows profiling to verify that Kokkos optimizations
 * generate equivalent GPU code.
 */

#include <gtest/gtest.h>

#ifdef __NVCC__

#include <cuda_runtime.h>
#include <vector>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace CudaReference {

// ============================================================================
// View Operations - CUDA Reference
// ============================================================================

__global__ void init_kernel(double* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = i * 3.14;
    }
}

__global__ void sum_reduce_kernel(const double* data, double* partial_sums, int N) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    sdata[tid] = (i < N) ? data[i] : 0.0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

void test_view_uninitialized_cuda() {
    const int N = 1000;
    double* d_data;
    
    // Allocate without initialization (fast)
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(double)));
    
    // Initialize with kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    init_kernel<<<gridSize, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compute sum with reduction
    int numBlocks = gridSize;
    double* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, numBlocks * sizeof(double)));
    
    sum_reduce_kernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(
        d_data, d_partial, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Final reduction on CPU
    std::vector<double> h_partial(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, 
                         numBlocks * sizeof(double), cudaMemcpyDeviceToHost));
    
    double sum = 0;
    for (double val : h_partial) {
        sum += val;
    }
    
    double expected = N * (N - 1) / 2.0 * 3.14;
    EXPECT_NEAR(sum, expected, 1e-6);
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_partial));
}

// ============================================================================
// Execution Spaces - CUDA Reference
// ============================================================================

__global__ void range_reduce_kernel(int* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = i * 2;
    }
}

__global__ void int_sum_reduce_kernel(const int* data, int* partial_sums, int N) {
    extern __shared__ int sdata_int[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata_int[tid] = (i < N) ? data[i] : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_int[tid] += sdata_int[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata_int[0];
    }
}

void test_explicit_stream_cuda() {
    const int N = 1000;
    int* d_data;
    cudaStream_t stream;
    
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Launch on explicit stream
    range_reduce_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Reduce on same stream
    int* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, gridSize * sizeof(int)));
    
    int_sum_reduce_kernel<<<gridSize, blockSize, blockSize * sizeof(int), stream>>>(
        d_data, d_partial, N);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    std::vector<int> h_partial(gridSize);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, 
                         gridSize * sizeof(int), cudaMemcpyDeviceToHost));
    
    int sum = 0;
    for (int val : h_partial) {
        sum += val;
    }
    
    int expected = N * (N - 1);
    EXPECT_EQ(sum, expected);
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Atomics - CUDA Reference
// ============================================================================

__global__ void atomic_add_kernel(int* counter, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(counter, 1);
    }
}

void test_atomic_add_cuda() {
    const int N = 1000;
    int* d_counter;
    
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    atomic_add_kernel<<<gridSize, blockSize>>>(d_counter, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int h_counter;
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    
    EXPECT_EQ(h_counter, N);
    
    CUDA_CHECK(cudaFree(d_counter));
}

// ============================================================================
// MDRangePolicy - CUDA Reference
// ============================================================================

__global__ void md_range_2d_kernel(int* data, int M, int N) {
    // Mimic MDRangePolicy with {1, 64} tiling
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < M && i < N) {
        data[j * N + i] = j + i;
    }
}

void test_mdrange_2d_cuda() {
    const int M = 100;
    const int N = 200;
    int* d_data;
    
    CUDA_CHECK(cudaMalloc(&d_data, M * N * sizeof(int)));
    
    // Use recommended tiling: {1, 64}
    dim3 blockSize(64, 1);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    md_range_2d_kernel<<<gridSize, blockSize>>>(d_data, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check results
    std::vector<int> h_data(M * N);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, 
                         M * N * sizeof(int), cudaMemcpyDeviceToHost));
    
    EXPECT_EQ(h_data[0], 0);  // (0,0)
    EXPECT_EQ(h_data[(M-1) * N + (N-1)], M + N - 2);  // (M-1, N-1)
    
    CUDA_CHECK(cudaFree(d_data));
}

// ============================================================================
// ILP - Avoid Global Memory Dependencies (CUDA Reference)
// ============================================================================

__global__ void ilp_bad_kernel(double* data, int N, double threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // BAD: Read-after-write from global memory
        data[i] = data[i] + 1.0;
        if (data[i] > threshold) {  // Read from global - SLOW
            data[i] *= 2.0;
        }
    }
}

__global__ void ilp_good_kernel(double* data, int N, double threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // GOOD: Use register for intermediate value
        double value = data[i];
        value = value + 1.0;
        if (value > threshold) {  // Use register - FAST
            value *= 2.0;
        }
        data[i] = value;
    }
}

void test_ilp_optimization_cuda() {
    const int N = 1000;
    const double threshold = 500.0;
    double* d_data;
    
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(double)));
    
    // Initialize
    std::vector<double> h_init(N);
    for (int i = 0; i < N; i++) {
        h_init[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_init.data(), 
                         N * sizeof(double), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    ilp_good_kernel<<<gridSize, blockSize>>>(d_data, N, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<double> h_result(N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, 
                         N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Verify results
    for (int i = 0; i < N; i++) {
        double expected = i + 1.0;
        if (expected > threshold) {
            expected *= 2.0;
        }
        EXPECT_DOUBLE_EQ(h_result[i], expected);
    }
    
    CUDA_CHECK(cudaFree(d_data));
}

// ============================================================================
// Memory Coalescing - CUDA Reference
// ============================================================================

__global__ void coalesced_access_kernel(const double* input, double* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Coalesced: consecutive threads access consecutive memory
        output[i] = input[i] * 2.0;
    }
}

__global__ void strided_access_kernel(const double* input, double* output, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Strided: consecutive threads access memory with stride
        int idx = i * stride;
        if (idx < N) {
            output[idx] = input[idx] * 2.0;
        }
    }
}

void test_memory_coalescing_cuda() {
    const int N = 1000;
    double *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(double)));
    
    // Initialize
    std::vector<double> h_init(N);
    for (int i = 0; i < N; i++) {
        h_init[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_init.data(), 
                         N * sizeof(double), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Test coalesced access (this should be fast)
    coalesced_access_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<double> h_result(N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_output, 
                         N * sizeof(double), cudaMemcpyDeviceToHost));
    
    EXPECT_DOUBLE_EQ(h_result[0], 0.0);
    EXPECT_DOUBLE_EQ(h_result[10], 20.0);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

} // namespace CudaReference

// ============================================================================
// Google Test Wrappers
// ============================================================================

TEST(CudaReference, ViewUninitialized) {
    CudaReference::test_view_uninitialized_cuda();
}

TEST(CudaReference, ExplicitStream) {
    CudaReference::test_explicit_stream_cuda();
}

TEST(CudaReference, AtomicAdd) {
    CudaReference::test_atomic_add_cuda();
}

TEST(CudaReference, MDRange2D) {
    CudaReference::test_mdrange_2d_cuda();
}

TEST(CudaReference, ILPOptimization) {
    CudaReference::test_ilp_optimization_cuda();
}

TEST(CudaReference, MemoryCoalescing) {
    CudaReference::test_memory_coalescing_cuda();
}

#endif // __NVCC__
