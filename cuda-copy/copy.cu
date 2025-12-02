#include<iostream>
#include <cstdlib>
#include <string>
#include "TimeElapsed.h"

// Print usage information
void printUsage(const char* programName) {
  std::cout << "Usage: " << programName << " <n> <threadnum> <kernel_name> [loop_unroll_times]" << std::endl;
  std::cout << "  n: Number of elements to copy (must be a positive integer)" << std::endl;
  std::cout << "  threadnum: Number of threads per block (must be a positive integer)" << std::endl;
  std::cout << "  kernel_name: Kernel to use (baseline, loop_unroll, vectorize, or vectorize_unroll)" << std::endl;
  std::cout << "  loop_unroll_times: Required when kernel_name is 'loop_unroll' (must be a positive integer)" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  " << programName << " 1000000 1024 baseline" << std::endl;
  std::cout << "  " << programName << " 1000000 1024 loop_unroll 4" << std::endl;
  std::cout << "  " << programName << " 1000000 1024 vectorize" << std::endl;
  std::cout << "  " << programName << " 1000000 1024 vectorize_unroll" << std::endl;
}

// Check CUDA error and exit if failed
void checkCudaError(cudaError_t err, const char* operation) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error in " << operation << ": " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

//baseline copy kernel
__global__ void copy_baseline(float* src, float* dst, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

//loop unroll copy kernel
__global__ void copy_loop_unroll(float* src, float* dst, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i+=stride) {
    dst[i] = src[i];
  }
}

//vectorize copy kernel
__global__ void copy_vectorize(float* src, float* dst, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int float4_count = n / 4;
  
  // Copy aligned float4 elements
  if (idx < float4_count) {
    reinterpret_cast<float4*>(dst)[idx] = reinterpret_cast<float4*>(src)[idx];
  }
  
  // Handle remaining elements (when n is not a multiple of 4)
  // Use threads from the last block to handle remaining elements
  // int remaining = n % 4;
  // if (remaining > 0 && blockIdx.x == gridDim.x - 1) {
  //   int remaining_start = float4_count * 4;
  //   int remaining_idx = remaining_start + threadIdx.x;
    
  //   // Only threads in the last block and within remaining count handle remaining elements
  //   if (threadIdx.x < remaining && remaining_idx < n) {
  //     dst[remaining_idx] = src[remaining_idx];
  //   }
  // }
}

// Optimized vectorize copy kernel with loop unrolling and read-only cache
// Each thread processes multiple float4 elements to improve memory bandwidth utilization
// Key optimizations:
// 1. Loop unrolling: each thread processes multiple float4 elements
// 2. __ldg() for read-only cache optimization (A100 has L2 read-only cache)
// 3. Reduced grid size: fewer blocks, more work per thread reduces overhead
// 4. Coalesced memory access: maintains 128-byte aligned access pattern
__global__ void copy_vectorize_unroll(float* src, float* dst, int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int float4_count = n / 4;
  
  // Calculate starting index for this thread
  int idx = tid + bid * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  
  // Process multiple float4 elements per thread (loop unroll)
  // This reduces the number of threads needed and improves memory bandwidth utilization
  // Each thread processes stride/blockDim.x float4 elements
  for (int i = idx; i < float4_count; i += stride) {
    // Use __ldg() for read-only data to utilize read-only cache
    // This can improve performance on devices with read-only cache (A100 has L2 cache)
    // __ldg() loads data through the read-only cache path, reducing L1 cache pollution
    float4 val = __ldg(reinterpret_cast<const float4*>(src) + i);
    reinterpret_cast<float4*>(dst)[i] = val;
  }
  
  // Handle remaining elements (when n is not a multiple of 4)
  // Only the last block handles remaining elements
  int remaining = n % 4;
  if (remaining > 0 && bid == gridDim.x - 1) {
    int remaining_start = float4_count * 4;
    int remaining_idx = remaining_start + tid;
    if (remaining_idx < n && tid < remaining) {
      dst[remaining_idx] = __ldg(src + remaining_idx);
    }
  }
}

int main(int argc, char** argv) {
  // Check minimum argument count
  if (argc < 4) {
    std::cerr << "Error: Invalid number of arguments." << std::endl;
    std::cerr << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // Parse and validate input parameters
  int n = 0;
  int threadnum = 0;
  std::string kernel_name;
  int loop_unroll_times = 0;
  
  try {
    n = std::stoi(argv[1]);
  } catch (const std::exception& e) {
    std::cerr << "Error: Invalid input parameter. '" << argv[1] << "' is not a valid integer." << std::endl;
    std::cerr << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  try {
    threadnum = std::stoi(argv[2]);
  } catch (const std::exception& e) {
    std::cerr << "Error: Invalid input parameter. '" << argv[2] << "' is not a valid integer." << std::endl;
    std::cerr << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  kernel_name = std::string(argv[3]);

  // Validate kernel name
  if (kernel_name != "baseline" && kernel_name != "loop_unroll" && kernel_name != "vectorize" && kernel_name != "vectorize_unroll") {
    std::cerr << "Error: Invalid kernel name. Must be 'baseline', 'loop_unroll', 'vectorize', or 'vectorize_unroll' (got '" << kernel_name << "')." << std::endl;
    std::cerr << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // Parse loop_unroll_times if using loop_unroll kernel
  if (kernel_name == "loop_unroll") {
    if (argc < 5) {
      std::cerr << "Error: loop_unroll_times parameter is required when using 'loop_unroll' kernel." << std::endl;
      std::cerr << std::endl;
      printUsage(argv[0]);
      return 1;
    }
    try {
      loop_unroll_times = std::stoi(argv[4]);
    } catch (const std::exception& e) {
      std::cerr << "Error: Invalid input parameter. '" << argv[4] << "' is not a valid integer." << std::endl;
      std::cerr << std::endl;
      printUsage(argv[0]);
      return 1;
    }
    if (loop_unroll_times <= 0) {
      std::cerr << "Error: loop_unroll_times must be a positive integer (got " << loop_unroll_times << ")." << std::endl;
      std::cerr << std::endl;
      printUsage(argv[0]);
      return 1;
    }
  }

  // Validate that n is positive
  if (n <= 0) {
    std::cerr << "Error: Number of elements must be a positive integer (got " << n << ")." << std::endl;
    std::cerr << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // Validate that threadnum is positive
  if (threadnum <= 0) {
    std::cerr << "Error: Number of threads per block must be a positive integer (got " << threadnum << ")." << std::endl;
    std::cerr << std::endl;
    printUsage(argv[0]);
    return 1;
  }
  //create host arrays
  float* host_src = new float[n];
  float* host_dst = new float[n];

  //initialize host arrays
  for (int i = 0; i < n; i++) {
    host_src[i] = i;
  }

  //copy host src to host dst
  for (int i = 0; i < n; i++) {
    host_dst[i] = host_src[i];
  }

  float* device_src = nullptr;
  float* device_dst = nullptr;
  checkCudaError(cudaMalloc((void**)&device_src, n * sizeof(float)), "cudaMalloc (device_src)");
  checkCudaError(cudaMalloc((void**)&device_dst, n * sizeof(float)), "cudaMalloc (device_dst)");
  checkCudaError(cudaMemcpy(device_src, host_src, n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy (HostToDevice)");

  dim3 block_size(threadnum);
  dim3 grid_size((n + block_size.x - 1) / block_size.x);
  
  // Launch kernel based on selection
  {
    RECORD_START();
    if (kernel_name == "baseline") {
      copy_baseline<<<grid_size, block_size>>>(device_src, device_dst, n);
    } else if (kernel_name == "loop_unroll") {
      dim3 loop_unroll_grid_size(grid_size.x / loop_unroll_times);
      copy_loop_unroll<<<loop_unroll_grid_size, block_size>>>(device_src, device_dst, n);
    } else if (kernel_name == "vectorize") {
      // Calculate grid size for float4 operations: need (n+3)/4 float4 elements
      int float4_count = (n + 3) / 4;
      dim3 vectorize_grid_size((float4_count + block_size.x - 1) / block_size.x);
      copy_vectorize<<<vectorize_grid_size, block_size>>>(device_src, device_dst, n);
    } else if (kernel_name == "vectorize_unroll") {
      // Optimized vectorize with loop unrolling
      // Strategy: Use fewer blocks but each thread processes more data
      // This improves memory bandwidth utilization by:
      // 1. Reducing kernel launch overhead
      // 2. Increasing work per thread (better instruction-level parallelism)
      // 3. Better utilizing memory bandwidth with fewer but more efficient threads
      int float4_count = (n + 3) / 4;
      // Calculate base grid size
      int base_grid_size = (float4_count + block_size.x - 1) / block_size.x;
      // Reduce grid size to make each thread process more data
      // This is a key optimization: fewer threads but each does more work
      // For A100, reducing grid size by 4-8x typically improves bandwidth
      // Each thread will process 4-8x more float4 elements
      int optimal_grid_size = (base_grid_size + 7) / 8;  // Reduce by ~8x for maximum bandwidth
      if (optimal_grid_size < 1) optimal_grid_size = 1;
      // Ensure we have enough threads to cover all data
      // If grid size is too small, increase it
      int min_grid_size = (float4_count + block_size.x * 8 - 1) / (block_size.x * 8);
      if (optimal_grid_size < min_grid_size) optimal_grid_size = min_grid_size;
      
      dim3 vectorize_unroll_grid_size(optimal_grid_size);
      copy_vectorize_unroll<<<vectorize_unroll_grid_size, block_size>>>(device_src, device_dst, n);
    }
    RECORD_STOP();
  }
  
  // Check for kernel launch errors
  checkCudaError(cudaGetLastError(), "kernel launch");
  // Synchronize to ensure kernel completes
  checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  
  checkCudaError(cudaMemcpy(host_dst, device_dst, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy (DeviceToHost)");

  for (int i = 0; i < n; i++) {
    if (host_dst[i] != host_src[i]) {
      std::cout << "Error: host_dst[" << i << "] = " << host_dst[i] << " != host_src[" << i << "] = " << host_src[i] << std::endl;
      return 1;
    }
  }
  std::cout << "Copy successful" << std::endl;
  //free device arrays
  cudaFree(device_src);
  cudaFree(device_dst);
  delete[] host_src;
  delete[] host_dst;
  return 0;
}