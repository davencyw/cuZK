#pragma once

#include <functional>
#include <iostream>
#include <stdexcept>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace cuZK {
namespace ErrorHandling {

// Common exception types
class ValidationError : public std::invalid_argument {
public:
  explicit ValidationError(const std::string &message)
      : std::invalid_argument(message) {}
};

class ComputationError : public std::runtime_error {
public:
  explicit ComputationError(const std::string &message)
      : std::runtime_error(message) {}
};

class IndexError : public std::out_of_range {
public:
  explicit IndexError(const std::string &message)
      : std::out_of_range(message) {}
};

// Common validation functions
inline void validate_range(size_t value, size_t min_val, size_t max_val,
                           const std::string &param_name) {
  if (value < min_val || value > max_val) {
    throw ValidationError(
        param_name + " must be between " + std::to_string(min_val) + " and " +
        std::to_string(max_val) + ", got " + std::to_string(value));
  }
}

inline void validate_index(size_t index, size_t max_index,
                           const std::string &context) {
  if (index >= max_index) {
    throw IndexError(context + ": index " + std::to_string(index) +
                     " out of range [0, " + std::to_string(max_index) + ")");
  }
}

inline void validate_non_empty(size_t size, const std::string &context) {
  if (size == 0) {
    throw ValidationError(context + " cannot be empty");
  }
}

#ifdef CUDA_ENABLED
// Base CUDA error checking macro - handles error reporting
#define CUDA_CHECK_BASE(call, error_action)                                    \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(error) << std::endl;                     \
      error_action;                                                            \
    }                                                                          \
  } while (0)

// Main CUDA error checking macros
#define CUDA_CHECK_THROW(call)                                                 \
  CUDA_CHECK_BASE(                                                             \
      call, throw ComputationError("CUDA error at " + std::string(__FILE__) +  \
                                   ":" + std::to_string(__LINE__) + " - " +    \
                                   cudaGetErrorString(error)))

#define CUDA_CHECK_RETURN(call) CUDA_CHECK_BASE(call, return false)

#define CUDA_CHECK_VOID(call) CUDA_CHECK_BASE(call, (void)0)

// CUDA memory allocation with error checking
#define CUDA_MALLOC_CHECK(ptr, size)                                           \
  CUDA_CHECK_BASE(cudaMalloc(&ptr, size), ptr = nullptr)

// CUDA kernel launch error checking
#define CUDA_KERNEL_CHECK() CUDA_CHECK_BASE(cudaGetLastError(), return false)

// Generic CUDA check with custom cleanup/error handling
#define CUDA_CHECK_CLEANUP(call, cleanup_code)                                 \
  CUDA_CHECK_BASE(call, cleanup_code; return false)

// CUDA synchronization with error checking
inline bool cuda_sync_check() {
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << "CUDA sync error: " << cudaGetErrorString(error) << std::endl;
    return false;
  }
  return true;
}

// CUDA synchronization with cleanup on failure
template <typename CleanupFunc>
inline bool cuda_sync_check_cleanup(CleanupFunc cleanup) {
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << "CUDA sync error: " << cudaGetErrorString(error) << std::endl;
    cleanup();
    return false;
  }
  return true;
}

#endif // CUDA_ENABLED

} // namespace ErrorHandling
} // namespace cuZK