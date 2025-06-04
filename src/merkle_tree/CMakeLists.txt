# N-ary Merkle Tree Library
cmake_minimum_required(VERSION 3.14)

# Create the merkle_tree library
add_library(merkle_tree
    merkle_tree.cpp
)

# Link with poseidon library (merkle tree depends on poseidon hash)
target_link_libraries(merkle_tree 
    poseidon
)

target_include_directories(merkle_tree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA Merkle Tree Library (optional, only if CUDA is available)
if(CUDA_AVAILABLE)
    # Create CUDA merkle tree library
    add_library(merkle_tree_cuda
        merkle_tree_cuda.cu
        ../poseidon/poseidon_cuda.cu
        ../poseidon/cuda_field_element.cu
        ../poseidon/field_arithmetic_cuda.cu
    )
    
    # Set CUDA properties
    set_target_properties(merkle_tree_cuda PROPERTIES 
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )
    
    # Link CUDA runtime
    target_link_libraries(merkle_tree_cuda
        merkle_tree
        poseidon
        CUDA::cudart
        CUDA::curand
    )
    
    target_include_directories(merkle_tree_cuda PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ../poseidon
    )
    
    target_compile_definitions(merkle_tree_cuda PUBLIC CUDA_ENABLED)
    
    # Enable CUDA separable compilation
    set_property(TARGET merkle_tree_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Create tests
add_executable(merkle_tree_tests
    test/test_merkle_tree.cpp
    test/test_merkle_benchmark.cpp
)

target_link_libraries(merkle_tree_tests 
    merkle_tree
    poseidon
    gtest_main
    Threads::Threads
)

# Include GoogleTest for both regular and CUDA tests
include(GoogleTest)

# CUDA tests (optional)
if(CUDA_AVAILABLE)
    add_executable(merkle_tree_cuda_tests
        test/test_merkle_tree_cuda.cpp
    )
    
    target_link_libraries(merkle_tree_cuda_tests
        merkle_tree_cuda
        merkle_tree
        poseidon
        gtest_main
        Threads::Threads
    )
    
    # CUDA benchmark tests
    add_executable(merkle_tree_cuda_benchmark_tests
        test/test_merkle_benchmark_cuda.cpp
    )
    
    target_link_libraries(merkle_tree_cuda_benchmark_tests
        merkle_tree_cuda
        merkle_tree
        poseidon
        gtest_main
        Threads::Threads
    )
    
    # Add CUDA tests to CTest using add_test (more reliable than gtest_discover_tests for CUDA)
    add_test(NAME merkle_tree_cuda_tests COMMAND merkle_tree_cuda_tests)
    add_test(NAME merkle_tree_cuda_benchmark_tests COMMAND merkle_tree_cuda_benchmark_tests)
    
    # Add custom target for running CUDA tests
    add_custom_target(test_merkle_tree_cuda
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_cuda_tests"
        DEPENDS merkle_tree_cuda_tests
        COMMENT "Running CUDA Merkle Tree tests"
    )
    
    # Add custom target for running CUDA benchmarks
    add_custom_target(benchmark_merkle_tree_cuda
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_cuda_benchmark_tests"
        DEPENDS merkle_tree_cuda_benchmark_tests
        COMMENT "Running CUDA Merkle Tree benchmarks"
    )
    
    # Add combined CUDA test target
    add_custom_target(test_all_cuda
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_cuda.*"
        DEPENDS merkle_tree_cuda_tests merkle_tree_cuda_benchmark_tests
        COMMENT "Running all CUDA Merkle Tree tests and benchmarks"
    )
endif()

# Add regular tests to CTest
gtest_discover_tests(merkle_tree_tests)

# Add custom target for running just this project's tests
add_custom_target(test_merkle_tree 
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_tests"
    DEPENDS merkle_tree_tests
    COMMENT "Running Merkle Tree tests"
)

# Add custom target for running just the benchmark tests
add_custom_target(benchmark_merkle_tree 
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_tests.*Benchmark"
    DEPENDS merkle_tree_tests
    COMMENT "Running Merkle Tree benchmarks"
) 