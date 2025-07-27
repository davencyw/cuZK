# N-ary Merkle Tree Library
cmake_minimum_required(VERSION 3.14)

# Create the merkle_tree library with CPU sources
set(MERKLE_TREE_CPU_SOURCES
    merkle_tree.cpp
)

add_library(merkle_tree ${MERKLE_TREE_CPU_SOURCES})

# Link with poseidon library (merkle tree depends on poseidon hash)
target_link_libraries(merkle_tree 
    poseidon
)

target_include_directories(merkle_tree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create optional CUDA extension library
if(CUDA_AVAILABLE)
    set(MERKLE_TREE_CUDA_SOURCES
        merkle_tree_cuda.cu
        # Include Poseidon CUDA sources directly to avoid device linking issues
        ../poseidon/cuda/cuda_field_element.cu
        ../poseidon/cuda/field_arithmetic_cuda.cu
        ../poseidon/cuda/poseidon_cuda.cu
    )
    
    add_library(merkle_tree_cuda ${MERKLE_TREE_CUDA_SOURCES})
    
    # Set CUDA properties
    set_target_properties(merkle_tree_cuda PROPERTIES 
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "70;75;80;86"
    )
    
    # Link CUDA runtime and dependencies
    target_link_libraries(merkle_tree_cuda
        merkle_tree
        poseidon
        CUDA::cudart
    )
    
    target_include_directories(merkle_tree_cuda PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ../poseidon
        ../common
    )
    
    target_compile_definitions(merkle_tree_cuda PUBLIC CUDA_ENABLED)
    
    # Also add CUDA definitions to base library when CUDA is available
    target_compile_definitions(merkle_tree PUBLIC CUDA_ENABLED)
endif()

# Create CPU tests
set(MERKLE_TREE_CPU_TEST_SOURCES
    test/test_merkle_tree.cpp
    test/test_merkle_benchmark.cpp
)

add_executable(merkle_tree_tests ${MERKLE_TREE_CPU_TEST_SOURCES})

target_link_libraries(merkle_tree_tests 
    merkle_tree
    poseidon
    gtest_main
    Threads::Threads
)

# Add CPU tests to CTest
include(GoogleTest)
gtest_discover_tests(merkle_tree_tests)

# Create CUDA tests if CUDA is available
if(CUDA_AVAILABLE)
    # CUDA functionality tests
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
    
    # Add CUDA tests to CTest using add_test for better CUDA compatibility
    add_test(NAME merkle_tree_cuda_tests COMMAND merkle_tree_cuda_tests)
    add_test(NAME merkle_tree_cuda_benchmark_tests COMMAND merkle_tree_cuda_benchmark_tests)
    
    # Add custom target for running CUDA functionality tests
    add_custom_target(test_merkle_tree_cuda
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_cuda_tests"
        DEPENDS merkle_tree_cuda_tests
        COMMENT "Running Merkle Tree CUDA tests"
    )
    
    # Add custom target for running CUDA benchmarks
    add_custom_target(benchmark_merkle_tree_cuda
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_cuda_benchmark_tests"
        DEPENDS merkle_tree_cuda_benchmark_tests
        COMMENT "Running Merkle Tree CUDA benchmarks"
    )
    
    # Add combined CUDA test target
    add_custom_target(test_merkle_tree_cuda_all
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_cuda.*"
        DEPENDS merkle_tree_cuda_tests merkle_tree_cuda_benchmark_tests
        COMMENT "Running all Merkle Tree CUDA tests and benchmarks"
    )
endif()

# Add custom target for running just CPU tests
add_custom_target(test_merkle_tree_cpu
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_tests"
    DEPENDS merkle_tree_tests
    COMMENT "Running Merkle Tree CPU tests"
)

# Add custom target for running CPU benchmarks
add_custom_target(benchmark_merkle_tree_cpu
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_tests.*Benchmark"
    DEPENDS merkle_tree_tests
    COMMENT "Running Merkle Tree CPU benchmarks"
)

# Add custom target for running all merkle tree tests
if(CUDA_AVAILABLE)
    add_custom_target(test_merkle_tree_all
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree.*"
        DEPENDS merkle_tree_tests merkle_tree_cuda_tests merkle_tree_cuda_benchmark_tests
        COMMENT "Running all Merkle Tree tests (CPU and CUDA)"
    )
    
    add_custom_target(benchmark_merkle_tree_all
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree.*[Bb]enchmark"
        DEPENDS merkle_tree_tests merkle_tree_cuda_benchmark_tests
        COMMENT "Running all Merkle Tree benchmarks (CPU and CUDA)"
    )
else()
    add_custom_target(test_merkle_tree_all
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_tests"
        DEPENDS merkle_tree_tests
        COMMENT "Running all Merkle Tree tests (CPU only)"
    )
    
    add_custom_target(benchmark_merkle_tree_all
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "merkle_tree_tests.*Benchmark"
        DEPENDS merkle_tree_tests
        COMMENT "Running all Merkle Tree benchmarks (CPU only)"
    )
endif()

# Maintain backward compatibility
add_custom_target(test_merkle_tree 
    DEPENDS test_merkle_tree_all
    COMMENT "Running Merkle Tree tests (alias for test_merkle_tree_all)"
)

add_custom_target(benchmark_merkle_tree 
    DEPENDS benchmark_merkle_tree_all
    COMMENT "Running Merkle Tree benchmarks (alias for benchmark_merkle_tree_all)"
) 