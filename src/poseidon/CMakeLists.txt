# Poseidon Hash Library
cmake_minimum_required(VERSION 3.14)

# Create the poseidon library with CPU sources
set(POSEIDON_CPU_SOURCES
    poseidon.cpp
    field_arithmetic.cpp
)

add_library(poseidon ${POSEIDON_CPU_SOURCES})
target_include_directories(poseidon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create optional CUDA extension library
if(CUDA_AVAILABLE)
    set(POSEIDON_CUDA_SOURCES
        cuda/cuda_field_element.cu
        cuda/field_arithmetic_cuda.cu
        cuda/poseidon_cuda.cu
        cuda/poseidon_cuda_optimized.cu
        cuda/poseidon_cuda_benchmarks.cpp
    )
    
    add_library(poseidon_cuda ${POSEIDON_CUDA_SOURCES})
    
    # Set CUDA properties
    set_target_properties(poseidon_cuda PROPERTIES 
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "70;75;80;86"
    )
    
    # Link CUDA runtime and base poseidon library
    target_link_libraries(poseidon_cuda 
        poseidon
        CUDA::cudart
    )
    
    target_include_directories(poseidon_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(poseidon_cuda PUBLIC CUDA_ENABLED)
    
    # Also add CUDA definitions to base library when CUDA is available
    target_compile_definitions(poseidon PUBLIC CUDA_ENABLED)
endif()

# Create CPU tests
set(POSEIDON_CPU_TEST_SOURCES
    test/test_poseidon.cpp
    test/test_field_arithmetic.cpp
    test/test_field_accumulation.cpp
)

add_executable(poseidon_tests ${POSEIDON_CPU_TEST_SOURCES})

target_link_libraries(poseidon_tests 
    poseidon
    gtest_main
    Threads::Threads
)

# Add CPU tests to CTest
include(GoogleTest)
gtest_discover_tests(poseidon_tests)

# Create CUDA tests if CUDA is available
if(CUDA_AVAILABLE)
    # CUDA functionality tests
    add_executable(poseidon_cuda_tests
        test/test_field_arithmetic_cuda.cpp
        test/test_poseidon_cuda.cpp
    )
    
    target_link_libraries(poseidon_cuda_tests
        poseidon_cuda
        poseidon
        gtest_main
        Threads::Threads
    )
    
    # Add CUDA tests to CTest using add_test for better CUDA compatibility
    add_test(NAME poseidon_cuda_tests COMMAND poseidon_cuda_tests)
    
    # Add custom target for running CUDA functionality tests
    add_custom_target(test_poseidon_cuda
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "poseidon_cuda_tests"
        DEPENDS poseidon_cuda_tests
        COMMENT "Running Poseidon CUDA tests"
    )
endif()

# Add custom target for running just CPU tests
add_custom_target(test_poseidon_cpu
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/poseidon_tests
    DEPENDS poseidon_tests
    COMMENT "Running Poseidon CPU tests"
)

# Add custom target for running all poseidon tests
if(CUDA_AVAILABLE)
    add_custom_target(test_poseidon_all
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "PoseidonTest|FieldArithmeticTest|FieldAccumulationTest|poseidon_cuda_tests"
        DEPENDS poseidon_tests poseidon_cuda_tests
        COMMENT "Running all Poseidon tests (CPU and CUDA)"
    )
else()
    add_custom_target(test_poseidon_all
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "PoseidonTest|FieldArithmeticTest|FieldAccumulationTest"
        DEPENDS poseidon_tests
        COMMENT "Running all Poseidon tests (CPU only)"
    )
endif()

# Create benchmark executable
add_executable(poseidon_benchmark
    test/benchmark.cpp
)

target_link_libraries(poseidon_benchmark
    poseidon
    Threads::Threads
)

# Add CUDA support to benchmark if available
if(CUDA_AVAILABLE)
    target_link_libraries(poseidon_benchmark poseidon_cuda)
    target_compile_definitions(poseidon_benchmark PRIVATE CUDA_ENABLED)
endif()

# Create CUDA profiler executable if CUDA is available
if(CUDA_AVAILABLE)
    add_executable(poseidon_cuda_profiler
        cuda/poseidon_cuda_profiler.cpp
    )
    
    target_link_libraries(poseidon_cuda_profiler
        poseidon_cuda
        poseidon
        Threads::Threads
    )
    
    # Add custom target for running CUDA profiling with Nsight Systems
    add_custom_target(profile_poseidon_cuda
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../../profile_poseidon_nsys.sh --basic-only
        DEPENDS poseidon_cuda_profiler
        COMMENT "Running Poseidon CUDA kernel profiling with NVIDIA Nsight Systems"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    
    # Add custom target for comprehensive profiling
    add_custom_target(profile_poseidon_comprehensive
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../../profile_poseidon_nsys.sh -c
        DEPENDS poseidon_cuda_profiler
        COMMENT "Running comprehensive Poseidon CUDA kernel profiling"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

# Add custom target for running benchmarks
add_custom_target(benchmark_poseidon
    COMMAND $<TARGET_FILE:poseidon_benchmark>
    DEPENDS poseidon_benchmark
    COMMENT "Running Poseidon performance benchmarks"
)

# Maintain backward compatibility
add_custom_target(test_poseidon 
    DEPENDS test_poseidon_all
    COMMENT "Running Poseidon tests (alias for test_poseidon_all)"
) 