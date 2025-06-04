# Poseidon Hash Library
cmake_minimum_required(VERSION 3.14)

# Create the poseidon library
set(POSEIDON_SOURCES
    poseidon.cpp
    field_arithmetic.cpp
)

# Add CUDA sources if CUDA is available
if(CUDA_AVAILABLE)
    list(APPEND POSEIDON_SOURCES cuda_field_element.cu field_arithmetic_cuda.cu poseidon_cuda.cu)
    add_library(poseidon ${POSEIDON_SOURCES})
    
    # Set CUDA properties
    set_target_properties(poseidon PROPERTIES 
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )
    
    # Link CUDA runtime (modern approach) - CUDAToolkit found at top level
    target_link_libraries(poseidon CUDA::cudart)
    
    target_compile_definitions(poseidon PUBLIC CUDA_ENABLED)
else()
    add_library(poseidon ${POSEIDON_SOURCES})
endif()

target_include_directories(poseidon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create tests
set(POSEIDON_TEST_SOURCES
    test/test_poseidon.cpp
    test/test_field_arithmetic.cpp
)

# Add CUDA tests if CUDA is available
if(CUDA_AVAILABLE)
    list(APPEND POSEIDON_TEST_SOURCES test/test_field_arithmetic_cuda.cpp test/test_poseidon_cuda.cpp)
endif()

add_executable(poseidon_tests ${POSEIDON_TEST_SOURCES})

target_link_libraries(poseidon_tests 
    poseidon
    gtest_main
    Threads::Threads
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(poseidon_tests)

# Add custom target for running just this project's tests
add_custom_target(test_poseidon 
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "poseidon_tests"
    DEPENDS poseidon_tests
    COMMENT "Running Poseidon tests"
) 