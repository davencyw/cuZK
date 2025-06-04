cmake_minimum_required(VERSION 3.14)
project(cuZK VERSION 0.0.1 LANGUAGES CXX)

# Enable compile_commands.json generation for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    # Find CUDA toolkit
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA support enabled")
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found - building CPU-only version")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Include Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Include sub-projects
add_subdirectory(src/poseidon)
add_subdirectory(src/merkle_tree)

# Add common include directory for all targets
include_directories(src/common)

# Collect all test targets
set(ALL_TEST_TARGETS poseidon_tests merkle_tree_tests)

# Add CUDA tests if available
if(CUDA_AVAILABLE)
    list(APPEND ALL_TEST_TARGETS merkle_tree_cuda_tests)
endif()

# Create a target to run all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${ALL_TEST_TARGETS}
    COMMENT "Running all tests"
)
