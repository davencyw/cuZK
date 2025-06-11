# Matrix Multiplication Verification using Freivald's Algorithm
cmake_minimum_required(VERSION 3.14)

# Find Eigen3
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    # If system Eigen is not found, fetch it
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(Eigen)
    
    # Create an alias for the fetched version
    add_library(Eigen3::Eigen ALIAS eigen)
endif()

# Create the Freivald's algorithm library
add_library(freivald STATIC
    matrix_generator.cpp
    verifier.cpp
)

target_include_directories(freivald PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(freivald PUBLIC
    Eigen3::Eigen
)

target_compile_features(freivald PUBLIC cxx_std_17)

# Create the test executable
add_executable(freivald_tests
    test/test_freivald.cpp
)

target_link_libraries(freivald_tests PRIVATE
    freivald
    gtest_main
    gtest
    Eigen3::Eigen
)

target_compile_features(freivald_tests PRIVATE cxx_std_17)

# Add test to CTest
add_test(
    NAME freivald_tests
    COMMAND freivald_tests
)

# Create the example executable
add_executable(freivald_example
    example.cpp
)

target_link_libraries(freivald_example PRIVATE
    freivald
    Eigen3::Eigen
)

target_compile_features(freivald_example PRIVATE cxx_std_17)

# Create the benchmark executable
add_executable(freivald_benchmark
    test/benchmark_verification.cpp
)

target_link_libraries(freivald_benchmark PRIVATE
    freivald
    Eigen3::Eigen
)

target_compile_features(freivald_benchmark PRIVATE cxx_std_17)




 