# Test for Multivariate Polynomial Library
cmake_minimum_required(VERSION 3.16)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Test executable
add_executable(test_mv_polynomial test_mv_polynomial.cpp)

# Link with libraries
target_link_libraries(test_mv_polynomial 
    mv_polynomial
    poseidon
)

# Include directories
target_include_directories(test_mv_polynomial PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../poseidon
)

# Add test to ctest
add_test(NAME MvPolynomailTest COMMAND test_mv_polynomial) 