# Multivariate Polynomial Library
cmake_minimum_required(VERSION 3.16)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(MV_POLYNOMIAL_SOURCES
    mv_polynomial.cpp
    mv_polynomial.hpp
)

# Create the library
add_library(mv_polynomial ${MV_POLYNOMIAL_SOURCES})

# Include directories
target_include_directories(mv_polynomial PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../poseidon
)

# Link with poseidon field arithmetic
target_link_libraries(mv_polynomial poseidon)

# Compiler flags
target_compile_options(mv_polynomial PRIVATE
    -Wall
    -Wextra
    -O3
    -march=native
)

# Add test subdirectory if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
    add_subdirectory(test)
endif()

# Install the library
install(TARGETS mv_polynomial
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES mv_polynomial.hpp
    DESTINATION include/mv_polynomial
) 