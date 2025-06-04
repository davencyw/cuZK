# Makefile for cuZK C++ Project
# This makefile provides convenient targets for building and testing all sub-projects

# Configuration
BUILD_DIR := build
CMAKE_BUILD_TYPE ?= Release
NUM_JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.PHONY: all
all: build

# Help target
.PHONY: help
help:
	@echo "$(BLUE)cuZK C++ Project Makefile$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  $(GREEN)help$(NC)          - Show this help message"
	@echo "  $(GREEN)configure$(NC)     - Configure CMake build system"
	@echo "  $(GREEN)build$(NC)         - Build all sub-projects"
	@echo "  $(GREEN)clean$(NC)         - Clean build directory"
	@echo ""
	@echo "$(YELLOW)Testing targets:$(NC)"
	@echo "  $(GREEN)test$(NC)          - Run all tests (summary output)"
	@echo "  $(GREEN)test-verbose$(NC)  - Run all tests (detailed output)"
	@echo "  $(GREEN)test-cuda$(NC)     - Run CUDA-specific tests (shows setup info if CUDA unavailable)"
	@echo "  $(GREEN)benchmark$(NC)     - Run benchmark tests with performance output"
	@echo ""
	@echo "$(YELLOW)Debug targets:$(NC)"
	@echo "  $(GREEN)debug$(NC)         - Build with debug information"
	@echo ""
	@echo "$(YELLOW)Utility targets:$(NC)"
	@echo "  $(GREEN)format$(NC)        - Format code (if clang-format available)"
	@echo "  $(GREEN)lint$(NC)          - Lint code (if cppcheck available)"
	@echo ""
	@echo "$(YELLOW)Variables:$(NC)"
	@echo "  CMAKE_BUILD_TYPE  - Build type (Release/Debug/RelWithDebInfo)"
	@echo "  NUM_JOBS          - Number of parallel jobs (default: $(NUM_JOBS))"

# Configure CMake
.PHONY: configure
configure:
	@echo "$(BLUE)Configuring CMake...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	@echo "$(GREEN)Configuration complete!$(NC)"

# Build all projects
.PHONY: build
build: configure
	@echo "$(BLUE)Building all sub-projects...$(NC)"
	@cd $(BUILD_DIR) && make -j$(NUM_JOBS)
	@echo "$(GREEN)Build complete!$(NC)"

# Debug build
.PHONY: debug
debug:
	@echo "$(BLUE)Building debug version...$(NC)"
	@$(MAKE) build CMAKE_BUILD_TYPE=Debug
	@echo "$(GREEN)Debug build complete!$(NC)"

# Clean build directory
.PHONY: clean
clean:
	@echo "$(YELLOW)Cleaning build directory...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)Clean complete!$(NC)"

# Run all tests
.PHONY: test
test: build
	@echo "$(BLUE)Running all tests...$(NC)"
	@cd $(BUILD_DIR) && make run_all_tests
	@echo "$(GREEN)All tests completed!$(NC)"

# Run all tests with verbose output
.PHONY: test-verbose
test-verbose: build
	@echo "$(BLUE)Running all tests with verbose output...$(NC)"
	@cd $(BUILD_DIR) && ctest --verbose
	@echo "$(GREEN)Verbose tests completed!$(NC)"

# Run benchmark tests
.PHONY: benchmark
benchmark: build
	@echo "$(BLUE)Running benchmark tests...$(NC)"
	@cd $(BUILD_DIR) && ctest -R "Benchmark" --verbose
	@echo "$(GREEN)Benchmark tests completed!$(NC)"

# Run CUDA tests (if available)
.PHONY: test-cuda
test-cuda: build
	@echo "$(BLUE)Running CUDA tests...$(NC)"
	@if cd $(BUILD_DIR) && ctest -R "merkle_tree_cuda_tests" --verbose; then \
		echo "$(GREEN)CUDA tests completed!$(NC)"; \
	else \
		echo "$(YELLOW)CUDA tests unavailable or failed. This might be expected if CUDA is not installed.$(NC)"; \
		echo "$(YELLOW)Check CUDA installation and GPU availability.$(NC)"; \
	fi

# Format code (if clang-format is available)
.PHONY: format
format:
	@if command -v clang-format >/dev/null 2>&1; then \
		echo "$(BLUE)Formatting code...$(NC)"; \
		find src -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i; \
		echo "$(GREEN)Code formatting complete!$(NC)"; \
	else \
		echo "$(YELLOW)clang-format not found, skipping formatting$(NC)"; \
	fi

# Lint code (if cppcheck is available)
.PHONY: lint
lint:
	@if command -v cppcheck >/dev/null 2>&1; then \
		echo "$(BLUE)Linting code...$(NC)"; \
		cppcheck --enable=all --std=c++17 --suppress=missingIncludeSystem src/; \
		echo "$(GREEN)Code linting complete!$(NC)"; \
	else \
		echo "$(YELLOW)cppcheck not found, skipping linting$(NC)"; \
	fi

