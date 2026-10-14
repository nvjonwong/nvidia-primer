#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Profile Test Helper
# Makes it easy to profile individual tests with Nsight Systems and list CUDA/NVTX events

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TEST_EXECUTABLE="${BUILD_DIR}/kokkos_tests"

# Function to check nsys availability
check_nsys() {
    if ! command -v nsys &> /dev/null; then
        echo -e "${RED}Error: nsys not found. Please ensure Nsight Systems is installed.${NC}"
        echo ""
        echo "Nsight Systems should be available in the container or on your system."
        echo "Check with: which nsys"
        echo ""
        exit 1
    fi
}

# Function to list all available tests
list_tests() {
    echo -e "${GREEN}Available Tests:${NC}"
    echo "=================="
    if [ -f "$TEST_EXECUTABLE" ]; then
        "$TEST_EXECUTABLE" --gtest_list_tests | grep -v "^  " | sed 's/\.$//'
    else
        echo -e "${RED}Test executable not found. Please build first:${NC}"
        echo "  cd ${SCRIPT_DIR} && ./build.sh"
        exit 1
    fi
}

# Function to list tests with their full names (including test cases)
list_all_tests() {
    echo -e "${GREEN}Available Test Cases:${NC}"
    echo "======================"
    if [ -f "$TEST_EXECUTABLE" ]; then
        "$TEST_EXECUTABLE" --gtest_list_tests
    else
        echo -e "${RED}Test executable not found. Please build first.${NC}"
        exit 1
    fi
}

# Function to profile a test
profile_test() {
    check_nsys
    
    local test_filter="$1"
    local output_name="${2:-profile}"
    local profile_output="${BUILD_DIR}/${output_name}"
    
    echo -e "${GREEN}Profiling test: ${BLUE}${test_filter}${NC}"
    echo "Output file: ${profile_output}.nsys-rep"
    echo ""
    
    # Run nsys with comprehensive options
    cd "$BUILD_DIR"
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --cuda-um-cpu-page-faults=true \
        --cuda-um-gpu-page-faults=true \
        --output="${output_name}" \
        --force-overwrite=true \
        --export=sqlite \
        "${TEST_EXECUTABLE}" --gtest_filter="${test_filter}"
    
    echo ""
    echo -e "${GREEN}Profile complete: ${profile_output}.nsys-rep${NC}"
    echo ""
    
    # Generate and display report
    echo -e "${YELLOW}=== CUDA API Summary ===${NC}"
    nsys stats "${profile_output}.nsys-rep" --report cuda_api_sum --force-export=true 2>&1 | head -30
    
    echo ""
    echo -e "${YELLOW}=== CUDA Kernel Summary ===${NC}"
    nsys stats "${profile_output}.nsys-rep" --report cuda_gpu_kern_sum --force-export=true 2>&1 | head -30
    
    echo ""
    echo -e "${YELLOW}=== NVTX Events ===${NC}"
    nsys stats "${profile_output}.nsys-rep" --report nvtx_sum --force-export=true 2>&1 | head -40
    
    echo ""
    echo -e "${GREEN}Full reports available with:${NC}"
    echo "  nsys stats ${profile_output}.nsys-rep --report cuda_api_sum"
    echo "  nsys stats ${profile_output}.nsys-rep --report cuda_gpu_kern_sum"
    echo "  nsys stats ${profile_output}.nsys-rep --report nvtx_sum"
    echo "  nsys stats ${profile_output}.nsys-rep --report cuda_gpu_mem_time_sum"
    echo ""
    echo -e "${GREEN}Open in GUI:${NC}"
    echo "  nsys-ui ${profile_output}.nsys-rep"
    echo ""
}

# Function to export detailed event list to text
export_events() {
    check_nsys
    
    local profile_file="$1"
    local output_txt="${profile_file%.nsys-rep}_events.txt"
    
    echo -e "${GREEN}Exporting detailed events to: ${output_txt}${NC}"
    
    {
        echo "======================================"
        echo "CUDA API CALLS (Detailed)"
        echo "======================================"
        nsys stats "$profile_file" --report cuda_api_sum --format csv --force-export=true 2>&1
        
        echo ""
        echo "======================================"
        echo "CUDA KERNELS (Detailed)"
        echo "======================================"
        nsys stats "$profile_file" --report cuda_gpu_kern_sum --format csv --force-export=true 2>&1
        
        echo ""
        echo "======================================"
        echo "NVTX RANGES (Detailed)"
        echo "======================================"
        nsys stats "$profile_file" --report nvtx_sum --format csv --force-export=true 2>&1
        
        echo ""
        echo "======================================"
        echo "MEMORY OPERATIONS"
        echo "======================================"
        nsys stats "$profile_file" --report cuda_gpu_mem_time_sum --format csv --force-export=true 2>&1
        
    } > "$output_txt"
    
    echo -e "${GREEN}Events exported to: ${output_txt}${NC}"
}

# Function to show help
show_help() {
    echo -e "${GREEN}Profile Test Helper${NC}"
    echo "===================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  list              List available test suites"
    echo "  list-all          List all test cases with full names"
    echo "  profile <filter>  Profile tests matching the filter"
    echo "  export <file>     Export detailed summary events from .nsys-rep"
    echo "  timeline <file>   Export chronological timeline (CUDA API + NVTX)"
    echo "  recipes           List installed nsys stats reports and recipes"
    echo "  names [mode]     Verify kernel renaming (unit|preflight|nsys|ncu|all)"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  # List all tests"
    echo "  $0 list"
    echo ""
    echo "  # Profile a specific test"
    echo "  $0 profile 'ExecutionSpace.DefaultSpace'"
    echo ""
    echo "  # Profile all tests in a suite"
    echo "  $0 profile 'ExecutionSpace.*'"
    echo ""
    echo "  # Profile with custom output name"
    echo "  $0 profile 'ExecutionSpace.DefaultSpace' my_profile"
    echo ""
    echo "  # Export summary events from profile"
    echo "  $0 export build/my_profile.nsys-rep"
    echo ""
    echo "  # Export chronological timeline (shows order of all operations)"
    echo "  $0 timeline build/my_profile.nsys-rep"
    echo ""
    echo "Filter syntax (gtest):"
    echo "  *                 Match any characters"
    echo "  Suite.Test        Match specific test"
    echo "  Suite.*           Match all tests in suite"
    echo "  *Pattern*         Match pattern anywhere"
    echo ""
}

# Function to export timeline
export_timeline() {
    check_nsys
    
    local profile_file="$1"
    local sqlite_file="${profile_file%.nsys-rep}_timeline"
    local output_txt="${profile_file%.nsys-rep}_timeline.txt"
    
    echo -e "${GREEN}Exporting timeline from: ${profile_file}${NC}"
    
    # Export to SQLite
    nsys export -t sqlite "$profile_file" --force-overwrite=true -o "$sqlite_file" 2>&1 | tail -3
    
    # Extract timeline using Python script
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$script_dir/extract_timeline.py" ]; then
        echo -e "${GREEN}Generating timeline...${NC}"
        python3 "$script_dir/extract_timeline.py" "$sqlite_file" -o "$output_txt"
        echo ""
        echo -e "${GREEN}Timeline exported to: ${output_txt}${NC}"
        echo ""
        echo "Preview:"
        head -30 "$output_txt"
        echo ""
        echo "(Full timeline in $output_txt)"
    else
        echo -e "${YELLOW}Note: extract_timeline.py not found, only SQLite export created${NC}"
        echo "SQLite file: $sqlite_file"
    fi
}

run_profiling_name_tests() {
    "${SCRIPT_DIR}/verify_profiling_names.sh" "${1:-all}"
}

discover_profiling_names() {
    python3 "${SCRIPT_DIR}/profiling_name_tools.py" --discover
}

# Parse command
case "${1:-help}" in
    list)
        list_tests
        ;;
    list-all)
        list_all_tests
        ;;
    profile)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Test filter required${NC}"
            echo "Usage: $0 profile <filter> [output_name]"
            echo ""
            echo "Available tests:"
            list_tests
            exit 1
        fi
        profile_test "$2" "${3:-profile}"
        ;;
    export)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Profile file required${NC}"
            echo "Usage: $0 export <file.nsys-rep>"
            exit 1
        fi
        export_events "$2"
        ;;
    timeline)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Profile file required${NC}"
            echo "Usage: $0 timeline <file.nsys-rep>"
            exit 1
        fi
        export_timeline "$2"
        ;;
    recipes)
        discover_profiling_names
        ;;
    names)
        run_profiling_name_tests "${2:-all}"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
