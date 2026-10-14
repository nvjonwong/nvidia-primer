#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# create_all_baselines.sh
#
# Creates golden baselines for ALL Kokkos test categories
# This comprehensive baseline can be used to track behavior changes across Kokkos versions
#
# Usage:
#   ./create_all_baselines.sh [kokkos_version]
#
# Example:
#   ./create_all_baselines.sh 5.0.1
#

set -e

# Get Kokkos version (default to 5.0.1)
KOKKOS_VERSION="${1:-5.0.1}"
BASELINE_DIR="baselines/kokkos_${KOKKOS_VERSION}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

find_kokkos_cache() {
    local candidate
    local kokkos_dir
    local kokkos_root

    for candidate in \
        "../../kokkos/build/CMakeCache.txt" \
        "../kokkos/build/CMakeCache.txt" \
        "${KOKKOS_BUILD_DIR:-}/CMakeCache.txt"; do
        if [ -n "$candidate" ] && [ -f "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done

    if [ -f "CMakeCache.txt" ]; then
        kokkos_dir=$(grep "^Kokkos_DIR:" CMakeCache.txt 2>/dev/null | cut -d= -f2- || true)
        if [ -n "$kokkos_dir" ]; then
            kokkos_root="${kokkos_dir%/lib/cmake/Kokkos}"
            for candidate in "$kokkos_root/../build/CMakeCache.txt" "$kokkos_root/CMakeCache.txt"; do
                if [ -f "$candidate" ]; then
                    echo "$candidate"
                    return
                fi
            done
        fi

        kokkos_root=$(grep "^Kokkos_ROOT:" CMakeCache.txt 2>/dev/null | cut -d= -f2- || true)
        if [ -n "$kokkos_root" ]; then
            for candidate in "$kokkos_root/../build/CMakeCache.txt" "$kokkos_root/CMakeCache.txt"; do
                if [ -f "$candidate" ]; then
                    echo "$candidate"
                    return
                fi
            done
        fi
    fi

    echo "CMakeCache.txt"
}

api_call_count() {
    local pattern="$1"
    local file="$2"

    if [ ! -f "$file" ]; then
        echo "0"
        return
    fi

    awk -F'|' -v pattern="$pattern" '
        function trim(s) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
            return s
        }
        {
            api = trim($10)
            calls = trim($4)
            if (api ~ pattern) {
                total += calls
            }
        }
        END {
            print total + 0
        }
    ' "$file"
}

echo "=============================================================================="
echo "Creating Comprehensive Golden Baseline for Kokkos ${KOKKOS_VERSION}"
echo "=============================================================================="
echo ""
echo "This will create baselines for:"
echo "  1. Allocation Threshold Tests (40KB boundary)"
echo "  2. Synchronization Behavior Tests"
echo "  3. Execution Space Tests"
echo "  4. View Tests"
echo "  5. Stream Ordered View Tests"
echo "  6. All Other Tests"
echo ""

# Create baseline directory
mkdir -p "${BASELINE_DIR}"

# Check if tests are built
if [ ! -f "build/kokkos_tests" ]; then
    echo "ERROR: kokkos_tests not found. Please build tests first:"
    echo "  ./build.sh"
    exit 1
fi

cd build
KOKKOS_CMAKE_CACHE=$(find_kokkos_cache)
TIMELINE_EXTRACTOR=""
for extractor in "../extract_timeline.py" "../scripts/extract_timeline.py"; do
    if [ -f "$extractor" ]; then
        TIMELINE_EXTRACTOR="$extractor"
        break
    fi
done

# Get system information (do once for all tests)
echo "=============================================================================="
echo "Step 1/7: Collecting System Information"
echo "=============================================================================="
{
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Kokkos Version: ${KOKKOS_VERSION}"
    echo "Baseline Created: ${TIMESTAMP}"
    echo ""
    
    echo "=== GPU Information ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
    else
        echo "nvidia-smi not available"
    fi
    echo ""
    
    echo "=== CUDA Version ==="
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release"
    else
        echo "nvcc not found"
    fi
    echo ""
    
    echo "=== Kokkos Build Configuration ==="
    if [ -f "$KOKKOS_CMAKE_CACHE" ]; then
        echo "Kokkos CMake Cache: ${KOKKOS_CMAKE_CACHE}"
        echo "Kokkos Flags:"
        grep "Kokkos_ENABLE" "$KOKKOS_CMAKE_CACHE" | grep -v "^//" | grep "=ON" || echo "  (none found)"
        echo ""
        echo "Critical Flags for Threshold Tests:"
        grep "Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC" "$KOKKOS_CMAKE_CACHE" | grep -v "^//" || echo "  KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC: OFF (not set)"
    else
        echo "Kokkos CMake cache not found"
    fi
    echo ""
    
    echo "=== Test Suite C++ Compiler ==="
    grep "CMAKE_CXX_COMPILER:" CMakeCache.txt 2>/dev/null | head -1 || echo "Not found"
    echo ""
    
    echo "=== Test Suite Build Type ==="
    grep "CMAKE_BUILD_TYPE:" CMakeCache.txt 2>/dev/null | head -1 || echo "Not found"
    echo ""
} > "../${BASELINE_DIR}/system_info.txt"

echo "✓ System information saved"
echo ""

# Test categories to profile
# Keys = baseline file prefix, values = "description"
# Filters are defined separately because bash associative arrays don't preserve order
declare -A TEST_CATEGORIES=(
    ["AllocationThreshold"]="40KB allocation threshold behavior"
    ["SyncBehavior"]="Synchronization patterns"
    ["ExecutionSpace"]="Execution space and stream behavior"
    ["Views"]="View creation and initialization"
    ["StreamOrderedView"]="Zero-sync allocation with StreamOrderedView"
    ["ViewTraits"]="View memory traits, atomics, and MDRange"
    ["UVMandILP"]="UVM aliasing, deallocation, and instruction-level parallelism"
    ["CudaReference"]="CUDA reference implementations"
)

# gtest filters per category (must match actual test suite names)
declare -A TEST_FILTERS=(
    ["AllocationThreshold"]="AllocationThreshold.*:AllocationThresholdInfo.*"
    ["SyncBehavior"]="SyncBehavior.*"
    ["ExecutionSpace"]="ExecutionSpace.*"
    ["Views"]="ViewCreation.*:ViewHelpers.*:Helpers.*"
    ["StreamOrderedView"]="StreamOrderedView.*"
    ["ViewTraits"]="ViewTraits.*:Atomics.*:MDRangePolicy.*"
    ["UVMandILP"]="UVMDeallocation.*:ViewAliasing.*:InstructionLevelParallelism.*"
    ["CudaReference"]="CudaReference.*"
)

# Run each test category
test_num=1
total_tests=${#TEST_CATEGORIES[@]}

for category in "${!TEST_CATEGORIES[@]}"; do
    filter="${TEST_FILTERS[$category]}"
    echo "=============================================================================="
    echo "Step $((test_num + 1))/$((total_tests + 1)): Profiling ${category} Tests"
    echo "Description: ${TEST_CATEGORIES[$category]}"
    echo "Filter: ${filter}"
    echo "=============================================================================="
    
    # Run tests first (without profiling)
    echo "Running tests without profiling..."
    ./kokkos_tests --gtest_filter="${filter}" 2>&1 | tee "../${BASELINE_DIR}/${category}_test_output.txt" || {
        echo "⚠️  WARNING: Some tests failed or skipped. Check ${category}_test_output.txt"
    }
    echo ""
    
    # Profile tests
    echo "Profiling tests with Nsight Systems..."
    nsys profile \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        --output="../${BASELINE_DIR}/${category}_baseline" \
        --force-overwrite=true \
        --sample=none \
        --backtrace=none \
        ./kokkos_tests --gtest_filter="${filter}" 2>&1 | grep -v "PASSED\|Skipping" || true
    
    echo "✓ ${category} baseline created"
    echo ""
    
    ((test_num++))
done

echo "=============================================================================="
echo "Step $((total_tests + 1))/$((total_tests + 1)): Generating Statistics and Analysis"
echo "=============================================================================="

# Generate statistics for each category
for category in "${!TEST_CATEGORIES[@]}"; do
    if [ -f "../${BASELINE_DIR}/${category}_baseline.nsys-rep" ]; then
        echo "Analyzing ${category}..."
        
        # Generate CSV statistics
        nsys stats "../${BASELINE_DIR}/${category}_baseline.nsys-rep" \
            --force-export=true \
            --report cuda_api_sum \
            --format csv \
            --output "../${BASELINE_DIR}/${category}" \
            --force-overwrite true 2>/dev/null || true
        
        # Extract synchronization info
        nsys stats "../${BASELINE_DIR}/${category}_baseline.nsys-rep" \
            --force-export=true \
            --report cuda_api_sum \
            --format table 2>/dev/null \
            | grep -E "Malloc|Free|Synchronize|Memcpy" \
            > "../${BASELINE_DIR}/${category}_sync.txt" 2>/dev/null || echo "No sync data" > "../${BASELINE_DIR}/${category}_sync.txt"
        
        # Extract timeline if script exists
        if [ -n "$TIMELINE_EXTRACTOR" ]; then
            python3 "$TIMELINE_EXTRACTOR" \
                "../${BASELINE_DIR}/${category}_baseline.sqlite" \
                > "../${BASELINE_DIR}/${category}_timeline.txt" 2>/dev/null || echo "Timeline extraction failed" > "../${BASELINE_DIR}/${category}_timeline.txt"
        else
            echo "Timeline extractor not found" > "../${BASELINE_DIR}/${category}_timeline.txt"
        fi
    fi
done

echo "✓ Statistics generated for all categories"
echo ""

# Create comprehensive summary
echo "Creating comprehensive summary..."
{
    echo "========================================================================"
    echo "Kokkos ${KOKKOS_VERSION} - Comprehensive Baseline Summary"
    echo "========================================================================"
    echo "Created: $(date)"
    echo "Baseline Directory: ${BASELINE_DIR}"
    echo ""
    
    echo "========================================================================"
    echo "Test Categories Profiled"
    echo "========================================================================"
    echo ""
    for category in "${!TEST_CATEGORIES[@]}"; do
        echo "✓ ${category}: ${TEST_CATEGORIES[$category]}"
    done
    echo ""
    
    echo "========================================================================"
    echo "Critical Configuration Checks"
    echo "========================================================================"
    echo ""
    
    # Check async malloc flag
    if grep -Eq "(Kokkos|KOKKOS)_ENABLE_IMPL_CUDA_MALLOC_ASYNC.*ON" "../${BASELINE_DIR}/system_info.txt" 2>/dev/null; then
        echo "✅ KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC: ON"
        echo "   → 40KB threshold tests will show allocation path differences"
    else
        echo "⚠️  KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC: OFF or not found"
        echo "   → 40KB threshold will NOT affect allocation behavior"
        echo "   → All allocations will use cudaMalloc/cudaFree"
    fi
    echo ""
    
    # Check debug mode
    if grep -q "CMAKE_BUILD_TYPE.*Debug" "../${BASELINE_DIR}/system_info.txt" 2>/dev/null; then
        echo "⚠️  Build Type: Debug"
        echo "   → May include extra synchronization for bounds checking"
    else
        echo "✅ Build Type: Release (or not Debug)"
        echo "   → Should have minimal implicit synchronization"
    fi
    echo ""
    
    echo "========================================================================"
    echo "Synchronization Summary by Category"
    echo "========================================================================"
    echo ""
    
    for category in "${!TEST_CATEGORIES[@]}"; do
        echo "--- ${category} ---"
        if [ -f "../${BASELINE_DIR}/${category}_sync.txt" ]; then
            # Count key synchronization calls
            malloc_count=$(api_call_count "^cudaMalloc$" "../${BASELINE_DIR}/${category}_sync.txt")
            malloc_async_count=$(api_call_count "^cudaMallocAsync" "../${BASELINE_DIR}/${category}_sync.txt")
            dev_sync_count=$(api_call_count "^cudaDeviceSynchronize$" "../${BASELINE_DIR}/${category}_sync.txt")
            stream_sync_count=$(api_call_count "^cudaStreamSynchronize$" "../${BASELINE_DIR}/${category}_sync.txt")
            
            echo "  cudaMalloc:            $malloc_count"
            echo "  cudaMallocAsync:       $malloc_async_count"
            echo "  cudaDeviceSynchronize: $dev_sync_count"
            echo "  cudaStreamSynchronize: $stream_sync_count"
        else
            echo "  (no sync data available)"
        fi
        echo ""
    done
    
    echo "========================================================================"
    echo "Files Created"
    echo "========================================================================"
    echo ""
    ls -lh "../${BASELINE_DIR}/" | tail -n +2
    echo ""
    
    echo "========================================================================"
    echo "Usage"
    echo "========================================================================"
    echo ""
    echo "View specific baseline:"
    echo "  nsys-ui ${BASELINE_DIR}/AllocationThreshold_baseline.nsys-rep"
    echo ""
    echo "View synchronization summary:"
    echo "  cat ${BASELINE_DIR}/AllocationThreshold_sync.txt"
    echo ""
    echo "View timeline:"
    echo "  less ${BASELINE_DIR}/AllocationThreshold_timeline.txt"
    echo ""
    echo "Compare with future version:"
    echo "  ./compare_all_baselines.sh ${KOKKOS_VERSION} <new_version>"
    echo ""
    
} > "../${BASELINE_DIR}/BASELINE_SUMMARY.txt"

cat "../${BASELINE_DIR}/BASELINE_SUMMARY.txt"

# Create metadata
{
    echo "=== Baseline Metadata ==="
    echo "Version: ${KOKKOS_VERSION}"
    echo "Created: $(date)"
    echo "Timestamp: ${TIMESTAMP}"
    echo "Categories: ${#TEST_CATEGORIES[@]}"
    echo ""
    echo "Test Categories:"
    for category in "${!TEST_CATEGORIES[@]}"; do
        echo "  - ${category}"
    done
} > "../${BASELINE_DIR}/metadata.txt"

echo ""
echo "=============================================================================="
echo "Comprehensive Golden Baseline Created Successfully!"
echo "=============================================================================="
echo ""
echo "Location: ${BASELINE_DIR}/"
echo ""
echo "Baseline includes:"
for category in "${!TEST_CATEGORIES[@]}"; do
    echo "  ✓ ${category} (profile, sync analysis, timeline)"
done
echo ""
echo "Key files:"
echo "  - BASELINE_SUMMARY.txt       (Comprehensive summary)"
echo "  - system_info.txt            (System configuration)"
echo "  - *_baseline.nsys-rep        (Nsight profiles)"
echo "  - *_sync.txt                 (Sync analysis per category)"
echo "  - *_timeline.txt             (Event timelines)"
echo "  - *_test_output.txt          (Test execution logs)"
echo ""
echo "To compare with future versions:"
echo "  ./compare_all_baselines.sh ${KOKKOS_VERSION} <new_version>"
echo ""
echo "To view allocation threshold baseline specifically:"
echo "  nsys-ui ${BASELINE_DIR}/AllocationThreshold_baseline.nsys-rep"
echo "  cat ${BASELINE_DIR}/AllocationThreshold_sync.txt"
echo ""
