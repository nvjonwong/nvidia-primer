#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# compare_all_baselines.sh
#
# Compares comprehensive baselines between two Kokkos versions
#
# Usage:
#   ./compare_all_baselines.sh <baseline_version> <new_version>
#
# Example:
#   ./compare_all_baselines.sh 5.0.1 5.1.0
#

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <baseline_version> <new_version>"
    echo "Example: $0 5.0.1 5.1.0"
    exit 1
fi

BASELINE_VERSION="$1"
NEW_VERSION="$2"

BASELINE_DIR="baselines/kokkos_${BASELINE_VERSION}"
NEW_DIR="baselines/kokkos_${NEW_VERSION}"

echo "=============================================================================="
echo "Comparing Comprehensive Kokkos Baselines"
echo "=============================================================================="
echo "Baseline: ${BASELINE_VERSION}"
echo "New:      ${NEW_VERSION}"
echo ""

# Check if baseline exists
if [ ! -d "${BASELINE_DIR}" ]; then
    echo "ERROR: Baseline not found at ${BASELINE_DIR}"
    echo "Create it first with: ./create_all_baselines.sh ${BASELINE_VERSION}"
    exit 1
fi

# Create new baseline if it doesn't exist
if [ ! -d "${NEW_DIR}" ]; then
    echo "New baseline not found. Creating it now..."
    ./create_all_baselines.sh "${NEW_VERSION}"
    echo ""
fi

REPORT_FILE="baselines/comprehensive_comparison_${BASELINE_VERSION}_vs_${NEW_VERSION}.txt"

count_matches() {
    local pattern="$1"
    local file="$2"

    if [ ! -f "$file" ]; then
        echo "0"
        return
    fi

    grep -Ec -- "$pattern" "$file" 2>/dev/null || true
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

echo "Generating comprehensive comparison report..."
{
    echo "========================================================================"
    echo "Kokkos Comprehensive Baseline Comparison Report"
    echo "========================================================================"
    echo "Date: $(date)"
    echo "Baseline Version: ${BASELINE_VERSION}"
    echo "New Version:      ${NEW_VERSION}"
    echo ""
    
    # System configuration comparison
    echo "========================================================================"
    echo "System Configuration Comparison"
    echo "========================================================================"
    echo ""
    echo "=== Baseline (${BASELINE_VERSION}) ==="
    if [ -f "${BASELINE_DIR}/system_info.txt" ]; then
        cat "${BASELINE_DIR}/system_info.txt"
    else
        echo "System info not found"
    fi
    echo ""
    echo "=== New (${NEW_VERSION}) ==="
    if [ -f "${NEW_DIR}/system_info.txt" ]; then
        cat "${NEW_DIR}/system_info.txt"
    else
        echo "System info not found"
    fi
    echo ""
    
    # Test categories (must match keys in create_all_baselines.sh)
    declare -A TEST_CATEGORIES=(
        ["AllocationThreshold"]="40KB Allocation Threshold"
        ["SyncBehavior"]="Synchronization Behavior"
        ["ExecutionSpace"]="Execution Spaces"
        ["Views"]="View Operations"
        ["StreamOrderedView"]="Stream Ordered Views"
        ["ViewTraits"]="View Traits, Atomics & MDRange"
        ["UVMandILP"]="UVM, Aliasing & ILP"
        ["CudaReference"]="CUDA Reference Implementations"
    )

    declare -A API_PATTERNS=(
        ["cudaMalloc"]="^cudaMalloc$"
        ["cudaMallocAsync"]="^cudaMallocAsync"
        ["cudaFree"]="^cudaFree$"
        ["cudaFreeAsync"]="^cudaFreeAsync"
        ["cudaDeviceSynchronize"]="^cudaDeviceSynchronize$"
        ["cudaStreamSynchronize"]="^cudaStreamSynchronize$"
        ["cudaMemcpy"]="^cudaMemcpy$"
    )
    
    echo "========================================================================"
    echo "Per-Category Synchronization Comparison"
    echo "========================================================================"
    echo ""
    
    for category in "${!TEST_CATEGORIES[@]}"; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "${category}: ${TEST_CATEGORIES[$category]}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        baseline_sync="${BASELINE_DIR}/${category}_sync.txt"
        new_sync="${NEW_DIR}/${category}_sync.txt"
        
        if [ -f "$baseline_sync" ] && [ -f "$new_sync" ]; then
            # Compare key metrics
            echo "CUDA API Call Counts:"
            echo "─────────────────────────────────────────────────────"
            printf "%-30s | %-10s | %-10s | %-10s\n" "API Call" "Baseline" "New" "Difference"
            echo "─────────────────────────────────────────────────────"
            
            for api in "cudaMalloc" "cudaMallocAsync" "cudaFree" "cudaFreeAsync" "cudaDeviceSynchronize" "cudaStreamSynchronize" "cudaMemcpy"; do
                baseline_count=$(api_call_count "${API_PATTERNS[$api]}" "$baseline_sync")
                new_count=$(api_call_count "${API_PATTERNS[$api]}" "$new_sync")
                
                diff=$((new_count - baseline_count))
                
                if [ $diff -gt 0 ]; then
                    symbol="⬆"
                    status="INCREASED"
                elif [ $diff -lt 0 ]; then
                    symbol="⬇"
                    status="DECREASED"
                else
                    symbol="━"
                    status="UNCHANGED"
                fi
                
                printf "%-30s | %-10s | %-10s | %+d %s\n" "$api" "$baseline_count" "$new_count" "$diff" "$symbol"
            done
            echo ""
            
            # Highlight significant changes
            dev_sync_baseline=$(api_call_count "${API_PATTERNS[cudaDeviceSynchronize]}" "$baseline_sync")
            dev_sync_new=$(api_call_count "${API_PATTERNS[cudaDeviceSynchronize]}" "$new_sync")
            dev_sync_diff=$((dev_sync_new - dev_sync_baseline))
            
            if [ $dev_sync_diff -gt 0 ]; then
                echo "⚠️  WARNING: cudaDeviceSynchronize calls INCREASED by $dev_sync_diff"
                echo "   This may indicate a performance regression in ${category}"
            elif [ $dev_sync_diff -lt 0 ]; then
                echo "✅ IMPROVEMENT: cudaDeviceSynchronize calls DECREASED by ${dev_sync_diff#-}"
                echo "   Performance improvement in ${category}"
            else
                echo "✓ No change in cudaDeviceSynchronize count"
            fi
            echo ""
            
        else
            echo "⚠️  Sync data not available for comparison"
            echo ""
        fi
    done
    
    echo "========================================================================"
    echo "Critical Findings Summary"
    echo "========================================================================"
    echo ""
    
    # Check for critical changes
    critical_changes=0
    
    # Check AllocationThreshold specifically
    if [ -f "${BASELINE_DIR}/AllocationThreshold_sync.txt" ] && [ -f "${NEW_DIR}/AllocationThreshold_sync.txt" ]; then
        baseline_malloc=$(api_call_count "${API_PATTERNS[cudaMalloc]}" "${BASELINE_DIR}/AllocationThreshold_sync.txt")
        new_malloc=$(api_call_count "${API_PATTERNS[cudaMalloc]}" "${NEW_DIR}/AllocationThreshold_sync.txt")
        
        baseline_malloc_async=$(api_call_count "${API_PATTERNS[cudaMallocAsync]}" "${BASELINE_DIR}/AllocationThreshold_sync.txt")
        new_malloc_async=$(api_call_count "${API_PATTERNS[cudaMallocAsync]}" "${NEW_DIR}/AllocationThreshold_sync.txt")
        
        if [ "$new_malloc_async" != "$baseline_malloc_async" ]; then
            echo "📝 ALLOCATION THRESHOLD BEHAVIOR CHANGED"
            echo "   cudaMallocAsync: $baseline_malloc_async → $new_malloc_async"
            echo "   This may indicate the 40KB threshold was modified"
            echo ""
            ((critical_changes+=1))
        fi
    fi
    
    # Check async malloc configuration
    baseline_has_async=$(count_matches "(Kokkos|KOKKOS)_ENABLE_IMPL_CUDA_MALLOC_ASYNC.*ON" "${BASELINE_DIR}/system_info.txt")
    new_has_async=$(count_matches "(Kokkos|KOKKOS)_ENABLE_IMPL_CUDA_MALLOC_ASYNC.*ON" "${NEW_DIR}/system_info.txt")
    
    if [ "$baseline_has_async" != "$new_has_async" ]; then
        echo "⚠️  CONFIGURATION CHANGE: KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC"
        if [ "$new_has_async" = "0" ]; then
            echo "   Baseline: ON → New: OFF"
            echo "   40KB threshold will no longer affect behavior"
        else
            echo "   Baseline: OFF → New: ON"
            echo "   40KB threshold is now active"
        fi
        echo ""
        ((critical_changes+=1))
    fi
    
    # Overall synchronization trend
    total_baseline_syncs=0
    total_new_syncs=0
    
    for category in "${!TEST_CATEGORIES[@]}"; do
        if [ -f "${BASELINE_DIR}/${category}_sync.txt" ] && [ -f "${NEW_DIR}/${category}_sync.txt" ]; then
            baseline_dev_syncs=$(api_call_count "${API_PATTERNS[cudaDeviceSynchronize]}" "${BASELINE_DIR}/${category}_sync.txt")
            baseline_stream_syncs=$(api_call_count "${API_PATTERNS[cudaStreamSynchronize]}" "${BASELINE_DIR}/${category}_sync.txt")
            new_dev_syncs=$(api_call_count "${API_PATTERNS[cudaDeviceSynchronize]}" "${NEW_DIR}/${category}_sync.txt")
            new_stream_syncs=$(api_call_count "${API_PATTERNS[cudaStreamSynchronize]}" "${NEW_DIR}/${category}_sync.txt")
            baseline_syncs=$((baseline_dev_syncs + baseline_stream_syncs))
            new_syncs=$((new_dev_syncs + new_stream_syncs))
            total_baseline_syncs=$((total_baseline_syncs + baseline_syncs))
            total_new_syncs=$((total_new_syncs + new_syncs))
        fi
    done
    
    sync_diff=$((total_new_syncs - total_baseline_syncs))
    
    echo "OVERALL SYNCHRONIZATION TREND:"
    echo "  Baseline total: $total_baseline_syncs"
    echo "  New total:      $total_new_syncs"
    echo "  Difference:     $sync_diff"
    echo ""
    
    if [ $sync_diff -gt 0 ]; then
        if [ $total_baseline_syncs -gt 0 ]; then
            percent_change=$(awk -v diff="$sync_diff" -v baseline="$total_baseline_syncs" 'BEGIN { printf "%.1f", (diff * 100.0) / baseline }')
            echo "⚠️  OVERALL REGRESSION: +$percent_change% more synchronizations"
        else
            echo "⚠️  OVERALL REGRESSION: synchronizations appeared where baseline had none"
        fi
        ((critical_changes+=1))
    elif [ $sync_diff -lt 0 ]; then
        if [ $total_baseline_syncs -gt 0 ]; then
            percent_change=$(awk -v diff="$sync_diff" -v baseline="$total_baseline_syncs" 'BEGIN { printf "%.1f", (diff * -100.0) / baseline }')
            echo "✅ OVERALL IMPROVEMENT: -$percent_change% fewer synchronizations"
        else
            echo "✅ OVERALL IMPROVEMENT: fewer synchronizations"
        fi
    else
        echo "✓ No overall change in synchronization count"
    fi
    echo ""
    
    if [ $critical_changes -eq 0 ]; then
        echo "✅ NO CRITICAL CHANGES DETECTED"
        echo "   Behavior appears consistent between versions"
    else
        echo "⚠️  $critical_changes CRITICAL CHANGE(S) DETECTED"
        echo "   Review detailed comparison above"
    fi
    echo ""
    
    echo "========================================================================"
    echo "Next Steps"
    echo "========================================================================"
    echo ""
    echo "1. Review category-specific changes above"
    echo ""
    echo "2. Open profiles for visual comparison:"
    echo "   nsys-ui ${BASELINE_DIR}/AllocationThreshold_baseline.nsys-rep &"
    echo "   nsys-ui ${NEW_DIR}/AllocationThreshold_baseline.nsys-rep &"
    echo ""
    echo "3. Compare timelines:"
    echo "   diff ${BASELINE_DIR}/AllocationThreshold_timeline.txt \\"
    echo "        ${NEW_DIR}/AllocationThreshold_timeline.txt"
    echo ""
    echo "4. Check test outputs for failures:"
    echo "   diff ${BASELINE_DIR}/AllocationThreshold_test_output.txt \\"
    echo "        ${NEW_DIR}/AllocationThreshold_test_output.txt"
    echo ""
    echo "5. If significant changes found, update documentation:"
    echo "   - KOKKOS_${NEW_VERSION//./_}_SYNCHRONIZATION_VERIFICATION.md"
    echo "   - NvidiaKokkosPrimer.md"
    echo ""
    
} | tee "${REPORT_FILE}"

echo ""
echo "=============================================================================="
echo "Comparison Complete!"
echo "=============================================================================="
echo "Report saved to: ${REPORT_FILE}"
echo ""
echo "Summary saved to both versions' directories:"
cp "${REPORT_FILE}" "${BASELINE_DIR}/comparison_with_${NEW_VERSION}.txt" 2>/dev/null || true
cp "${REPORT_FILE}" "${NEW_DIR}/comparison_with_${BASELINE_VERSION}.txt" 2>/dev/null || true
echo "  - ${BASELINE_DIR}/comparison_with_${NEW_VERSION}.txt"
echo "  - ${NEW_DIR}/comparison_with_${BASELINE_VERSION}.txt"
echo ""
