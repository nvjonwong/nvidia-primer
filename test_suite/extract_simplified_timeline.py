#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extract simplified timeline showing only key events for regression testing.
Output format is easy to diff between Kokkos versions.
"""

import sqlite3
import sys
import argparse
from pathlib import Path


IMPORTANT_CUDA_APIS = (
    "cudaMalloc", "cudaFree", "cudaMallocAsync", "cudaFreeAsync",
    "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaLaunchKernel",
    "cudaMemcpy", "cudaMemcpyAsync", "cudaMemset", "cudaMemsetAsync",
    "cudaHostAlloc", "cudaFreeHost",
)


def extract_events(cursor):
    """Extract CUDA API calls, NVTX events, and kernel launches."""
    events = []
    
    # Get name mapping
    names = {}
    try:
        cursor.execute("SELECT id, value FROM StringIds")
        for row in cursor.fetchall():
            names[row[0]] = row[1]
    except sqlite3.OperationalError:
        pass
    
    # Extract CUDA API calls (focus on sync and memory operations)
    try:
        cursor.execute("""
            SELECT start, end, nameId, correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            ORDER BY start
        """)
        
        for row in cursor.fetchall():
            start, end, name_id, corr_id = row
            api_name = names.get(name_id, f'API_{name_id}')
            
            # Filter to important APIs only
            if any(api in api_name for api in IMPORTANT_CUDA_APIS):
                events.append({
                    'type': 'CUDA_API',
                    'name': api_name,
                    'start': start,
                    'end': end,
                    'duration': end - start if end else 0
                })
    except sqlite3.OperationalError:
        pass
    
    # Extract NVTX events
    try:
        cursor.execute("""
            SELECT start, end, text, eventType
            FROM NVTX_EVENTS
            ORDER BY start
        """)
        
        for row in cursor.fetchall():
            start, end, text, event_type = row
            if text:  # Only include named events
                events.append({
                    'type': 'NVTX',
                    'name': text,
                    'start': start,
                    'end': end if end else start,
                    'duration': (end - start) if end else 0
                })
    except sqlite3.OperationalError:
        pass
    
    # Extract kernel launches
    try:
        cursor.execute("""
            SELECT start, end, demangledName, correlationId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY start
        """)
        
        for row in cursor.fetchall():
            start, end, name_id, corr_id = row
            
            # Get kernel name
            if isinstance(name_id, int):
                kernel_name = names.get(name_id, f'Kernel_{name_id}')
            else:
                kernel_name = name_id or f'Kernel_{corr_id}'
            
            # Simplify kernel name (remove template arguments)
            if '(' in kernel_name:
                kernel_name = kernel_name.split('(')[0]
            if len(kernel_name) > 80:
                kernel_name = kernel_name[:77] + '...'
            
            events.append({
                'type': 'KERNEL',
                'name': kernel_name,
                'start': start,
                'end': end,
                'duration': end - start if end else 0
            })
    except sqlite3.OperationalError:
        pass
    
    return events


def format_time_ns(ns):
    """Format nanoseconds for output."""
    return f"{ns:>15d}"


def print_simplified_timeline(events, show_durations=False):
    """Print events in simple, diff-friendly format."""
    # Sort by start time
    events.sort(key=lambda x: x['start'])
    
    if not events:
        print("# No events found")
        return
    
    # Get earliest event for relative timing
    start_time = events[0]['start']
    
    print("# Simplified Timeline - Easy to diff between Kokkos versions")
    print("# Format: RELATIVE_TIME_NS | TYPE | EVENT_NAME")
    if show_durations:
        print("# (with durations)")
    print("")
    
    for event in events:
        rel_time = event['start'] - start_time
        
        if show_durations:
            print(f"{format_time_ns(rel_time)} | {event['type']:<10} | {event['name']} | duration={event['duration']}ns")
        else:
            print(f"{format_time_ns(rel_time)} | {event['type']:<10} | {event['name']}")


def count_sync_operations(events):
    """Count synchronization operations for summary."""
    counts = {
        'cudaDeviceSynchronize': 0,
        'cudaStreamSynchronize': 0,
        'cudaMallocAsync': 0,
        'cudaMalloc': 0,
        'cudaFreeAsync': 0,
        'cudaFree': 0,
        'cudaLaunchKernel': 0,
        'kernels': 0
    }
    
    for event in events:
        name = event['name']
        if 'cudaDeviceSynchronize' in name:
            counts['cudaDeviceSynchronize'] += 1
        elif 'cudaStreamSynchronize' in name:
            counts['cudaStreamSynchronize'] += 1
        elif 'cudaMallocAsync' in name:
            counts['cudaMallocAsync'] += 1
        elif 'cudaMalloc' in name and 'Async' not in name:
            counts['cudaMalloc'] += 1
        elif 'cudaFreeAsync' in name:
            counts['cudaFreeAsync'] += 1
        elif 'cudaFree' in name and 'Async' not in name:
            counts['cudaFree'] += 1
        elif 'cudaLaunchKernel' in name:
            counts['cudaLaunchKernel'] += 1
        elif event['type'] == 'KERNEL':
            counts['kernels'] += 1
    
    return counts


def main():
    parser = argparse.ArgumentParser(
        description='Extract simplified timeline for regression testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Extract simplified timeline
  ./extract_simplified_timeline.py profile.sqlite
  
  # Save to file for baseline
  ./extract_simplified_timeline.py profile.sqlite -o baseline.txt
  
  # Include durations (less stable for diffs)
  ./extract_simplified_timeline.py profile.sqlite --durations
  
  # Show summary statistics
  ./extract_simplified_timeline.py profile.sqlite --summary
        '''
    )
    
    parser.add_argument('sqlite_file', help='SQLite file from nsys export')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--durations', action='store_true',
                        help='Include event durations (makes diff less stable)')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary statistics')
    
    args = parser.parse_args()
    
    if not Path(args.sqlite_file).exists():
        print(f"Error: File not found: {args.sqlite_file}", file=sys.stderr)
        sys.exit(1)
    
    # Connect to database
    conn = sqlite3.connect(args.sqlite_file)
    cursor = conn.cursor()
    
    # Extract events
    events = extract_events(cursor)
    conn.close()
    
    # Output
    if args.output:
        sys.stdout = open(args.output, 'w')
    
    if args.summary:
        counts = count_sync_operations(events)
        print("# Summary Statistics")
        print(f"# cudaDeviceSynchronize:  {counts['cudaDeviceSynchronize']}")
        print(f"# cudaStreamSynchronize:  {counts['cudaStreamSynchronize']}")
        print(f"# cudaMallocAsync:        {counts['cudaMallocAsync']}")
        print(f"# cudaMalloc:             {counts['cudaMalloc']}")
        print(f"# cudaFreeAsync:          {counts['cudaFreeAsync']}")
        print(f"# cudaFree:               {counts['cudaFree']}")
        print(f"# cudaLaunchKernel:       {counts['cudaLaunchKernel']}")
        print(f"# Kernels executed:       {counts['kernels']}")
        print("")
    
    print_simplified_timeline(events, show_durations=args.durations)
    
    if args.output:
        print(f"Simplified timeline saved to: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
