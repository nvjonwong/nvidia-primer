#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extract chronological timeline of CUDA API calls and NVTX events from nsys SQLite export.
Shows all events in order with timestamps.
"""

import sqlite3
import sys
import argparse
from pathlib import Path


def get_tables(cursor):
    """Get all tables in the database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row[0] for row in cursor.fetchall()]


def extract_cuda_api_events(cursor):
    """Extract CUDA API calls with start/end times."""
    events = []
    
    # Get name mapping first
    names = {}
    try:
        cursor.execute("SELECT id, value FROM StringIds")
        for row in cursor.fetchall():
            names[row[0]] = row[1]
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("""
            SELECT start, end, nameId, correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            ORDER BY start
        """)
        
        for row in cursor.fetchall():
            start, end, name_id, corr_id = row
            events.append({
                'type': 'CUDA_API',
                'name': names.get(name_id, f'API_{name_id}'),
                'start': start,
                'end': end,
                'duration': end - start if end else 0,
                'correlation': corr_id
            })
    except sqlite3.OperationalError as e:
        print(f"Note: Could not query CUDA API events: {e}", file=sys.stderr)
    
    return events


def extract_nvtx_events(cursor):
    """Extract NVTX markers and ranges."""
    events = []
    
    # Try NVTX_EVENTS table
    try:
        cursor.execute("""
            SELECT start, end, text, eventType
            FROM NVTX_EVENTS
            ORDER BY start
        """)
        
        for row in cursor.fetchall():
            start, end, text, event_type = row
            events.append({
                'type': 'NVTX',
                'name': text or f'Event_{event_type}',
                'start': start,
                'end': end if end else start,
                'duration': (end - start) if end else 0,
                'event_type': event_type
            })
    except sqlite3.OperationalError:
        pass
    
    return events


def extract_kernel_events(cursor):
    """Extract GPU kernel launches."""
    events = []
    
    # Get name mapping first
    names = {}
    try:
        cursor.execute("SELECT id, value FROM StringIds")
        for row in cursor.fetchall():
            names[row[0]] = row[1]
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("""
            SELECT start, end, demangledName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY start
        """)
        
        for row in cursor.fetchall():
            start, end, name_id, corr_id, gx, gy, gz, bx, by, bz = row
            # name_id might be a string directly or an ID
            if isinstance(name_id, int):
                name = names.get(name_id, f'Kernel_{name_id}')
            else:
                name = name_id or f'Kernel_{corr_id}'
            
            # Truncate very long kernel names
            if len(name) > 80:
                name = name[:77] + '...'
            
            events.append({
                'type': 'KERNEL',
                'name': name,
                'start': start,
                'end': end,
                'duration': end - start if end else 0,
                'correlation': corr_id,
                'grid': f'({gx},{gy},{gz})',
                'block': f'({bx},{by},{bz})'
            })
    except sqlite3.OperationalError as e:
        print(f"Note: Could not query kernel events: {e}", file=sys.stderr)
    
    return events


def format_time(ns):
    """Format nanoseconds in a human-readable way."""
    if ns < 1000:
        return f"{ns}ns"
    elif ns < 1000000:
        return f"{ns/1000:.2f}us"
    elif ns < 1000000000:
        return f"{ns/1000000:.2f}ms"
    else:
        return f"{ns/1000000000:.3f}s"


def print_timeline(events, show_correlation=False, filter_type=None):
    """Print events in chronological order."""
    # Filter if needed
    if filter_type:
        events = [e for e in events if e['type'] == filter_type]
    
    # Sort by start time
    events.sort(key=lambda x: x['start'])
    
    if not events:
        print("No events found.")
        return
    
    # Get earliest event for relative timing
    start_time = events[0]['start']
    
    print(f"{'Time':>12} | {'Rel.Time':>12} | {'Duration':>12} | {'Type':>10} | Event")
    print("-" * 120)
    
    for event in events:
        rel_time = event['start'] - start_time
        
        line = f"{format_time(event['start']):>12} | {format_time(rel_time):>12} | {format_time(event['duration']):>12} | {event['type']:>10} | {event['name']}"
        
        # Add extra info for kernels
        if event['type'] == 'KERNEL' and 'grid' in event:
            line += f" [Grid{event['grid']} Block{event['block']}]"
        
        if show_correlation and 'correlation' in event:
            line += f" (corr:{event['correlation']})"
        
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description='Extract chronological timeline from nsys SQLite export',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Show all events
  ./extract_timeline.py test.sqlite
  
  # Show only CUDA API calls
  ./extract_timeline.py test.sqlite --type CUDA_API
  
  # Show only NVTX events
  ./extract_timeline.py test.sqlite --type NVTX
  
  # Show with correlation IDs
  ./extract_timeline.py test.sqlite --correlation
        '''
    )
    
    parser.add_argument('sqlite_file', help='SQLite file from nsys export')
    parser.add_argument('--type', choices=['CUDA_API', 'NVTX', 'KERNEL'], 
                        help='Filter to specific event type')
    parser.add_argument('--correlation', action='store_true',
                        help='Show correlation IDs')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    if not Path(args.sqlite_file).exists():
        print(f"Error: File not found: {args.sqlite_file}", file=sys.stderr)
        sys.exit(1)
    
    # Connect to database
    conn = sqlite3.connect(args.sqlite_file)
    cursor = conn.cursor()
    
    # Show available tables for debugging
    tables = get_tables(cursor)
    print(f"Database contains {len(tables)} tables", file=sys.stderr)
    print(f"Extracting events from: {args.sqlite_file}", file=sys.stderr)
    print(file=sys.stderr)
    
    # Extract all event types
    all_events = []
    all_events.extend(extract_cuda_api_events(cursor))
    all_events.extend(extract_nvtx_events(cursor))
    all_events.extend(extract_kernel_events(cursor))
    
    conn.close()
    
    # Output
    if args.output:
        sys.stdout = open(args.output, 'w')
    
    print_timeline(all_events, show_correlation=args.correlation, filter_type=args.type)
    
    # Summary
    if not args.output:
        print(file=sys.stderr)
        print(f"Total events: {len(all_events)}", file=sys.stderr)
        event_counts = {}
        for e in all_events:
            event_counts[e['type']] = event_counts.get(e['type'], 0) + 1
        for etype, count in sorted(event_counts.items()):
            print(f"  {etype}: {count}", file=sys.stderr)


if __name__ == '__main__':
    main()
