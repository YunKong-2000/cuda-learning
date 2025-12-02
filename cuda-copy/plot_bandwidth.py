#!/usr/bin/env python3
"""
Parse ncu output from run_copy.sh log file and plot bandwidth vs data size.
"""

import re
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file):
    """
    Parse the log file to extract:
    - Data size (number of elements * sizeof(float))
    - Block size (block_dim)
    - Loop unroll times (for loop_unroll kernel)
    - DRAM bytes read (unit varies: byte, Kbyte, Mbyte, Gbyte)
    - DRAM bytes write (unit varies: byte, Kbyte, Mbyte, Gbyte)
    - GPU time duration (unit varies: us, ms)
    
    All units are converted to bytes and seconds respectively before calculating bandwidth.
    
    Returns a tuple: (results, kernel_type)
    where kernel_type is 'baseline' or 'loop_unroll'
    """
    results = []
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Detect kernel type or compare mode
    kernel_type = 'baseline'  # default
    is_compare_mode = False
    
    # Check for compare mode
    # Updated to detect compare mode with vectorize_unroll
    # Check for compare mode indicators: multiple kernel runs or compare message
    if ('Comparing all kernels' in content or 'Comparing all three kernels' in content or 
        ('>>> Running baseline kernel...' in content and '>>> Running vectorize kernel...' in content) or
        ('>>> Running vectorize_unroll kernel...' in content)):
        is_compare_mode = True
        kernel_type = 'compare'
    elif 'Using kernel: loop_unroll' in content or 'Testing with loop_unroll_times' in content:
        kernel_type = 'loop_unroll'
    elif 'Using kernel: vectorize_unroll' in content:
        kernel_type = 'vectorize_unroll'
    elif 'Using kernel: vectorize' in content:
        kernel_type = 'vectorize'
    
    # Split content by test sections
    # Pattern: "Testing with 2^n = num_elements elements"
    test_pattern = r'Testing with 2\^(\d+) = (\d+) elements'
    
    # Pattern for block_dim: "Testing with block_dim = X threads per block"
    block_dim_pattern = r'Testing with block_dim = (\d+) threads per block'
    
    # Pattern for loop_unroll_times: "Testing with loop_unroll_times = X"
    loop_unroll_pattern = r'Testing with loop_unroll_times = (\d+)'
    
    # Pattern for kernel name: "Kernel: baseline/vectorize/loop_unroll"
    kernel_name_pattern = r'Kernel:\s+(\w+)'
    
    # Find all test sections
    test_matches = list(re.finditer(test_pattern, content))
    
    for i, test_match in enumerate(test_matches):
        start_pos = test_match.start()
        # Find the end of this test section (start of next test or end of file)
        if i + 1 < len(test_matches):
            end_pos = test_matches[i + 1].start()
        else:
            end_pos = len(content)
        
        test_section = content[start_pos:end_pos]
        
        # Extract test parameters
        n = int(test_match.group(1))
        num_elements = int(test_match.group(2))
        
        # Calculate data size in bytes (float = 4 bytes)
        data_size_bytes = num_elements * 4
        
        # Find all block_dim tests within this data size test section
        block_dim_matches = list(re.finditer(block_dim_pattern, test_section))
        
        if not block_dim_matches:
            print(f"Warning: No block_dim found for 2^{n} = {num_elements} elements")
            continue
        
        # Process each block_dim test
        for j, block_dim_match in enumerate(block_dim_matches):
            block_dim_start = block_dim_match.start()
            # Find the end of this block_dim test (start of next block_dim or end of test section)
            if j + 1 < len(block_dim_matches):
                block_dim_end = block_dim_matches[j + 1].start()
            else:
                block_dim_end = len(test_section)
            
            block_dim_section = test_section[block_dim_start:block_dim_end]
            
            # Extract block_dim
            block_size = int(block_dim_match.group(1))
            
            # Extract kernel name (for compare mode or single kernel mode)
            kernel_name = None
            if is_compare_mode:
                kernel_match = re.search(kernel_name_pattern, block_dim_section)
                if kernel_match:
                    kernel_name = kernel_match.group(1)
                else:
                    print(f"Warning: No kernel name found for 2^{n} = {num_elements} elements, block_dim = {block_size}")
                    continue
            else:
                # For single kernel mode, use detected kernel_type
                if kernel_type == 'baseline':
                    kernel_name = 'baseline'
                elif kernel_type == 'vectorize':
                    kernel_name = 'vectorize'
                elif kernel_type == 'vectorize_unroll':
                    kernel_name = 'vectorize_unroll'
                elif kernel_type == 'loop_unroll':
                    kernel_name = 'loop_unroll'
            
            # Extract loop_unroll_times if this is a loop_unroll kernel
            loop_unroll_times = None
            if kernel_name == 'loop_unroll':
                loop_unroll_match = re.search(loop_unroll_pattern, block_dim_section)
                if loop_unroll_match:
                    loop_unroll_times = int(loop_unroll_match.group(1))
                else:
                    print(f"Warning: No loop_unroll_times found for 2^{n} = {num_elements} elements, block_dim = {block_size}")
                    continue
            
            # Look for ncu output section
            if '--- ncu Raw Measurement Results ---' not in block_dim_section:
                if kernel_name == 'loop_unroll':
                    print(f"Warning: No ncu output found for 2^{n} = {num_elements} elements, block_dim = {block_size}, kernel = {kernel_name}, loop_unroll_times = {loop_unroll_times}")
                else:
                    print(f"Warning: No ncu output found for 2^{n} = {num_elements} elements, block_dim = {block_size}, kernel = {kernel_name}")
                continue
            
            # Extract metrics from ncu output
            # Pattern for metric lines: "dram__bytes_read.sum         Gbyte    4.30    4.30    4.30"
            # The format is: metric_name, unit, min, max, average
            metrics_pattern = r'(dram__bytes_read\.sum|dram__bytes_write\.sum|gpu__time_duration\.sum)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
            
            metrics = {}
            for metric_match in re.finditer(metrics_pattern, block_dim_section):
                metric_name = metric_match.group(1)
                metric_unit = metric_match.group(2)
                # Use Average value (group 5, the last column)
                metric_value = float(metric_match.group(5))
                
                metrics[metric_name] = {
                    'value': metric_value,
                    'unit': metric_unit
                }
            
            # Check if we got all required metrics
            required_metrics = ['dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum']
            if not all(m in metrics for m in required_metrics):
                if kernel_name == 'loop_unroll':
                    print(f"Warning: Missing metrics for 2^{n} = {num_elements} elements, block_dim = {block_size}, kernel = {kernel_name}, loop_unroll_times = {loop_unroll_times}")
                else:
                    print(f"Warning: Missing metrics for 2^{n} = {num_elements} elements, block_dim = {block_size}, kernel = {kernel_name}")
                print(f"  Found metrics: {list(metrics.keys())}")
                continue
            
            # Convert units to bytes and seconds
            # Note: ncu uses decimal units (SI standard)
            # Memory units: byte, Kbyte, Mbyte, Gbyte
            def convert_bytes_to_bytes(value, unit):
                """Convert memory units to bytes."""
                unit_lower = unit.lower()
                if unit_lower == 'byte':
                    return value
                elif unit_lower == 'kbyte':
                    return value * 1e3
                elif unit_lower == 'mbyte':
                    return value * 1e6
                elif unit_lower == 'gbyte':
                    return value * 1e9
                else:
                    raise ValueError(f"Unknown memory unit: {unit}")
            
            # Time units: us, ms
            def convert_time_to_seconds(value, unit):
                """Convert time units to seconds."""
                unit_lower = unit.lower()
                if unit_lower == 'us' or unit_lower == 'usecond':
                    return value * 1e-6
                elif unit_lower == 'ms' or unit_lower == 'msecond':
                    return value * 1e-3
                else:
                    raise ValueError(f"Unknown time unit: {unit}")
            
            # Convert dram bytes
            bytes_read = convert_bytes_to_bytes(
                metrics['dram__bytes_read.sum']['value'],
                metrics['dram__bytes_read.sum']['unit']
            )
            bytes_write = convert_bytes_to_bytes(
                metrics['dram__bytes_write.sum']['value'],
                metrics['dram__bytes_write.sum']['unit']
            )
            
            # Convert time duration
            time_seconds = convert_time_to_seconds(
                metrics['gpu__time_duration.sum']['value'],
                metrics['gpu__time_duration.sum']['unit']
            )
            
            # Calculate bandwidth: (bytes_read + bytes_write) / time
            total_bytes = bytes_read + bytes_write
            bandwidth_bytes_per_sec = total_bytes / time_seconds
            bandwidth_gb_per_sec = bandwidth_bytes_per_sec / 1e9
            
            result = {
                'n': n,
                'num_elements': num_elements,
                'data_size_bytes': data_size_bytes,
                'block_size': block_size,
                'kernel_name': kernel_name,
                'bytes_read': bytes_read,
                'bytes_write': bytes_write,
                'time_seconds': time_seconds,
                'bandwidth_bytes_per_sec': bandwidth_bytes_per_sec,
                'bandwidth_gb_per_sec': bandwidth_gb_per_sec
            }
            
            if loop_unroll_times is not None:
                result['loop_unroll_times'] = loop_unroll_times
            
            results.append(result)
    
    return results, kernel_type

def plot_bandwidth(results, kernel_type='baseline', output_file='bandwidth_plot.png'):
    """
    Plot bandwidth vs data size with log2 scale on x-axis.
    For compare mode: different kernel_name values are shown as different lines.
    For baseline kernel: different block_size values are shown as different lines.
    For loop_unroll kernel: different loop_unroll_times values are shown as different lines.
    """
    if not results:
        print("Error: No data to plot")
        return
    
    # Group results based on kernel type
    grouped_results = defaultdict(list)
    
    if kernel_type == 'compare':
        # Compare mode: group by kernel_name
        for r in results:
            if 'kernel_name' in r:
                grouped_results[r['kernel_name']].append(r)
    elif kernel_type == 'loop_unroll':
        # Loop unroll mode: group by loop_unroll_times
        for r in results:
            if 'loop_unroll_times' in r:
                grouped_results[r['loop_unroll_times']].append(r)
    else:
        # Baseline, vectorize, or vectorize_unroll mode: group by block_size
        for r in results:
            grouped_results[r['block_size']].append(r)
    
    # Sort group keys for consistent ordering
    group_keys = sorted(grouped_results.keys())
    
    if not group_keys:
        print("Error: No valid grouping found")
        return
    
    # Create figure
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    # Define colors and markers for different groups
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_keys)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot a line for each group
    for idx, group_key_value in enumerate(group_keys):
        group_results = grouped_results[group_key_value]
        # Sort by data size for this group
        group_results.sort(key=lambda x: x['data_size_bytes'])
        
        data_sizes = [r['data_size_bytes'] for r in group_results]
        bandwidths = [r['bandwidth_bytes_per_sec'] for r in group_results]
        
        # Plot with different color and marker for each group
        marker = markers[idx % len(markers)]
        color = colors[idx]
        
        if kernel_type == 'compare':
            # In compare mode, group_key_value is kernel_name
            label = f'{group_key_value}'
        elif kernel_type == 'loop_unroll':
            label = f'loop_unroll_times = {group_key_value}'
        else:
            label = f'block_size = {group_key_value}'
        
        plt.plot(data_sizes, bandwidths, marker=marker, linestyle='-', 
                linewidth=2, markersize=8, color=color, label=label)
    
    # Set x-axis to log2 scale
    plt.xscale('log', base=2)
    
    # Format x-axis labels
    ax.set_xlabel('Data Size (bytes, log₂ scale)', fontsize=12)
    ax.set_ylabel('Bandwidth (bytes/s)', fontsize=12)
    
    # Set title based on kernel type
    if kernel_type == 'compare':
        ax.set_title('DRAM Bandwidth vs Data Size (Kernel Comparison)', fontsize=14, fontweight='bold')
    elif kernel_type == 'loop_unroll':
        ax.set_title('DRAM Bandwidth vs Data Size (by Loop Unroll Times)', fontsize=14, fontweight='bold')
    elif kernel_type == 'vectorize' or kernel_type == 'vectorize_unroll':
        ax.set_title('DRAM Bandwidth vs Data Size (by Block Size)', fontsize=14, fontweight='bold')
    else:
        ax.set_title('DRAM Bandwidth vs Data Size (by Block Size)', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    # Format x-axis ticks to show powers of 2
    all_data_sizes = [r['data_size_bytes'] for r in results]
    max_size = max(all_data_sizes)
    min_size = min(all_data_sizes)
    min_power = int(np.floor(np.log2(min_size)))
    max_power = int(np.ceil(np.log2(max_size)))
    
    # Create tick positions (powers of 2)
    tick_positions = [2**p for p in range(min_power, max_power + 1)]
    # Format labels: show 2^n format
    tick_labels = [f'2^{p}' for p in range(min_power, max_power + 1)]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    
    # Format y-axis to show GB/s
    y_ticks = ax.get_yticks()
    y_labels = [f'{y/1e9:.2f} GB/s' for y in y_ticks]
    ax.set_yticklabels(y_labels)
    
    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Optionally add value annotations (can be commented out if too cluttered)
    # for r in results:
    #     plt.annotate(
    #         f"{r['bandwidth_gb_per_sec']:.2f} GB/s",
    #         (r['data_size_bytes'], r['bandwidth_bytes_per_sec']),
    #         textcoords="offset points",
    #         xytext=(0, 10),
    #         ha='center',
    #         fontsize=8,
    #         alpha=0.6
    #     )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Print summary grouped by group_key
    print("\nSummary:")
    print("=" * 100)
    for group_key_value in group_keys:
        group_results = grouped_results[group_key_value]
        group_results.sort(key=lambda x: x['data_size_bytes'])
        
        if kernel_type == 'compare':
            print(f"\nKernel = {group_key_value}:")
        elif kernel_type == 'loop_unroll':
            print(f"\nLoop Unroll Times = {group_key_value}:")
        else:
            print(f"\nBlock Size = {group_key_value}:")
        
        print("-" * 100)
        print(f"{'Data Size (bytes)':<20} {'Bandwidth (GB/s)':<20} {'Time (ms)':<15}")
        print("-" * 100)
        for r in group_results:
            print(f"{r['data_size_bytes']:<20} {r['bandwidth_gb_per_sec']:<20.2f} {r['time_seconds']*1000:<15.2f}")
    print("=" * 100)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_bandwidth.py <log_file> [output_image]")
        print("Example: python3 plot_bandwidth.py run_copy.log bandwidth.png")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'bandwidth_plot.png'
    
    print(f"Parsing log file: {log_file}")
    results, kernel_type = parse_log_file(log_file)
    
    if not results:
        print("Error: No valid data found in log file")
        sys.exit(1)
    
    print(f"Found {len(results)} valid test results")
    print(f"Detected kernel type: {kernel_type}")
    plot_bandwidth(results, kernel_type, output_file)

if __name__ == '__main__':
    main()

