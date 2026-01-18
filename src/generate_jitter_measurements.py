"""
Generate jitter measurements for PASync paper figures.

This script produces:
1. PTS interval distribution (rendering jitter baseline)
2. Decoding time distribution (decoding baseline)  
3. Three-stage jitter CDF (rendering, network, decoding)

Data columns (from load_detailed_framerate_log):
    0 'render_index', 'frame_index', 'frame_type', 'size', 'loss_type',
    5 'client_receive_ts', 'receive_and_unpack', 'decoder_outside_queue', 
      'decoder_insided_queue', 'decode', 'render_queue', 'display',
    12 'proxy_recv_ts', 'proxy_recv_time', 'proxy_send_delay', 'send_time',
    16 'net_time', 'proc_time', 'tot_time',
    19 'basic_net_ts', 'ul_jitter', 'dl_jitter'
    33 pts, ets, dts, sts, ...
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

import load_data

# Configuration
FRAME_INTERVAL = 16.666667  # ms (60 FPS)
JITTER_THRESHOLD = 2  # ms (τ in the paper)
OUTPUT_DIR = "measurements"


def collect_all_traces(root_path, max_files=None):
    """Collect all valid trace files from the data directory."""
    trace_files = []
    for data_folder in os.listdir(root_path):
        if not data_folder.startswith("2024-06-14"):
            continue
        data_path = os.path.join(root_path, data_folder)
        if not os.path.isdir(data_path):
            continue
        for session_folder in os.listdir(data_path):
            session_path = os.path.join(data_path, session_folder)
            if not os.path.isdir(session_path):
                continue
            for file_name in os.listdir(session_path):
                if file_name.endswith(".csv"):
                    trace_files.append(os.path.join(session_path, file_name))
    
    if max_files is not None:
        trace_files = trace_files[:max_files]
    
    return trace_files


def compute_jitter_stats(file_path, frame_interval=FRAME_INTERVAL):
    """
    Compute jitter statistics for a single trace file.
    
    Returns:
        dict with keys:
            - pts_intervals: PTS interval deviations from nominal (rendering jitter)
            - decode_times: actual decoding times
            - network_jitters: network jitter (dl_jitter)
            - rendering_jitters: rendering jitter (PTS deviation from expected)
            - decoding_jitters: decoding jitter (actual - EWMA baseline)
    """
    try:
        data, info = load_data.load_detailed_framerate_log(
            file_path, start_idx=0, len_limit=60 * 60 * 20
        )
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    if data is None or data.shape[0] < 1000:
        return None
    
    # Filter valid frames (loss_type == 0)
    valid_idx = np.where(data[:, 4] == 0)[0]
    if len(valid_idx) < 500:
        return None
    data = data[valid_idx]
    
    # 1. PTS intervals (rendering baseline)
    pts = data[:, 33]  # Presentation Timestamp
    pts_intervals = np.diff(pts)
    # Filter out scene changes or large gaps
    pts_intervals = pts_intervals[(pts_intervals > 0) & (pts_intervals < 100)]
    pts_deviations = pts_intervals - frame_interval  # deviation from nominal
    
    # 2. Decoding times
    decode_times = data[:, 9]  # decode column
    decode_times = decode_times[decode_times > 0]  # filter invalid
    
    # 3. Network jitter (already computed in data)
    network_jitters = data[1:, 21]  # dl_jitter column
    
    # 4. Rendering jitter: PTS deviation from expected (using anchor extrapolation)
    # Simple version: deviation from previous frame's PTS + nominal interval
    expected_pts = pts[:-1] + frame_interval
    actual_pts = pts[1:]
    rendering_jitters = actual_pts - expected_pts
    
    # 5. Decoding jitter: actual - EWMA baseline
    # Compute EWMA baseline
    alpha = 0.99
    ewma_baseline = np.zeros(len(decode_times))
    ewma_baseline[0] = decode_times[0]
    for i in range(1, len(decode_times)):
        ewma_baseline[i] = alpha * ewma_baseline[i-1] + (1 - alpha) * decode_times[i-1]
    decoding_jitters = decode_times - ewma_baseline
    
    return {
        'pts_intervals': pts_intervals,
        'pts_deviations': pts_deviations,
        'decode_times': decode_times,
        'network_jitters': network_jitters,
        'rendering_jitters': rendering_jitters,
        'decoding_jitters': decoding_jitters,
    }


def aggregate_stats(trace_files):
    """Aggregate statistics across all trace files."""
    all_pts_intervals = []
    all_pts_deviations = []
    all_decode_times = []
    all_network_jitters = []
    all_rendering_jitters = []
    all_decoding_jitters = []
    
    for file_path in tqdm(trace_files, desc="Processing traces"):
        stats = compute_jitter_stats(file_path)
        if stats is None:
            continue
        
        all_pts_intervals.extend(stats['pts_intervals'])
        all_pts_deviations.extend(stats['pts_deviations'])
        all_decode_times.extend(stats['decode_times'])
        all_network_jitters.extend(stats['network_jitters'])
        all_rendering_jitters.extend(stats['rendering_jitters'])
        all_decoding_jitters.extend(stats['decoding_jitters'])
    
    return {
        'pts_intervals': np.array(all_pts_intervals),
        'pts_deviations': np.array(all_pts_deviations),
        'decode_times': np.array(all_decode_times),
        'network_jitters': np.array(all_network_jitters),
        'rendering_jitters': np.array(all_rendering_jitters),
        'decoding_jitters': np.array(all_decoding_jitters),
    }


def plot_pts_distribution(pts_intervals, output_path):
    """Plot PTS interval distribution."""
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Histogram
    ax.hist(pts_intervals, bins=100, density=True, alpha=0.7, color='steelblue', 
            edgecolor='black', linewidth=0.5)
    
    # Add vertical line at nominal interval
    ax.axvline(x=FRAME_INTERVAL, color='red', linestyle='--', linewidth=1.5, 
               label=f'Nominal ({FRAME_INTERVAL:.1f} ms)')
    
    ax.set_xlabel('PTS Interval (ms)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_xlim(0, 50)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nPTS Interval Statistics:")
    print(f"  Mean: {np.mean(pts_intervals):.2f} ms")
    print(f"  Std:  {np.std(pts_intervals):.2f} ms")
    print(f"  Median: {np.median(pts_intervals):.2f} ms")


def plot_decode_time_distribution(decode_times, output_path):
    """Plot decoding time distribution."""
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Histogram
    ax.hist(decode_times, bins=100, density=True, alpha=0.7, color='forestgreen',
            edgecolor='black', linewidth=0.5)
    
    # Add vertical line at mean
    mean_decode = np.mean(decode_times)
    ax.axvline(x=mean_decode, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean ({mean_decode:.1f} ms)')
    
    ax.set_xlabel('Decoding Time (ms)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_xlim(0, min(20, np.percentile(decode_times, 99)))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nDecoding Time Statistics:")
    print(f"  Mean: {np.mean(decode_times):.2f} ms")
    print(f"  Std:  {np.std(decode_times):.2f} ms")
    print(f"  P95:  {np.percentile(decode_times, 95):.2f} ms")


def plot_jitter_cdf(stats, output_path, threshold=JITTER_THRESHOLD):
    """Plot three-stage jitter CDF."""
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Filter and prepare data (use absolute jitter values)
    rendering = np.abs(stats['rendering_jitters'])
    network = np.abs(stats['network_jitters'])
    decoding = np.abs(stats['decoding_jitters'])
    
    # Clip to reasonable range for visualization
    max_val = 20
    rendering = rendering[rendering < max_val]
    network = network[network < max_val]
    decoding = decoding[decoding < max_val]
    
    # Sort for CDF
    rendering_sorted = np.sort(rendering)
    network_sorted = np.sort(network)
    decoding_sorted = np.sort(decoding)
    
    # CDF values
    rendering_cdf = np.arange(1, len(rendering_sorted) + 1) / len(rendering_sorted)
    network_cdf = np.arange(1, len(network_sorted) + 1) / len(network_sorted)
    decoding_cdf = np.arange(1, len(decoding_sorted) + 1) / len(decoding_sorted)
    
    # Plot CDFs
    ax.plot(rendering_sorted, rendering_cdf, label='Rendering', linewidth=1.5, color='steelblue')
    ax.plot(network_sorted, network_cdf, label='Network', linewidth=1.5, color='orangered')
    ax.plot(decoding_sorted, decoding_cdf, label='Decoding', linewidth=1.5, color='forestgreen')
    
    # Add threshold line
    ax.axvline(x=threshold, color='gray', linestyle='--', linewidth=1.5,
               label=f'τ = {threshold} ms')
    
    ax.set_xlabel('Jitter (ms)', fontsize=10)
    ax.set_ylabel('CDF', fontsize=10)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nJitter Statistics (absolute values):")
    print(f"  Rendering - Mean: {np.mean(rendering):.2f} ms, P(>τ): {np.mean(rendering > threshold)*100:.1f}%")
    print(f"  Network   - Mean: {np.mean(network):.2f} ms, P(>τ): {np.mean(network > threshold)*100:.1f}%")
    print(f"  Decoding  - Mean: {np.mean(decoding):.2f} ms, P(>τ): {np.mean(decoding > threshold)*100:.1f}%")


def plot_combined_jitter_cdf_for_paper(stats, output_path, threshold=JITTER_THRESHOLD):
    """
    Plot three-stage jitter CDF optimized for paper inclusion.
    Single column width, clean styling.
    """
    import scienceplots
    
    with plt.style.context(["science", "ieee"]):
        plt.figure(figsize=(2, 1.2))
        
        # Filter and prepare data
        rendering = np.abs(stats['rendering_jitters'])
        network = np.abs(stats['network_jitters'])
        decoding = np.abs(stats['decoding_jitters'])
        
        max_val = 15
        rendering = rendering[rendering < max_val]
        network = network[network < max_val]
        decoding = decoding[decoding < max_val]
        
        # Sort for CDF
        rendering_sorted = np.sort(rendering)
        network_sorted = np.sort(network)
        decoding_sorted = np.sort(decoding)
        
        rendering_cdf = np.arange(1, len(rendering_sorted) + 1) / len(rendering_sorted)
        network_cdf = np.arange(1, len(network_sorted) + 1) / len(network_sorted)
        decoding_cdf = np.arange(1, len(decoding_sorted) + 1) / len(decoding_sorted)
        
        # Plot CDFs
        plt.plot(rendering_sorted, rendering_cdf, label='Rendering')
        plt.plot(network_sorted, network_cdf, label='Network')
        plt.plot(decoding_sorted, decoding_cdf, label='Decoding')
        
        # Threshold line
        # plt.axvline(x=threshold, color='gray', linestyle='--', linewidth=0.5,
        #            label=f'τ={threshold}ms')
        
        plt.xlabel('Jitter (ms)', fontsize=6)
        plt.ylabel('CDF', fontsize=6)
        plt.xlim(0, 5)
        plt.ylim(0.4, 1)
        plt.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)
        plt.legend(frameon=True, loc="best", labelspacing=0.1, columnspacing=0.1, handletextpad=0.1, fontsize=6)
        plt.tick_params(axis='both', which='both', pad=2, labelsize=6)
        
        plt.savefig(output_path)
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300)
        plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_jitter_measurements.py <data_root_path> [max_files]")
        print("Example: python generate_jitter_measurements.py /path/to/traces 100")
        sys.exit(1)
    
    root_path = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Collecting traces from {root_path}...")
    trace_files = collect_all_traces(root_path, max_files)
    print(f"Found {len(trace_files)} trace files")
    
    if len(trace_files) == 0:
        print("No trace files found!")
        sys.exit(1)
    
    print("\nAggregating statistics...")
    stats = aggregate_stats(trace_files)
    
    print(f"\nTotal samples collected:")
    print(f"  PTS intervals: {len(stats['pts_intervals'])}")
    print(f"  Decode times: {len(stats['decode_times'])}")
    print(f"  Network jitters: {len(stats['network_jitters'])}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_combined_jitter_cdf_for_paper(
        stats,
        os.path.join(OUTPUT_DIR, 'jitter_cdf_paper.pdf')
    )
    
    # Save raw statistics for potential use in paper
    np.savez(
        os.path.join(OUTPUT_DIR, 'jitter_stats.npz'),
        pts_intervals=stats['pts_intervals'],
        pts_deviations=stats['pts_deviations'],
        decode_times=stats['decode_times'],
        network_jitters=stats['network_jitters'],
        rendering_jitters=stats['rendering_jitters'],
        decoding_jitters=stats['decoding_jitters'],
    )
    
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    print("  - jitter_cdf_paper.pdf")
    print("  - jitter_stats.npz (raw data)")


if __name__ == "__main__":
    main()
