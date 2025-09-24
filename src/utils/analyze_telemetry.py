#!/usr/bin/env python
"""
Telemetry Analysis Tool for Fraud Detection System

This script analyzes telemetry data collected during the preprocessing pipeline.
It provides insights into performance, resource utilization, and data quality metrics.

Usage:
    python analyze_telemetry.py --telemetry-file <path_to_telemetry_json>
    python analyze_telemetry.py --telemetry-dir <directory_with_telemetry_files>
    python analyze_telemetry.py --compare <telemetry_file1> <telemetry_file2> [--output report.html]

Examples:
    # Analyze a single telemetry file
    python analyze_telemetry.py --telemetry-file data/processed/telemetry_20250924_213331.json
    
    # Analyze all telemetry files in a directory
    python analyze_telemetry.py --telemetry-dir data/processed/
    
    # Compare two different runs (e.g., GPU vs CPU)
    python analyze_telemetry.py --compare data/processed/telemetry_gpu.json data/processed/telemetry_cpu.json
"""

import os
import sys
import json
import argparse
import glob
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import matplotlib.dates as mdates


class TelemetryAnalyzer:
    def __init__(self, telemetry_data: Union[Dict, List[Dict]], run_names: Optional[List[str]] = None):
        """
        Initialize the telemetry analyzer with data from one or more telemetry files.
        
        Args:
            telemetry_data: Dictionary or list of dictionaries containing telemetry data
            run_names: Optional list of names for the different runs (for comparison)
        """
        if isinstance(telemetry_data, dict):
            self.telemetry = [telemetry_data]
        else:
            self.telemetry = telemetry_data
        
        # Assign names to the runs
        if run_names and len(run_names) == len(self.telemetry):
            self.run_names = run_names
        else:
            self.run_names = [f"Run {i+1}" for i in range(len(self.telemetry))]
    
    def extract_execution_timeline(self) -> pd.DataFrame:
        """Extract the execution timeline from all runs into a DataFrame."""
        timeline_data = []
        
        for i, telemetry in enumerate(self.telemetry):
            run_name = self.run_names[i]
            timestamps = telemetry.get('timestamps', {})
            
            # Process timestamps
            for event, timestamp in timestamps.items():
                # Find event pairs (start/end)
                if event.endswith('_start'):
                    base_event = event[:-6]  # Remove '_start'
                    end_event = f"{base_event}_end"
                    
                    if end_event in timestamps:
                        start_time = timestamp
                        end_time = timestamps[end_event]
                        duration = end_time - start_time
                        
                        timeline_data.append({
                            'run': run_name,
                            'event': base_event,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration,
                            'start_offset': start_time - telemetry.get('timestamps', {}).get('pipeline_start', 0),
                            'end_offset': end_time - telemetry.get('timestamps', {}).get('pipeline_start', 0)
                        })
        
        # Convert to DataFrame
        if timeline_data:
            return pd.DataFrame(timeline_data)
        else:
            return pd.DataFrame(columns=['run', 'event', 'start_time', 'end_time', 'duration', 'start_offset', 'end_offset'])
    
    def extract_performance_metrics(self) -> pd.DataFrame:
        """Extract performance metrics from all runs into a DataFrame."""
        metrics_data = []
        
        for i, telemetry in enumerate(self.telemetry):
            run_name = self.run_names[i]
            
            # Extract basic metrics
            basic_metrics = {
                'total_execution_time': telemetry.get('execution_time', 0),
                'rows_per_second': telemetry.get('metrics', {}).get('rows_per_second', 0),
                'row_count': telemetry.get('metrics', {}).get('raw_row_count', 0),
            }
            
            # Extract velocity calculation metrics
            velocity_metrics = {}
            if 'skip_velocity' in telemetry.get('metrics', {}):
                velocity_metrics['skip_velocity'] = telemetry['metrics']['skip_velocity']
            
            if 'gpu_velocity_calc_time' in telemetry.get('metrics', {}):
                velocity_metrics['gpu_velocity_calc_time'] = telemetry['metrics']['gpu_velocity_calc_time']
                velocity_metrics['transaction_processing_rate'] = (
                    telemetry['metrics'].get('transactions_processed', 0) / 
                    telemetry['metrics']['gpu_velocity_calc_time'] if telemetry['metrics']['gpu_velocity_calc_time'] > 0 else 0
                )
            elif 'cpu_velocity_calc_time' in telemetry.get('metrics', {}):
                velocity_metrics['cpu_velocity_calc_time'] = telemetry['metrics']['cpu_velocity_calc_time']
                velocity_metrics['transaction_processing_rate'] = (
                    telemetry['metrics'].get('transactions_processed', 0) / 
                    telemetry['metrics']['cpu_velocity_calc_time'] if telemetry['metrics']['cpu_velocity_calc_time'] > 0 else 0
                )
            
            # Extract system metrics
            sys_info = telemetry.get('system_info', {})
            system_metrics = {
                'cpu_count': sys_info.get('cpu_count', 0),
                'memory_total_gb': sys_info.get('memory_total', 0) / (1024**3),
                'has_gpu': sys_info.get('has_gpu', False),
            }
            
            if sys_info.get('gpu_info'):
                gpu_info = sys_info['gpu_info']
                system_metrics['gpu_name'] = gpu_info.get('name', 'Unknown')
                system_metrics['gpu_memory_gb'] = gpu_info.get('memory_total', 0) / (1024**3)
            
            # Extract file metrics
            file_metrics = {
                'input_file_size_mb': telemetry.get('metrics', {}).get('raw_file_size_bytes', 0) / (1024**2),
                'output_file_size_mb': telemetry.get('metrics', {}).get('output_file_size_mb', 0),
            }
            
            # Extract data quality metrics
            data_quality = {
                'missing_values_count': telemetry.get('metrics', {}).get('missing_values_count', 0),
                'duplicate_rows': telemetry.get('metrics', {}).get('duplicate_rows', 0),
                'fraud_transaction_count': telemetry.get('metrics', {}).get('fraud_transaction_count', 0),
                'fraud_transaction_percent': telemetry.get('metrics', {}).get('fraud_transaction_percent', 0),
            }
            
            # Combine all metrics
            combined_metrics = {
                'run': run_name,
                **basic_metrics,
                **velocity_metrics,
                **system_metrics,
                **file_metrics,
                **data_quality,
            }
            
            metrics_data.append(combined_metrics)
        
        # Convert to DataFrame
        return pd.DataFrame(metrics_data)
    
    def extract_events_timeline(self) -> pd.DataFrame:
        """Extract events into a timeline DataFrame."""
        events_data = []
        
        for i, telemetry in enumerate(self.telemetry):
            run_name = self.run_names[i]
            events = telemetry.get('events', [])
            
            for event in events:
                # Skip timestamp events as they're handled separately
                if event.get('message', '').startswith('Timestamp:'):
                    continue
                    
                events_data.append({
                    'run': run_name,
                    'timestamp': event.get('timestamp', 0),
                    'elapsed_seconds': event.get('elapsed_seconds', 0),
                    'message': event.get('message', ''),
                })
        
        # Convert to DataFrame and sort by elapsed time
        if events_data:
            df = pd.DataFrame(events_data)
            return df.sort_values('elapsed_seconds')
        else:
            return pd.DataFrame(columns=['run', 'timestamp', 'elapsed_seconds', 'message'])
    
    def plot_execution_timeline(self, output_file: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot the execution timeline for all runs.
        
        Args:
            output_file: If provided, save the plot to this file
            figsize: Figure size as (width, height) in inches
        """
        timeline_df = self.extract_execution_timeline()
        
        if timeline_df.empty:
            print("No timeline data available to plot")
            return
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Set up color palette for different runs
        palette = sns.color_palette("husl", len(self.run_names))
        run_colors = {run: palette[i] for i, run in enumerate(self.run_names)}
        
        # Group by run and event, taking the median for multiple instances of the same event
        timeline_summary = timeline_df.groupby(['run', 'event']).agg({
            'duration': 'median',
            'start_offset': 'median',
            'end_offset': 'median'
        }).reset_index()
        
        # Sort by start_offset within each run
        timeline_summary = timeline_summary.sort_values(['run', 'start_offset'])
        
        # Plot horizontal bars for each event
        y_pos = 0
        y_ticks = []
        y_labels = []
        
        for run_name in self.run_names:
            run_data = timeline_summary[timeline_summary['run'] == run_name]
            
            # Add run label
            y_ticks.append(y_pos + 0.5)
            y_labels.append(run_name)
            y_pos += 1
            
            # Plot each event for this run
            for _, row in run_data.iterrows():
                plt.barh(y_pos, row['duration'], left=row['start_offset'], 
                         color=run_colors[run_name], alpha=0.7, edgecolor='black')
                
                # Add event label if the bar is wide enough
                if row['duration'] > timeline_summary['duration'].max() * 0.05:
                    plt.text(row['start_offset'] + row['duration'] / 2, y_pos, 
                             row['event'], ha='center', va='center', 
                             fontsize=8, color='black')
                
                y_pos += 1
            
            # Add spacing between runs
            y_pos += 1
        
        # Set up plot appearance
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.xlabel('Time (seconds)')
        plt.yticks(y_ticks, y_labels)
        plt.title('Execution Timeline by Stage')
        
        # Add legend
        legend_handles = [plt.Rectangle((0,0), 1, 1, color=color) for color in run_colors.values()]
        plt.legend(legend_handles, run_colors.keys(), loc='upper right')
        
        # Save or show plot
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Timeline plot saved to {output_file}")
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_performance_comparison(self, output_file: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Plot a comparison of performance metrics across runs.
        
        Args:
            output_file: If provided, save the plot to this file
            figsize: Figure size as (width, height) in inches
        """
        metrics_df = self.extract_performance_metrics()
        
        if metrics_df.empty or len(metrics_df) < 2:
            print("Insufficient data for performance comparison")
            return
        
        # Select metrics to compare
        performance_metrics = [
            'total_execution_time', 'rows_per_second', 'transaction_processing_rate',
            'input_file_size_mb', 'output_file_size_mb'
        ]
        
        # Filter only metrics that exist in the DataFrame
        available_metrics = [m for m in performance_metrics if m in metrics_df.columns]
        
        if not available_metrics:
            print("No comparable performance metrics found")
            return
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=figsize)
        
        # If only one metric, convert axes to array for consistent indexing
        if len(available_metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(available_metrics):
            sns.barplot(x='run', y=metric, data=metrics_df, ax=axes[i])
            axes[i].set_title(f'Comparison of {metric.replace("_", " ").title()}')
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on top of bars
            for j, v in enumerate(metrics_df[metric].values):
                axes[i].text(j, v, f"{v:.2f}", ha='center', va='bottom')
        
        # Set overall title
        fig.suptitle('Performance Metrics Comparison', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for title
        
        # Save or show plot
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Performance comparison plot saved to {output_file}")
        else:
            plt.show()
    
    def plot_event_frequency(self, output_file: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot the frequency of events over time.
        
        Args:
            output_file: If provided, save the plot to this file
            figsize: Figure size as (width, height) in inches
        """
        events_df = self.extract_events_timeline()
        
        if events_df.empty:
            print("No event data available to plot")
            return
        
        plt.figure(figsize=figsize)
        
        # Group events by run and time bins
        for run_name in self.run_names:
            run_events = events_df[events_df['run'] == run_name]
            
            if not run_events.empty:
                # Create histogram of events over time
                plt.hist(run_events['elapsed_seconds'], bins=30, alpha=0.5, label=run_name)
        
        plt.xlabel('Elapsed Time (seconds)')
        plt.ylabel('Number of Events')
        plt.title('Event Frequency Over Time')
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save or show plot
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Event frequency plot saved to {output_file}")
        else:
            plt.tight_layout()
            plt.show()
    
    def generate_performance_report(self) -> str:
        """
        Generate a detailed performance report as a string.
        
        Returns:
            A formatted string with the performance report
        """
        metrics_df = self.extract_performance_metrics()
        timeline_df = self.extract_execution_timeline()
        
        report = []
        report.append("# Fraud Detection Preprocessing Performance Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall statistics
        report.append("## Overall Performance Summary")
        
        # Format the metrics dataframe for display
        formatted_metrics = metrics_df.copy()
        for col in formatted_metrics.columns:
            if col not in ['run']:
                try:
                    if formatted_metrics[col].dtype in [np.float64, np.float32]:
                        formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:.2f}")
                except:
                    pass
        
        report.append(tabulate(formatted_metrics, headers='keys', tablefmt='pipe', showindex=False))
        report.append("")
        
        # Timeline statistics
        report.append("## Processing Stage Durations (seconds)")
        
        # Group by run and event, calculate statistics
        if not timeline_df.empty:
            timeline_stats = timeline_df.groupby(['run', 'event'])['duration'].agg(['mean', 'min', 'max']).reset_index()
            timeline_stats = timeline_stats.sort_values(['run', 'mean'], ascending=[True, False])
            
            # Format for display
            timeline_stats['mean'] = timeline_stats['mean'].apply(lambda x: f"{x:.3f}")
            timeline_stats['min'] = timeline_stats['min'].apply(lambda x: f"{x:.3f}")
            timeline_stats['max'] = timeline_stats['max'].apply(lambda x: f"{x:.3f}")
            
            report.append(tabulate(timeline_stats, headers=['Run', 'Stage', 'Mean (s)', 'Min (s)', 'Max (s)'], 
                                  tablefmt='pipe', showindex=False))
        else:
            report.append("No timeline data available")
        
        report.append("")
        
        # System information
        report.append("## System Information")
        
        for i, telemetry in enumerate(self.telemetry):
            report.append(f"### {self.run_names[i]}")
            
            sys_info = telemetry.get('system_info', {})
            report.append(f"- OS: {sys_info.get('os', 'Unknown')} {sys_info.get('os_version', '')}")
            report.append(f"- Python Version: {sys_info.get('python_version', 'Unknown')}")
            report.append(f"- CPU Cores: {sys_info.get('cpu_count', 'Unknown')} physical, {sys_info.get('cpu_logical_count', 'Unknown')} logical")
            report.append(f"- Memory: {sys_info.get('memory_total', 0) / (1024**3):.2f} GB")
            
            if sys_info.get('has_gpu', False) and sys_info.get('gpu_info'):
                gpu_info = sys_info['gpu_info']
                report.append(f"- GPU: {gpu_info.get('name', 'Unknown')}")
                report.append(f"- GPU Memory: {gpu_info.get('memory_total', 0) / (1024**3):.2f} GB")
                report.append(f"- CUDA Version: {gpu_info.get('cuda_version', 'Unknown')}")
            else:
                report.append("- GPU: Not available or not used")
            
            report.append("")
        
        # Data quality metrics
        report.append("## Data Quality Metrics")
        
        for i, telemetry in enumerate(self.telemetry):
            report.append(f"### {self.run_names[i]}")
            
            metrics = telemetry.get('metrics', {})
            report.append(f"- Original Rows: {metrics.get('raw_row_count', 0):,}")
            report.append(f"- Missing Values: {metrics.get('missing_values_count', 0):,}")
            report.append(f"- Duplicate Rows: {metrics.get('duplicate_rows', 0):,}")
            report.append(f"- Fraud Transactions: {metrics.get('fraud_transaction_count', 0):,} ({metrics.get('fraud_transaction_percent', 0):.2f}%)")
            
            # Infinity value handling
            if 'infinity_value_counts' in metrics:
                report.append("- Infinity Values Found:")
                inf_counts = metrics['infinity_value_counts']
                for feature, counts in inf_counts.items():
                    report.append(f"  - {feature}: {counts['inf']} infinity, {counts['nan']} NaN")
            
            report.append("")
        
        # Join the report with newlines
        return "\n".join(report)
    
    def save_report(self, output_file: str) -> None:
        """
        Save the performance report to a file.
        
        Args:
            output_file: Path to save the report
        """
        report = self.generate_performance_report()
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Performance report saved to {output_file}")


def load_telemetry_file(file_path: str) -> Dict:
    """
    Load telemetry data from a JSON file.
    
    Args:
        file_path: Path to the telemetry JSON file
    
    Returns:
        Dictionary containing the telemetry data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading telemetry file {file_path}: {str(e)}")
        sys.exit(1)


def find_telemetry_files(directory: str) -> List[str]:
    """
    Find all telemetry JSON files in a directory.
    
    Args:
        directory: Directory to search for telemetry files
    
    Returns:
        List of paths to telemetry files
    """
    # Get all JSON files in directory and subdirectories
    telemetry_files = glob.glob(os.path.join(directory, "**", "telemetry_*.json"), recursive=True)
    
    if not telemetry_files:
        print(f"No telemetry files found in {directory}")
        sys.exit(1)
    
    return telemetry_files


def main() -> None:
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze telemetry data from the fraud detection preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Create mutually exclusive group for input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--telemetry-file", help="Path to a single telemetry JSON file")
    input_group.add_argument("--telemetry-dir", help="Directory containing telemetry JSON files")
    input_group.add_argument("--compare", nargs='+', help="Two or more telemetry files to compare")
    
    # Output options
    parser.add_argument("--output", help="Output file for the report (default is stdout)")
    parser.add_argument("--plots-dir", help="Directory to save plots (default is current directory)")
    parser.add_argument("--format", choices=['text', 'markdown', 'html'], default='markdown',
                      help="Output format for the report")
    
    args = parser.parse_args()
    
    # Load telemetry data based on input mode
    telemetry_data = []
    run_names = []
    
    if args.telemetry_file:
        telemetry_data = [load_telemetry_file(args.telemetry_file)]
        run_names = ["Single Run"]
    
    elif args.telemetry_dir:
        telemetry_files = find_telemetry_files(args.telemetry_dir)
        telemetry_data = [load_telemetry_file(file) for file in telemetry_files]
        
        # Extract run names from filenames
        run_names = [os.path.basename(file).replace('telemetry_', '').replace('.json', '') for file in telemetry_files]
    
    elif args.compare:
        if len(args.compare) < 2:
            print("At least two telemetry files are required for comparison")
            sys.exit(1)
        
        telemetry_data = [load_telemetry_file(file) for file in args.compare]
        
        # Try to determine run type from metrics
        run_names = []
        for data in telemetry_data:
            metrics = data.get('metrics', {})
            if metrics.get('use_gpu', False):
                run_names.append("GPU Run")
            elif metrics.get('skip_velocity', False):
                run_names.append("CPU (Skip Velocity)")
            else:
                run_names.append("CPU Run")
        
        # If we couldn't determine unique names, use numbered runs
        if len(set(run_names)) < len(run_names):
            run_names = [f"Run {i+1}" for i in range(len(telemetry_data))]
    
    # Create analyzer
    analyzer = TelemetryAnalyzer(telemetry_data, run_names)
    
    # Generate and output report
    report = analyzer.generate_performance_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)
    
    # Create plots directory if specified
    plots_dir = args.plots_dir or '.'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plots
    if len(telemetry_data) >= 2:
        # Comparison plots
        analyzer.plot_performance_comparison(
            output_file=os.path.join(plots_dir, 'performance_comparison.png'))
    
    # Timeline plots (for all cases)
    analyzer.plot_execution_timeline(
        output_file=os.path.join(plots_dir, 'execution_timeline.png'))
    
    analyzer.plot_event_frequency(
        output_file=os.path.join(plots_dir, 'event_frequency.png'))


if __name__ == "__main__":
    main()
