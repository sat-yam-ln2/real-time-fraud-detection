"""
Preprocessing module for fraud detection system.

This module provides functions to load, clean, transform, and save transaction data.
It implements the canonical feature transformations defined in feature_defs.yml.

How to run:
    python preprocess.py --input <path_to_input_file> --output <path_to_output_file>

Example:
    # Generate synthetic data first if you don't have any
    python data/generate_synthetic.py --output data/raw/transactions_raw.csv
    
    # Then process the data
    python src/features/preprocess.py --input data/raw/transactions_raw.csv --output data/processed/transactions_processed.csv

Required arguments:
    --input: Path to the raw data CSV file
    --output: Path where the processed data will be saved
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import yaml
import time
import json
import psutil
import platform
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Optional, Union, Tuple

# Import GPU libraries conditionally
try:
    import cupy as cp
    import torch
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        GPU_INFO = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "cuda_version": torch.version.cuda
        }
    else:
        GPU_INFO = None
except ImportError:
    HAS_GPU = False
    GPU_INFO = None

# Telemetry class to track performance metrics
class Telemetry:
    def __init__(self):
        self.start_time = time.time()
        self.timestamps = {}
        self.counters = {}
        self.metrics = {}
        self.events = []
        
        # Capture system information
        self.system_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_logical_count": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "has_gpu": HAS_GPU,
            "gpu_info": GPU_INFO
        }
    
    def mark_timestamp(self, name: str) -> None:
        """Record a timestamp with a given name"""
        self.timestamps[name] = time.time()
        # Also log as an event
        self.log_event(f"Timestamp: {name}")
    
    def get_elapsed_time(self, start_name: str, end_name: str = None) -> float:
        """Calculate elapsed time between two timestamps"""
        if end_name is None:
            end_time = time.time()
        else:
            end_time = self.timestamps.get(end_name, time.time())
        
        start_time = self.timestamps.get(start_name, self.start_time)
        return end_time - start_time
    
    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a named counter"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += amount
    
    def add_metric(self, name: str, value: Union[int, float, str]) -> None:
        """Add a metric value"""
        self.metrics[name] = value
    
    def log_event(self, message: str) -> None:
        """Log an event with timestamp"""
        self.events.append({
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self.start_time,
            "message": message
        })
    
    def get_current_usage(self) -> Dict:
        """Get current CPU and memory usage"""
        process = psutil.Process(os.getpid())
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_usage": process.memory_info().rss,
            "memory_percent": process.memory_percent()
        }
    
    def get_summary(self) -> Dict:
        """Get complete telemetry summary"""
        return {
            "system_info": self.system_info,
            "execution_time": time.time() - self.start_time,
            "timestamps": self.timestamps,
            "counters": self.counters,
            "metrics": self.metrics,
            "events": self.events,
            "current_usage": self.get_current_usage(),
            "collected_at": datetime.now().isoformat()
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save telemetry data to a file"""
        # Get final summary
        summary = self.get_summary()
        
        # Save to JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Telemetry data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving telemetry data: {e}")

# Initialize telemetry
telemetry = Telemetry()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_feature_definitions(feature_defs_path: Optional[str] = None) -> Dict:
    """
    Load feature definitions from YAML file.
    
    Args:
        feature_defs_path: Path to feature definitions YAML file. If None, 
                          uses default path.
    
    Returns:
        Dictionary containing feature definitions.
    """
    if feature_defs_path is None:
        feature_defs_path = PROJECT_ROOT / 'src' / 'features' / 'feature_defs.yml'
    
    try:
        with open(feature_defs_path, 'r') as file:
            feature_defs = yaml.safe_load(file)
        logger.info(f"Successfully loaded feature definitions from {feature_defs_path}")
        return feature_defs
    except Exception as e:
        logger.error(f"Error loading feature definitions: {e}")
        raise


def load_raw(path: str) -> pd.DataFrame:
    """
    Load raw transaction data from CSV file.
    
    Args:
        path: Path to raw transaction data CSV file.
    
    Returns:
        DataFrame containing raw transaction data.
    """
    telemetry.mark_timestamp("load_raw_start")
    telemetry.log_event(f"Loading raw data from {path}")
    logger.info(f"Loading raw data from {path}")
    try:
        df = pd.read_csv(path)
        row_count = len(df)
        column_count = len(df.columns)
        telemetry.add_metric("raw_row_count", row_count)
        telemetry.add_metric("raw_column_count", column_count)
        telemetry.add_metric("raw_file_size_bytes", os.path.getsize(path))
        telemetry.mark_timestamp("load_raw_end")
        telemetry.log_event(f"Successfully loaded {row_count} rows from {path}")
        logger.info(f"Successfully loaded {row_count} rows from {path}")
        return df
    except Exception as e:
        telemetry.log_event(f"Error loading raw data: {str(e)}")
        logger.error(f"Error loading raw data: {e}")
        raise


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw transaction data by handling missing values and invalid entries.
    
    Args:
        df: DataFrame containing raw transaction data.
    
    Returns:
        Cleaned DataFrame.
    """
    telemetry.mark_timestamp("clean_start")
    telemetry.log_event("Starting data cleaning")
    logger.info("Cleaning raw data")
    
    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Track original shape
    original_shape = cleaned_df.shape
    telemetry.add_metric("original_row_count", original_shape[0])
    telemetry.add_metric("original_column_count", original_shape[1])
    logger.info(f"Original data shape: {original_shape}")
    
    # Check for and handle missing values
    missing_values = cleaned_df.isnull().sum()
    missing_total = missing_values.sum()
    if missing_total > 0:
        telemetry.add_metric("missing_values_count", missing_total)
        telemetry.add_metric("missing_values_percent", (missing_total / (original_shape[0] * original_shape[1])) * 100)
        telemetry.log_event(f"Found {missing_total} missing values")
        
        # Record missing values by column
        for col in missing_values[missing_values > 0].index:
            telemetry.add_metric(f"missing_values_{col}", missing_values[col])
        
        logger.info(f"Found {missing_total} missing values")
        logger.info(f"Missing values by column:\n{missing_values[missing_values > 0]}")
        
        # For V1-V28 features, impute with median
        v_columns = [col for col in cleaned_df.columns if col.startswith('V')]
        if v_columns:
            telemetry.log_event(f"Imputing {len(v_columns)} V-columns with median")
            imputer = SimpleImputer(strategy='median')
            cleaned_df[v_columns] = imputer.fit_transform(cleaned_df[v_columns])
        
        # For Amount, impute with median if any missing values
        amount_missing = cleaned_df['Amount'].isnull().sum()
        if amount_missing > 0:
            telemetry.log_event(f"Imputing {amount_missing} Amount values with median")
            cleaned_df['Amount'] = cleaned_df['Amount'].fillna(cleaned_df['Amount'].median())
        
        # For Time, impute with previous value if any missing values
        time_missing = cleaned_df['Time'].isnull().sum()
        if time_missing > 0:
            telemetry.log_event(f"Imputing {time_missing} Time values with ffill")
            cleaned_df['Time'] = cleaned_df['Time'].fillna(method='ffill')
    
    # Handle invalid values
    # Ensure Time is non-negative
    negative_times = (cleaned_df['Time'] < 0).sum()
    if negative_times > 0:
        telemetry.log_event(f"Fixed {negative_times} negative Time values")
        telemetry.add_metric("negative_time_values", negative_times)
        logger.warning(f"Found {negative_times} negative Time values, setting to 0")
        cleaned_df.loc[cleaned_df['Time'] < 0, 'Time'] = 0
    
    # Ensure Amount is non-negative
    negative_amounts = (cleaned_df['Amount'] < 0).sum()
    if negative_amounts > 0:
        telemetry.log_event(f"Fixed {negative_amounts} negative Amount values")
        telemetry.add_metric("negative_amount_values", negative_amounts)
        logger.warning(f"Found {negative_amounts} negative Amount values, setting to 0")
        cleaned_df.loc[cleaned_df['Amount'] < 0, 'Amount'] = 0
    
    # Ensure Class is binary (0 or 1)
    if not set(cleaned_df['Class'].unique()).issubset({0, 1, '0', '1'}):
        non_binary_classes = len(set(cleaned_df['Class'].unique()) - {0, 1, '0', '1'})
        telemetry.log_event(f"Fixed {non_binary_classes} non-binary Class values")
        telemetry.add_metric("non_binary_class_values", non_binary_classes)
        logger.warning(f"Found non-binary Class values: {cleaned_df['Class'].unique()}")
        # Convert to numeric first
        cleaned_df['Class'] = pd.to_numeric(cleaned_df['Class'], errors='coerce')
        # Then ensure it's binary
        cleaned_df['Class'] = cleaned_df['Class'].apply(lambda x: 1 if x == 1 else 0)
    
    # Convert Class to integer type
    cleaned_df['Class'] = cleaned_df['Class'].astype(int)
    
    # Check for duplicates
    duplicates = cleaned_df.duplicated().sum()
    if duplicates > 0:
        telemetry.log_event(f"Removed {duplicates} duplicate rows")
        telemetry.add_metric("duplicate_rows", duplicates)
        logger.warning(f"Found {duplicates} duplicate rows, removing")
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Calculate class distribution
    fraud_count = cleaned_df['Class'].sum()
    total_count = len(cleaned_df)
    fraud_percentage = (fraud_count / total_count) * 100
    telemetry.add_metric("fraud_transaction_count", fraud_count)
    telemetry.add_metric("fraud_transaction_percent", fraud_percentage)
    
    # Log cleaning results
    rows_removed = original_shape[0] - cleaned_df.shape[0]
    telemetry.add_metric("rows_removed_by_cleaning", rows_removed)
    telemetry.mark_timestamp("clean_end")
    telemetry.log_event(f"Cleaning complete - removed {rows_removed} rows")
    logger.info(f"After cleaning: {cleaned_df.shape} (removed {rows_removed} rows)")
    
    return cleaned_df


def feature_engineer(df: pd.DataFrame, skip_velocity_calculation: bool = False, use_gpu: bool = False) -> pd.DataFrame:
    """
    Engineer features for fraud detection model according to feature_defs.yml.
    
    Args:
        df: DataFrame containing cleaned transaction data.
        skip_velocity_calculation: If True, skips the computationally intensive
                                transaction velocity calculation for large datasets.
        use_gpu: If True, uses GPU acceleration for computationally intensive operations.
                 Requires CuPy and PyTorch to be installed.
    
    Returns:
        DataFrame with engineered features.
    """
    telemetry.mark_timestamp("feature_engineering_start")
    telemetry.log_event("Starting feature engineering")
    telemetry.add_metric("skip_velocity_calculation", skip_velocity_calculation)
    telemetry.add_metric("use_gpu", use_gpu)
    
    # Check if GPU usage was requested but GPU libraries are not available
    if use_gpu and not HAS_GPU:
        telemetry.log_event("GPU acceleration requested but not available - falling back to CPU")
        logger.warning("GPU acceleration requested but required libraries (CuPy/PyTorch) not found or GPU not available.")
        logger.warning("Falling back to CPU implementation. Install with: pip install cupy torch")
        use_gpu = False
    logger.info("Engineering features")
    
    # Create a copy to avoid modifying the original dataframe
    engineered_df = df.copy()
    
    # Load feature definitions
    feature_defs = load_feature_definitions()
    
    # ======= Time-based Features =======
    # Convert Time to datetime by assuming it's seconds from a reference date
    # For this implementation, we'll use an arbitrary reference date
    REF_DATE = datetime(2000, 1, 1)
    engineered_df['transaction_datetime'] = engineered_df['Time'].apply(
        lambda x: REF_DATE + timedelta(seconds=float(x))
    )
    
    # Extract hour of day
    engineered_df['hour_of_day'] = engineered_df['transaction_datetime'].dt.hour
    
    # Extract day of week
    engineered_df['day_of_week'] = engineered_df['transaction_datetime'].dt.dayofweek
    
    # Is weekend feature
    engineered_df['is_weekend'] = engineered_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # ======= Amount-based Features =======
    # Log transform for Amount (handle zero values)
    engineered_df['amount_scaled'] = np.log1p(engineered_df['Amount'])
    
    # ======= User Aggregated Features =======
    # Simulate user ID based on transaction patterns (in a real system, this would come from the data)
    # Here we're creating a synthetic user_id based on patterns in the V features
    # This is just for demonstration - in a real system, you'd have actual user IDs
    
    # Use combination of V1, V2, V3 to create a synthetic user_id
    # This is not perfect but helps demonstrate the feature engineering concept
    v_subset = ['V1', 'V2', 'V3']
    engineered_df['user_id'] = engineered_df[v_subset].apply(
        lambda row: hash(tuple(row.values.round(2))), axis=1
    ) % 10000  # Limit to a reasonable number of synthetic users
    
    # Group by user_id and calculate aggregates
    # In a real-time system, these would be calculated using a rolling window
    try:
        telemetry.log_event("Calculating user aggregates")
        user_aggregates = engineered_df.groupby('user_id').agg(
            user_avg_amount=('Amount', 'mean'),
            user_tx_count=('Amount', 'count'),
            user_amount_std=('Amount', 'std')
        ).reset_index()
        
        # Check for and handle problematic values in aggregates
        for col in ['user_avg_amount', 'user_amount_std']:
            # Count problematic values
            inf_count = np.isinf(user_aggregates[col]).sum()
            nan_count = np.isnan(user_aggregates[col]).sum()
            
            if inf_count > 0 or nan_count > 0:
                telemetry.log_event(f"Found {inf_count} infinity and {nan_count} NaN values in {col}")
                logger.warning(f"Found {inf_count} infinity and {nan_count} NaN values in {col}")
                
                # Replace infinity with zeros or means
                user_aggregates[col] = user_aggregates[col].replace([np.inf, -np.inf], np.nan)
                # Replace NaN with column mean or 0
                col_mean = user_aggregates[col].mean()
                if np.isnan(col_mean):
                    user_aggregates[col] = user_aggregates[col].fillna(0)
                else:
                    user_aggregates[col] = user_aggregates[col].fillna(col_mean)
    
    except Exception as e:
        telemetry.log_event(f"Error in user aggregates calculation: {str(e)}")
        logger.error(f"Error calculating user aggregates: {e}")
        # Create a simplified version with just count
        user_aggregates = engineered_df.groupby('user_id').agg(
            user_tx_count=('Amount', 'count')
        ).reset_index()
        # Add placeholder columns with zeros
        user_aggregates['user_avg_amount'] = 0
        user_aggregates['user_amount_std'] = 0
        logger.warning("Using simplified user aggregates due to calculation error")
    
    # Merge aggregates back to the main dataframe
    engineered_df = pd.merge(engineered_df, user_aggregates, on='user_id', how='left')
    
    # Replace NaN in std with 0 (for users with only one transaction)
    engineered_df['user_amount_std'] = engineered_df['user_amount_std'].fillna(0)
    
    # Ensure numeric values are finite
    numeric_cols = ['user_avg_amount', 'user_amount_std', 'user_tx_count']
    for col in numeric_cols:
        if col in engineered_df.columns:
            # Replace any remaining infinity or NaN
            engineered_df[col] = engineered_df[col].replace([np.inf, -np.inf], 0)
            engineered_df[col] = engineered_df[col].fillna(0)
    
    # ======= Transaction Velocity Features =======
    # Sort by user_id and time
    engineered_df = engineered_df.sort_values(['user_id', 'Time'])
    
    # Calculate time difference between consecutive transactions for the same user
    engineered_df['time_since_last_tx'] = engineered_df.groupby('user_id')['Time'].diff()
    
    # Calculate transaction velocity (1-hour window)
    ONE_HOUR_IN_SECONDS = 3600
    
    # Initialize with zero velocity
    engineered_df['tx_velocity_1h'] = 0
    
    # Check if we should skip velocity calculation for large datasets
    if skip_velocity_calculation:
        telemetry.mark_timestamp("velocity_calculation_start")
        telemetry.log_event("Skipping full transaction velocity calculation - using simple approximation")
        logger.info("Skipping transaction velocity calculation (disabled for large datasets)")
        
        # For large datasets, we'll use a simpler approximation based on time difference
        # This is not as accurate but much faster
        engineered_df['tx_velocity_1h'] = (
            engineered_df.groupby('user_id')['time_since_last_tx']
            .transform(lambda x: (x <= ONE_HOUR_IN_SECONDS).astype(int))
        )
        telemetry.mark_timestamp("velocity_calculation_end")
        
    else:
        # Fast approach for datasets with progress logging
        row_count = len(engineered_df)
        telemetry.add_metric("row_count_before_velocity", row_count)
        
        if row_count > 100000:
            telemetry.log_event(f"Large dataset detected for velocity calculation: {row_count:,} rows")
            logger.warning(f"Large dataset detected ({row_count:,} rows). Transaction velocity calculation may take a long time.")
            if not use_gpu:
                logger.warning("Consider using --use-gpu flag for faster processing or --skip-velocity-calculation for very large datasets.")
        
        if use_gpu and HAS_GPU:
            telemetry.mark_timestamp("gpu_velocity_calculation_start")
            telemetry.log_event("Starting GPU-accelerated transaction velocity calculation")
            logger.info("Calculating transaction velocities using GPU acceleration...")
            
            user_count = 0
            transaction_count = 0
            telemetry.add_metric("gpu_processing", True)
            
            # GPU-accelerated implementation using PyTorch
            # Group by user_id and process each group
            user_groups = engineered_df.groupby('user_id')
            total_users = len(user_groups)
            telemetry.add_metric("total_users", total_users)
            
            # Track GPU memory usage before processing
            if torch.cuda.is_available():
                telemetry.add_metric("gpu_memory_before", torch.cuda.memory_allocated())
            
            start_time = time.time()
            
            for i, (user_id, group) in enumerate(user_groups):
                # Log progress
                if i % 1000 == 0 or i == total_users - 1:
                    progress_pct = (i+1)/total_users*100
                    telemetry.log_event(f"GPU processing: {i+1}/{total_users} users ({progress_pct:.1f}%)")
                    logger.info(f"Processing user {i+1}/{total_users} ({progress_pct:.1f}%)")
                
                if len(group) <= 1:
                    continue
                
                user_count += 1
                transaction_count += len(group)
                
                # Sort by time
                group_sorted = group.sort_values('Time')
                times = group_sorted['Time'].values
                
                # Move data to GPU
                times_tensor = torch.tensor(times, device='cuda')
                
                # For each transaction, calculate time difference with all previous transactions
                # This is a vectorized version of the transaction velocity calculation
                # Create a tensor of shape (len(times), len(times)) where each row is the current time
                current_times = times_tensor.unsqueeze(1).expand(-1, len(times_tensor))
                # Create a tensor of shape (len(times), len(times)) where each column is a past time
                past_times = times_tensor.unsqueeze(0).expand(len(times_tensor), -1)
                
                # Calculate time differences (current - past)
                time_diffs = current_times - past_times
                
                # For each transaction, count how many previous transactions occurred within the last hour
                # We create a mask where True means the time difference is <= 1 hour and > 0
                mask = (time_diffs <= ONE_HOUR_IN_SECONDS) & (time_diffs > 0)
                
                # Sum the mask along the columns to get counts
                counts = mask.sum(dim=1).cpu().numpy()
                
                # Update the dataframe
                engineered_df.loc[group_sorted.index, 'tx_velocity_1h'] = counts
                
                # Record GPU memory usage every 1000 users
                if i % 1000 == 0 and torch.cuda.is_available():
                    telemetry.add_metric(f"gpu_memory_at_user_{i}", torch.cuda.memory_allocated())
            
            elapsed_time = time.time() - start_time
            telemetry.add_metric("gpu_velocity_calc_time", elapsed_time)
            telemetry.add_metric("users_processed", user_count)
            telemetry.add_metric("transactions_processed", transaction_count)
            telemetry.add_metric("transactions_per_second", transaction_count / elapsed_time)
            
            if torch.cuda.is_available():
                telemetry.add_metric("gpu_memory_after", torch.cuda.memory_allocated())
            
            telemetry.mark_timestamp("gpu_velocity_calculation_end")
            telemetry.log_event("GPU-accelerated transaction velocity calculation completed")
            logger.info("GPU-accelerated transaction velocity calculation completed.")
            
        else:
            telemetry.mark_timestamp("cpu_velocity_calculation_start")
            telemetry.log_event("Starting CPU transaction velocity calculation")
            logger.info("Calculating transaction velocities (optimized CPU method)...")
            
            user_count = 0
            transaction_count = 0
            transactions_processed = 0
            
            # Group by user_id
            user_groups = engineered_df.groupby('user_id')
            total_users = len(user_groups)
            telemetry.add_metric("total_users", total_users)
            
            start_time = time.time()
            
            # Process each user group with progress logging
            for i, (user_id, group) in enumerate(user_groups):
                # Log progress every 1000 users or at the end
                if i % 1000 == 0 or i == total_users - 1:
                    progress_pct = (i+1)/total_users*100
                    elapsed = time.time() - start_time
                    
                    # Record performance metrics
                    if elapsed > 0 and transactions_processed > 0:
                        tx_per_sec = transactions_processed / elapsed
                        telemetry.add_metric(f"tx_per_sec_at_user_{i}", tx_per_sec)
                    
                    telemetry.log_event(f"CPU processing: {i+1}/{total_users} users ({progress_pct:.1f}%)")
                    logger.info(f"Processing user {i+1}/{total_users} ({progress_pct:.1f}%)")
                
                if len(group) <= 1:
                    continue  # Skip users with only one transaction
                
                user_count += 1
                group_transactions = len(group)
                transaction_count += group_transactions
                
                # Sort by time
                group_sorted = group.sort_values('Time')
                times = group_sorted['Time'].values
                
                # For each time, find the number of transactions in the previous hour
                for j in range(len(times)):
                    current_time = times[j]
                    # Start searching from the most recent transaction that could be within an hour
                    start_idx = max(0, j - 100)  # Look back at most 100 transactions
                    
                    # Count transactions in the past hour
                    count = 0
                    for k in range(start_idx, j):
                        if current_time - times[k] <= ONE_HOUR_IN_SECONDS:
                            count += 1
                    
                    # Update the dataframe at the correct position
                    engineered_df.loc[group_sorted.index[j], 'tx_velocity_1h'] = count
                    transactions_processed += 1
            
            elapsed_time = time.time() - start_time
            telemetry.add_metric("cpu_velocity_calc_time", elapsed_time)
            telemetry.add_metric("users_processed", user_count)
            telemetry.add_metric("transactions_processed", transaction_count)
            telemetry.add_metric("transactions_per_second", transaction_count / elapsed_time if elapsed_time > 0 else 0)
            
            telemetry.mark_timestamp("cpu_velocity_calculation_end")
            telemetry.log_event("CPU transaction velocity calculation completed")
            logger.info("Transaction velocity calculation completed.")
    
    # Fill NaN values for first transactions
    # Use a large finite value instead of infinity
    max_time_value = engineered_df['Time'].max() * 2  # Use a large but finite value
    telemetry.add_metric("max_time_value_for_fill", max_time_value)
    
    # Replace NaN with the maximum time value (for first transactions)
    engineered_df['time_since_last_tx'] = engineered_df['time_since_last_tx'].fillna(max_time_value)
    engineered_df['tx_velocity_1h'] = engineered_df['tx_velocity_1h'].fillna(0)
    
    # Log how many values were filled
    filled_count = (engineered_df['time_since_last_tx'] >= max_time_value).sum()
    telemetry.add_metric("time_since_last_tx_filled_count", filled_count)
    telemetry.log_event(f"Filled {filled_count} missing time_since_last_tx values with {max_time_value} seconds")
    
    telemetry.log_event("Starting merchant feature calculation")
    telemetry.mark_timestamp("merchant_features_start")
    
    # ======= Merchant Features =======
    # Simulate merchant category based on V features
    # In a real system, you would have actual merchant IDs and categories
    engineered_df['merchant_category'] = engineered_df[['V5', 'V6', 'V7']].apply(
        lambda row: hash(tuple(row.values.round(2))), axis=1
    ) % 100  # Limit to 100 merchant categories
    
    # Calculate merchant risk score (for demonstration)
    # In a real system, this might come from historical fraud rates by merchant
    merchant_risk = engineered_df.groupby('merchant_category')['Class'].mean().reset_index()
    merchant_risk.rename(columns={'Class': 'merchant_risk_score'}, inplace=True)
    
    # Capture merchant category stats
    merchant_count = len(merchant_risk)
    telemetry.add_metric("merchant_category_count", merchant_count)
    
    # Calculate high risk merchants (risk score > 0.1)
    high_risk_merchants = (merchant_risk['merchant_risk_score'] > 0.1).sum()
    telemetry.add_metric("high_risk_merchant_count", high_risk_merchants)
    telemetry.add_metric("high_risk_merchant_percent", (high_risk_merchants / merchant_count) * 100)
    
    # Merge merchant risk back to main dataframe
    engineered_df = pd.merge(engineered_df, merchant_risk, on='merchant_category', how='left')
    
    telemetry.mark_timestamp("merchant_features_end")
    
    # ======= Feature Normalization =======
    telemetry.log_event("Starting feature normalization")
    telemetry.mark_timestamp("normalization_start")
    
    # Normalize numeric features
    numeric_features = ['amount_scaled', 'user_avg_amount', 'user_amount_std', 
                         'time_since_last_tx', 'tx_velocity_1h']
    
    # Handle infinity and very large values before normalization
    telemetry.log_event("Checking for infinity or very large values in features")
    
    # Count infinity and NaN values before cleaning
    inf_counts = {}
    for feature in numeric_features:
        if feature in engineered_df.columns:
            inf_count = np.isinf(engineered_df[feature]).sum()
            nan_count = np.isnan(engineered_df[feature]).sum()
            inf_counts[feature] = {'inf': inf_count, 'nan': nan_count}
            
            if inf_count > 0 or nan_count > 0:
                telemetry.log_event(f"Found {inf_count} infinity and {nan_count} NaN values in {feature}")
                logger.warning(f"Found {inf_count} infinity and {nan_count} NaN values in {feature}")
    
    telemetry.add_metric("infinity_value_counts", inf_counts)
    
    # Replace infinity with large but finite values
    for feature in numeric_features:
        if feature in engineered_df.columns:
            # Replace positive infinity with a large value (e.g., 1e8)
            engineered_df[feature] = engineered_df[feature].replace(np.inf, 1e8)
            # Replace negative infinity with a large negative value
            engineered_df[feature] = engineered_df[feature].replace(-np.inf, -1e8)
            # Replace NaN with 0 (or other appropriate value)
            engineered_df[feature] = engineered_df[feature].fillna(0)
    
    # Record basic statistics before normalization
    for feature in numeric_features:
        if feature in engineered_df.columns:
            telemetry.add_metric(f"{feature}_mean", engineered_df[feature].mean())
            telemetry.add_metric(f"{feature}_std", engineered_df[feature].std())
            telemetry.add_metric(f"{feature}_min", engineered_df[feature].min())
            telemetry.add_metric(f"{feature}_max", engineered_df[feature].max())
    
    # Apply robust scaling instead of standard scaling to handle outliers better
    try:
        scaler = StandardScaler()
        engineered_df[numeric_features] = scaler.fit_transform(engineered_df[numeric_features])
        telemetry.log_event("Standard scaling completed successfully")
    except Exception as e:
        # Fallback to manual normalization if StandardScaler fails
        telemetry.log_event(f"Standard scaling failed: {str(e)}")
        logger.warning(f"Standard scaling failed, using manual min-max normalization: {e}")
        
        # Apply manual min-max scaling
        for feature in numeric_features:
            if feature in engineered_df.columns:
                min_val = engineered_df[feature].min()
                max_val = engineered_df[feature].max()
                if max_val > min_val:  # Avoid division by zero
                    engineered_df[feature] = (engineered_df[feature] - min_val) / (max_val - min_val)
                else:
                    engineered_df[feature] = 0
        telemetry.log_event("Manual min-max normalization completed as fallback")
    
    telemetry.mark_timestamp("normalization_end")
    telemetry.mark_timestamp("feature_engineering_end")
    
    # Record final engineered data shape
    telemetry.add_metric("engineered_row_count", engineered_df.shape[0])
    telemetry.add_metric("engineered_column_count", engineered_df.shape[1])
    telemetry.log_event(f"Feature engineering complete. New shape: {engineered_df.shape}")
    logger.info(f"Successfully engineered features. New shape: {engineered_df.shape}")
    
    return engineered_df


def save_processed(df: pd.DataFrame, out_path: str) -> None:
    """
    Save processed data to CSV file.
    
    Args:
        df: DataFrame containing processed data.
        out_path: Path to save processed data CSV file.
    
    Returns:
        None
    """
    telemetry.mark_timestamp("save_start")
    telemetry.log_event(f"Saving processed data to {out_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    logger.info(f"Saving processed data to {out_path}")
    try:
        df.to_csv(out_path, index=False)
        
        # Record file size
        file_size = os.path.getsize(out_path)
        telemetry.add_metric("output_file_size_bytes", file_size)
        telemetry.add_metric("output_file_size_mb", file_size / (1024 * 1024))
        
        telemetry.log_event(f"Successfully saved {len(df)} rows to {out_path}")
        logger.info(f"Successfully saved {len(df)} rows to {out_path}")
    except Exception as e:
        telemetry.log_event(f"Error saving processed data: {str(e)}")
        logger.error(f"Error saving processed data: {e}")
        raise
    
    telemetry.mark_timestamp("save_end")


def main(input_path: str, output_path: str, skip_velocity: bool = False, use_gpu: bool = False) -> None:
    """
    Main function to run the full preprocessing pipeline.
    
    Args:
        input_path: Path to raw data CSV file.
        output_path: Path to save processed data CSV file.
        skip_velocity: If True, skips the computationally intensive velocity calculation.
        use_gpu: If True, uses GPU acceleration for computationally intensive operations.
    
    Returns:
        None
    """
    # Initialize the overall pipeline start time
    telemetry.mark_timestamp("pipeline_start")
    telemetry.log_event("Starting preprocessing pipeline")
    logger.info("Starting preprocessing pipeline")
    
    # Record input parameters
    telemetry.add_metric("input_path", input_path)
    telemetry.add_metric("output_path", output_path)
    telemetry.add_metric("skip_velocity", skip_velocity)
    telemetry.add_metric("use_gpu", use_gpu)
    
    if use_gpu:
        if HAS_GPU:
            telemetry.log_event("GPU acceleration enabled")
            logger.info("GPU acceleration enabled")
        else:
            telemetry.log_event("GPU acceleration requested but not available")
            logger.warning("GPU acceleration requested but required libraries not found or GPU not available")
            logger.warning("Install required packages with: pip install cupy torch")
    
    # Load raw data
    raw_df = load_raw(input_path)
    
    # Clean data
    cleaned_df = clean(raw_df)
    
    # Engineer features
    processed_df = feature_engineer(cleaned_df, skip_velocity_calculation=skip_velocity, use_gpu=use_gpu)
    
    # Save processed data
    save_processed(processed_df, output_path)
    
    # Mark pipeline completion
    telemetry.mark_timestamp("pipeline_end")
    telemetry.log_event("Preprocessing pipeline completed successfully")
    
    # Calculate total execution time
    total_time = telemetry.get_elapsed_time("pipeline_start", "pipeline_end")
    telemetry.add_metric("total_execution_time_seconds", total_time)
    telemetry.add_metric("rows_per_second", len(processed_df) / total_time if total_time > 0 else 0)
    
    # Create telemetry output path
    telemetry_filename = f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    telemetry_path = os.path.join(os.path.dirname(output_path), telemetry_filename)
    
    # Save telemetry data
    telemetry.save_to_file(telemetry_path)
    
    logger.info("Preprocessing pipeline completed successfully")
    logger.info(f"Telemetry data saved to {telemetry_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess transaction data for fraud detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process raw data
  python preprocess.py --input data/raw/transactions_raw.csv --output data/processed/transactions_processed.csv
  
  # Process raw data with GPU acceleration (requires CuPy and PyTorch)
  python preprocess.py --input data/raw/transactions_raw.csv --output data/processed/transactions_processed.csv --use-gpu
  
  # Process raw data faster (skip velocity calculation for large datasets)
  python preprocess.py --input data/raw/transactions_raw.csv --output data/processed/transactions_processed.csv --skip-velocity-calculation
  
  # If you don't have raw data, generate synthetic data first
  python data/generate_synthetic.py
  python preprocess.py --input data/raw/transactions_raw.csv --output data/processed/transactions_processed.csv
        """
    )
    parser.add_argument("--input", required=True, 
                      help="Path to raw transaction data CSV file (e.g., data/raw/transactions_raw.csv)")
    parser.add_argument("--output", required=True, 
                      help="Path to save processed data CSV file (e.g., data/processed/transactions_processed.csv)")
    parser.add_argument("--skip-velocity-calculation", action="store_true",
                      help="Skip the computationally intensive transaction velocity calculation (for large datasets)")
    parser.add_argument("--use-gpu", action="store_true",
                      help="Use GPU acceleration for computationally intensive operations (requires CuPy and PyTorch)")
    parser.add_argument("--telemetry-output", 
                      help="Path to save telemetry data JSON file (defaults to data/processed/telemetry_<timestamp>.json)")
    
    # If no arguments are provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Validate file paths
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        print("You may need to generate synthetic data first:")
        print("    python data/generate_synthetic.py")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # If custom telemetry path provided, override the default
    if args.telemetry_output:
        telemetry_dir = os.path.dirname(args.telemetry_output)
        if telemetry_dir and not os.path.exists(telemetry_dir):
            print(f"Creating telemetry directory: {telemetry_dir}")
            os.makedirs(telemetry_dir, exist_ok=True)
        
        # Run preprocessing with parameters
        main(args.input, args.output, skip_velocity=args.skip_velocity_calculation, use_gpu=args.use_gpu)
        
        # Save telemetry to custom path
        telemetry.save_to_file(args.telemetry_output)
    else:
        # Use default telemetry path from main function
        main(args.input, args.output, skip_velocity=args.skip_velocity_calculation, use_gpu=args.use_gpu)