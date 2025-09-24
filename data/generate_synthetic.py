#!/usr/bin/env python3
"""
Synthetic Transaction Data Generator

This script generates synthetic financial transaction data for testing and development
of the real-time fraud detection system. It can create data in batch mode or simulate
a continuous stream of transactions with controllable fraud rates, transaction volumes,
and time patterns.

Usage:
    python generate_synthetic.py [options]

Options:
    --mode [batch|stream]    Output mode (default: batch)
    --count N                Number of transactions to generate in batch mode (default: 10000)
    --fraud-rate X           Percentage of fraudulent transactions (default: 0.2)
    --output PATH            Output file path for batch mode (default: transactions_raw.csv)
    --rate N                 Transactions per minute in stream mode (default: 60)
    --distribution [uniform|poisson|daily-pattern]
                             Transaction timing distribution (default: daily-pattern)
    --user-count N           Number of unique users to simulate (default: 1000)
    --merchant-count N       Number of unique merchants to simulate (default: 500)
    --seed N                 Random seed for reproducibility
    --include-anomalies      Include anomalous transaction patterns (default: False)
"""

import argparse
import json
import numpy as np
import pandas as pd
import os
import time
import sys
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path


class TransactionGenerator:
    """Generate synthetic transaction data with controllable fraud patterns."""
    
    def __init__(
        self,
        user_count=1000,
        merchant_count=500,
        fraud_rate=0.002,  # 0.2% fraud by default
        include_anomalies=False,
        random_seed=None
    ):
        """
        Initialize the transaction generator.
        
        Args:
            user_count: Number of unique users to simulate
            merchant_count: Number of unique merchants to simulate
            fraud_rate: Percentage of fraudulent transactions (0-1)
            include_anomalies: Whether to include anomalous transaction patterns
            random_seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        self.user_count = user_count
        self.merchant_count = merchant_count
        self.fraud_rate = fraud_rate
        self.include_anomalies = include_anomalies
        
        # Generate synthetic user and merchant data
        self.users = self._generate_users()
        self.merchants = self._generate_merchants()
        
        # Prepare PCA components for generating V1-V28 features
        self.pca_components = self._initialize_pca_components()
        
        # Transaction counter for reference
        self.transaction_count = 0
        
        # Starting timestamp (will be used as a reference for the 'Time' field)
        self.start_time = datetime.now()
        
        # For stream mode, keep track of the last transaction time
        self.last_transaction_time = self.start_time
        
        print(f"Initialized transaction generator with {user_count} users, "
              f"{merchant_count} merchants, and {fraud_rate*100:.2f}% fraud rate")
    
    def _generate_users(self):
        """Generate synthetic user profiles."""
        users = []
        for i in range(self.user_count):
            # Generate a random user profile with realistic spending patterns
            user = {
                'user_id': str(uuid.uuid4()),
                'avg_amount': np.random.gamma(shape=5.0, scale=20.0),  # Average transaction amount
                'std_amount': np.random.gamma(shape=2.0, scale=10.0),  # Std dev of transaction amount
                'activity_level': np.random.gamma(shape=2.0, scale=1.0),  # Relative frequency of transactions
                'fraud_prone': np.random.random() < 0.05,  # 5% of users are more likely to experience fraud
                'merchant_categories': np.random.choice(
                    range(self.merchant_count), 
                    size=np.random.randint(1, 10), 
                    replace=False
                ).tolist(),  # Categories user typically shops at
                'active_hours': self._generate_active_hours(),  # Hours when user typically transacts
                'weekend_activity': np.random.random(),  # Higher values = more weekend transactions
            }
            users.append(user)
        return users
    
    def _generate_active_hours(self):
        """Generate a probability distribution of user's active hours."""
        active_hours = np.zeros(24)
        
        # Most users have 1-3 peak activity times
        peak_count = np.random.randint(1, 4)
        peak_hours = np.random.choice(24, size=peak_count, replace=False)
        
        # Set peak hours with high probability
        for hour in peak_hours:
            active_hours[hour] = np.random.uniform(0.7, 1.0)
            
            # Add some probability to adjacent hours (with periodic boundary conditions)
            active_hours[(hour - 1) % 24] += np.random.uniform(0.3, 0.6)
            active_hours[(hour + 1) % 24] += np.random.uniform(0.3, 0.6)
        
        # Add small random probability to all hours
        active_hours += np.random.uniform(0, 0.2, size=24)
        
        # Normalize to create a probability distribution
        return active_hours / active_hours.sum()
    
    def _generate_merchants(self):
        """Generate synthetic merchant profiles."""
        merchants = []
        for i in range(self.merchant_count):
            # Different merchant categories have different fraud rates
            base_fraud_rate = np.random.beta(0.5, 50) * 0.05  # Max 5% base fraud rate
            
            merchant = {
                'merchant_id': str(uuid.uuid4()),
                'category': np.random.randint(0, 20),  # 20 different merchant categories
                'avg_transaction': np.random.gamma(shape=5.0, scale=30.0),  # Average transaction size
                'fraud_rate': base_fraud_rate,  # Base fraud rate for this merchant
                'peak_hours': np.random.choice(24, size=np.random.randint(1, 5), replace=False).tolist(),
                'weekend_activity': np.random.random() < 0.7,  # 70% of merchants more active on weekends
            }
            merchants.append(merchant)
        return merchants
    
    def _initialize_pca_components(self):
        """
        Initialize PCA components for generating V1-V28 features.
        
        Returns a dictionary with components for normal and fraudulent transactions.
        """
        # For normal transactions, we'll use means and standard deviations
        # that will generate values similar to the original dataset
        normal_components = {
            'means': np.zeros(28),
            'stds': np.ones(28),
            'correlations': np.eye(28)  # Start with no correlations
        }
        
        # Introduce some correlations between features
        for i in range(5):
            # Pick two random features to correlate
            idx1, idx2 = np.random.choice(28, size=2, replace=False)
            strength = np.random.uniform(0.3, 0.7) * (1 if np.random.random() < 0.5 else -1)
            normal_components['correlations'][idx1, idx2] = strength
            normal_components['correlations'][idx2, idx1] = strength
        
        # For fraudulent transactions, we'll shift the means and adjust the standard deviations
        # to make them statistically different from normal transactions
        fraud_components = {
            'means': np.random.normal(0, 0.5, size=28),  # Small shifts in means
            'stds': np.random.uniform(0.7, 1.3, size=28),  # Some variance changes
            'correlations': normal_components['correlations'].copy()
        }
        
        # Introduce some additional correlations for fraud cases
        for i in range(8):
            # Pick two random features to correlate
            idx1, idx2 = np.random.choice(28, size=2, replace=False)
            strength = np.random.uniform(0.4, 0.8) * (1 if np.random.random() < 0.5 else -1)
            fraud_components['correlations'][idx1, idx2] = strength
            fraud_components['correlations'][idx2, idx1] = strength
        
        # Create specific feature shifts for fraud (based on domain knowledge)
        # These will make certain features more indicative of fraud
        fraud_indicators = np.random.choice(28, size=5, replace=False)
        for idx in fraud_indicators:
            fraud_components['means'][idx] = np.random.uniform(-1.5, 1.5)
            fraud_components['stds'][idx] = np.random.uniform(1.2, 1.8)
        
        return {
            'normal': normal_components,
            'fraud': fraud_components
        }
    
    def _generate_v_features(self, is_fraud):
        """
        Generate the V1-V28 features for a transaction.
        
        Args:
            is_fraud: Boolean indicating if this is a fraudulent transaction
            
        Returns:
            Array of V1-V28 values
        """
        components = self.pca_components['fraud'] if is_fraud else self.pca_components['normal']
        
        # Generate random normal values
        features = np.random.normal(0, 1, size=28)
        
        # Apply the correlation structure using Cholesky decomposition
        # This ensures the features have the desired correlation structure
        chol = np.linalg.cholesky(components['correlations'] @ components['correlations'].T + 0.01 * np.eye(28))
        features = np.dot(chol, features)
        
        # Scale by standard deviations and add means
        features = features * components['stds'] + components['means']
        
        # For fraudulent transactions, introduce some anomalies occasionally
        if is_fraud and self.include_anomalies and np.random.random() < 0.3:
            # Extreme values for a random subset of features
            anomaly_idx = np.random.choice(28, size=np.random.randint(1, 4), replace=False)
            features[anomaly_idx] = np.random.uniform(-5, 5, size=len(anomaly_idx))
        
        return features
    
    def _get_transaction_time(self, distribution='daily-pattern', rate=60):
        """
        Get the next transaction time based on the specified distribution.
        
        Args:
            distribution: The distribution to use ('uniform', 'poisson', or 'daily-pattern')
            rate: Transactions per minute for Poisson distribution
            
        Returns:
            The next transaction time as a datetime object
        """
        now = datetime.now()
        seconds_elapsed = (now - self.start_time).total_seconds()
        
        if distribution == 'uniform':
            # Uniform distribution - constant rate
            delta = 60.0 / rate
            next_time = self.last_transaction_time + timedelta(seconds=delta)
        
        elif distribution == 'poisson':
            # Poisson distribution - varying intervals with specified average rate
            interval = np.random.exponential(60.0 / rate)
            next_time = self.last_transaction_time + timedelta(seconds=interval)
        
        else:  # daily-pattern
            # Model the daily transaction pattern with peak hours
            hour_now = now.hour
            
            # Base rate multipliers for each hour (approximate real-world transaction patterns)
            hourly_pattern = np.array([
                0.2, 0.1, 0.05, 0.05, 0.05, 0.1,     # 0-5: Night (low activity)
                0.3, 0.8, 1.5, 1.8, 1.5, 1.2,        # 6-11: Morning to noon
                1.5, 1.7, 1.5, 1.3, 1.5, 2.0,        # 12-17: Afternoon
                2.5, 2.2, 1.8, 1.5, 1.0, 0.5         # 18-23: Evening to night
            ])
            
            # Weekend adjustment
            if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                # Different pattern on weekends
                hourly_pattern[6:12] *= 0.6  # Less morning activity
                hourly_pattern[12:18] *= 1.5  # More afternoon activity
                hourly_pattern[18:23] *= 1.3  # More evening activity
            
            # Calculate the current rate based on time of day
            current_rate = rate * hourly_pattern[hour_now]
            
            # Generate interval using Poisson with time-dependent rate
            if current_rate > 0:
                interval = np.random.exponential(60.0 / current_rate)
            else:
                interval = 60.0  # Default to once per minute if rate is zero
                
            next_time = self.last_transaction_time + timedelta(seconds=interval)
        
        # Update last transaction time
        self.last_transaction_time = next_time
        
        return next_time
    
    def _select_user_and_merchant(self, is_fraud):
        """
        Select a user and merchant for a transaction.
        
        Args:
            is_fraud: Boolean indicating if this is a fraudulent transaction
            
        Returns:
            Tuple of (user, merchant)
        """
        if is_fraud and self.include_anomalies:
            # For fraud, we might select users who are more fraud-prone
            fraud_prone_users = [u for u in self.users if u['fraud_prone']]
            if fraud_prone_users and np.random.random() < 0.7:
                user = random.choice(fraud_prone_users)
            else:
                user = random.choice(self.users)
                
            # For fraud, we might select merchants with higher fraud rates
            high_risk_merchants = [m for m in self.merchants 
                                 if m['fraud_rate'] > np.median([m['fraud_rate'] for m in self.merchants])]
            if high_risk_merchants and np.random.random() < 0.7:
                merchant = random.choice(high_risk_merchants)
            else:
                merchant = random.choice(self.merchants)
        else:
            # For normal transactions, select random user
            user = random.choice(self.users)
            
            # For normal transactions, prefer merchants from the user's preferred categories
            if user['merchant_categories'] and np.random.random() < 0.8:
                # Find merchants in user's preferred categories
                preferred_merchants = [m for m in self.merchants 
                                     if m['category'] in user['merchant_categories']]
                if preferred_merchants:
                    merchant = random.choice(preferred_merchants)
                else:
                    merchant = random.choice(self.merchants)
            else:
                merchant = random.choice(self.merchants)
        
        return user, merchant
    
    def _generate_amount(self, user, merchant, is_fraud, timestamp):
        """
        Generate a transaction amount based on user and merchant profiles.
        
        Args:
            user: User profile dictionary
            merchant: Merchant profile dictionary
            is_fraud: Boolean indicating if this is a fraudulent transaction
            timestamp: Transaction timestamp
            
        Returns:
            Transaction amount
        """
        # Base amount is influenced by both user's spending habits and merchant's average transaction
        base_amount = (user['avg_amount'] + merchant['avg_transaction']) / 2
        
        # Add some variability
        variability = (user['std_amount'] + 5) / 2
        
        if is_fraud and self.include_anomalies:
            # Fraud can be either small testing transactions or unusual large amounts
            if np.random.random() < 0.3:
                # Small "test" transaction
                amount = np.random.uniform(0.1, 5)
            else:
                # Unusual large amount (potentially multiple std devs above average)
                multiplier = np.random.uniform(1.5, 5)
                amount = base_amount * multiplier
                
                # Sometimes add cents to make it look less round (psychology of fraud)
                if np.random.random() < 0.7:
                    amount += np.random.uniform(0.01, 0.99)
        else:
            # Normal transaction with user's typical variability
            amount = np.random.normal(base_amount, variability)
            
            # Ensure amount is positive
            amount = max(0.01, amount)
            
            # Round to two decimal places (cents)
            amount = round(amount, 2)
        
        return amount
    
    def generate_transaction(self, timestamp=None, is_fraud=None):
        """
        Generate a single synthetic transaction.
        
        Args:
            timestamp: Optional specific timestamp for the transaction
            is_fraud: Optional boolean to force fraud/non-fraud (if None, uses fraud_rate)
            
        Returns:
            Dictionary with transaction data
        """
        # Increment transaction counter
        self.transaction_count += 1
        
        # Determine if this transaction is fraudulent if not specified
        if is_fraud is None:
            is_fraud = np.random.random() < self.fraud_rate
        
        # Get timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now()
        
        # Select user and merchant
        user, merchant = self._select_user_and_merchant(is_fraud)
        
        # Generate transaction amount
        amount = self._generate_amount(user, merchant, is_fraud, timestamp)
        
        # Generate V1-V28 features
        v_features = self._generate_v_features(is_fraud)
        
        # Time in seconds from the start
        time_seconds = (timestamp - self.start_time).total_seconds()
        
        # Create transaction record
        transaction = {
            'Time': time_seconds,
            'Amount': amount,
            'Class': 1 if is_fraud else 0,
            'user_id': user['user_id'],
            'merchant_id': merchant['merchant_id'],
            'merchant_category': merchant['category'],
            'timestamp': timestamp.isoformat(),
        }
        
        # Add V1-V28 features
        for i, v in enumerate(v_features):
            transaction[f'V{i+1}'] = v
        
        return transaction
    
    def generate_batch(self, count=10000, output_path=None, fraud_rate=None):
        """
        Generate a batch of synthetic transactions.
        
        Args:
            count: Number of transactions to generate
            output_path: Path to save the CSV file (if None, returns DataFrame)
            fraud_rate: Optional override for the instance's fraud_rate
            
        Returns:
            DataFrame with the generated transactions if output_path is None,
            otherwise None (data is saved to file)
        """
        print(f"Generating {count} synthetic transactions...")
        
        # Use provided fraud_rate if specified
        actual_fraud_rate = fraud_rate if fraud_rate is not None else self.fraud_rate
        
        # Pre-determine which transactions will be fraudulent
        fraud_indices = np.random.choice(
            count, 
            size=int(count * actual_fraud_rate),
            replace=False
        )
        fraud_flags = np.zeros(count, dtype=bool)
        fraud_flags[fraud_indices] = True
        
        transactions = []
        start_time = datetime.now()
        
        for i in range(count):
            # Progress indicator every 10%
            if i % (count // 10) == 0 and i > 0:
                progress = i / count * 100
                print(f"{progress:.1f}% complete ({i}/{count})")
            
            is_fraud = fraud_flags[i]
            
            # For batch mode, simulate transactions over a period (e.g., 30 days)
            # This makes the 'Time' field meaningful
            seconds_offset = np.random.uniform(0, 30 * 24 * 3600)  # 30 days in seconds
            timestamp = start_time + timedelta(seconds=seconds_offset)
            
            transaction = self.generate_transaction(timestamp=timestamp, is_fraud=is_fraud)
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # Ensure we have the exact fraud rate requested
        actual_fraud_count = df['Class'].sum()
        print(f"Generated {len(df)} transactions with {actual_fraud_count} fraudulent "
              f"({actual_fraud_count/len(df)*100:.2f}% fraud rate)")
        
        # Save to CSV if output_path is provided
        if output_path:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            print(f"Saving to {output_path}")
            
            # Save only the columns required for the fraud detection model
            # Standard columns: Time, V1-V28, Amount, Class
            standard_columns = ['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount', 'Class']
            
            # Ensure all standard columns exist (with empty values if needed)
            for col in standard_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Save only standard columns to CSV
            df[standard_columns].to_csv(output_path, index=False)
            
            # Save full data with extra fields to JSON for reference
            json_path = output_path.replace('.csv', '_full.json')
            with open(json_path, 'w') as f:
                json.dump(df.to_dict(orient='records'), f, indent=2)
            
            print(f"Saved {len(df)} transactions to {output_path}")
            
            # Also save a sample payload for API testing
            sample_payload_dir = Path(output_dir).parent / 'sample_payloads'
            sample_payload_dir.mkdir(exist_ok=True)
            
            sample_payload = {
                "transaction": df.iloc[0].to_dict()
            }
            
            sample_payload_path = sample_payload_dir / 'sample_payload.json'
            with open(sample_payload_path, 'w') as f:
                json.dump(sample_payload, f, indent=2)
            
            print(f"Saved sample API payload to {sample_payload_path}")
            
            return None
        else:
            return df
    
    def start_stream(self, rate=60, distribution='daily-pattern', output=None, duration=None):
        """
        Start generating a continuous stream of transactions.
        
        Args:
            rate: Base transactions per minute
            distribution: Transaction timing distribution 
                          ('uniform', 'poisson', or 'daily-pattern')
            output: Output method - either 'stdout', 'file:/path/to/file', 'kafka:topic', or None
            duration: Optional duration in seconds to run the stream
            
        Returns:
            None - this method runs until interrupted or duration is reached
        """
        print(f"Starting transaction stream ({distribution} distribution) "
              f"at ~{rate} transactions per minute")
        
        # Reset the last transaction time to now
        self.last_transaction_time = datetime.now()
        transaction_count = 0
        start_time = datetime.now()
        
        try:
            while True:
                # Check if duration has elapsed
                if duration is not None:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= duration:
                        print(f"Stream duration of {duration}s elapsed. Stopping.")
                        break
                
                # Get the next transaction time based on the distribution
                next_time = self._get_transaction_time(distribution, rate)
                
                # Sleep until the next transaction time
                sleep_seconds = (next_time - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                
                # Generate a transaction
                transaction = self.generate_transaction(timestamp=next_time)
                transaction_count += 1
                
                # Output the transaction
                if output is None:
                    # Just count transactions
                    if transaction_count % 10 == 0:
                        print(f"Generated {transaction_count} transactions...")
                
                elif output == 'stdout':
                    # Print to stdout as JSON
                    print(json.dumps(transaction))
                    sys.stdout.flush()
                
                elif output.startswith('file:'):
                    # Append to a file
                    file_path = output[5:]
                    with open(file_path, 'a') as f:
                        f.write(json.dumps(transaction) + '\n')
                    
                    if transaction_count % 10 == 0:
                        print(f"Generated {transaction_count} transactions to {file_path}")
                
                elif output.startswith('kafka:'):
                    # TODO: Implement Kafka producer if needed
                    topic = output[6:]
                    print(f"Kafka output not implemented yet. Would send to topic {topic}")
                
                else:
                    print(f"Unknown output method: {output}")
                    break
                
        except KeyboardInterrupt:
            print("\nStream interrupted. Shutting down.")
        
        print(f"Stream stopped after generating {transaction_count} transactions.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic transaction data for fraud detection testing'
    )
    
    parser.add_argument('--mode', choices=['batch', 'stream'], default='batch',
                        help='Output mode (batch or stream)')
    
    parser.add_argument('--count', type=int, default=10000,
                        help='Number of transactions to generate in batch mode')
    
    parser.add_argument('--fraud-rate', type=float, default=0.002,
                        help='Percentage of fraudulent transactions (default: 0.002 = 0.2%%)')
    
    parser.add_argument('--output', type=str, 
                        default='../data/raw/transactions_raw.csv',
                        help='Output file path for batch mode')
    
    parser.add_argument('--rate', type=int, default=60,
                        help='Transactions per minute in stream mode')
    
    parser.add_argument('--distribution', 
                        choices=['uniform', 'poisson', 'daily-pattern'],
                        default='daily-pattern',
                        help='Transaction timing distribution')
    
    parser.add_argument('--user-count', type=int, default=1000,
                        help='Number of unique users to simulate')
    
    parser.add_argument('--merchant-count', type=int, default=500,
                        help='Number of unique merchants to simulate')
    
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    parser.add_argument('--include-anomalies', action='store_true',
                        help='Include anomalous transaction patterns')
    
    parser.add_argument('--duration', type=int, help='Duration to run the stream (seconds)')
    
    parser.add_argument('--stream-output', type=str, choices=['stdout', 'file', 'kafka', 'none'],
                        default='none', help='Stream output method')
    
    parser.add_argument('--stream-target', type=str, 
                        help='Target for stream output (file path, Kafka topic)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.fraud_rate < 0 or args.fraud_rate > 1:
        parser.error("Fraud rate must be between 0 and 1")
    
    # Resolve output path for batch mode
    if args.mode == 'batch' and args.output:
        # Handle relative paths
        if not os.path.isabs(args.output):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            args.output = os.path.join(script_dir, args.output)
    
    # Resolve stream output target
    if args.mode == 'stream' and args.stream_output != 'none' and args.stream_output != 'stdout':
        if not args.stream_target:
            parser.error(f"--stream-target required for {args.stream_output} output")
        
        # Construct full output descriptor
        if args.stream_output == 'file':
            # Handle relative paths for file output
            if not os.path.isabs(args.stream_target):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                args.stream_target = os.path.join(script_dir, args.stream_target)
            args.full_stream_output = f"file:{args.stream_target}"
        elif args.stream_output == 'kafka':
            args.full_stream_output = f"kafka:{args.stream_target}"
        else:
            args.full_stream_output = args.stream_output
    else:
        args.full_stream_output = args.stream_output if args.stream_output == 'stdout' else None
    
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    print(f"Synthetic Transaction Generator")
    print(f"-------------------------------")
    print(f"Mode: {args.mode}")
    
    generator = TransactionGenerator(
        user_count=args.user_count,
        merchant_count=args.merchant_count,
        fraud_rate=args.fraud_rate,
        include_anomalies=args.include_anomalies,
        random_seed=args.seed
    )
    
    if args.mode == 'batch':
        print(f"Generating {args.count} transactions with {args.fraud_rate*100:.2f}% fraud rate")
        generator.generate_batch(
            count=args.count,
            output_path=args.output,
        )
    else:  # stream mode
        print(f"Starting stream at {args.rate} transactions/minute "
              f"with {args.fraud_rate*100:.2f}% fraud rate")
        print(f"Distribution: {args.distribution}")
        if args.duration:
            print(f"Duration: {args.duration}s")
        
        generator.start_stream(
            rate=args.rate,
            distribution=args.distribution,
            output=args.full_stream_output,
            duration=args.duration
        )