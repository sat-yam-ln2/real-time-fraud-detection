#!/usr/bin/env python
"""
Fraud Detection Model Retraining Script

This script handles scheduled retraining of fraud detection models
with new data, evaluation, and registration.

Usage:
    python retrain.py [--config CONFIG] [--data-path DATA_PATH] [--output-dir OUTPUT_DIR]

Example:
    python retrain.py --config src/models/hyperparams.yml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime

# Add the project root to system path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.model import FraudDetectionModel
from src.utils.logger import setup_logger
from src.features.preprocess import preprocess_transactions

# Setup logging
logger = setup_logger('retrain_model')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Retrain fraud detection model')
    parser.add_argument('--config', type=str,
                        default=str(project_root / 'src' / 'models' / 'hyperparams.yml'),
                        help='Path to hyperparameter config YAML file')
    parser.add_argument('--data-path', type=str,
                        default=str(project_root / 'data' / 'processed' / 'transactions_processed.csv'),
                        help='Path to processed data CSV file')
    parser.add_argument('--raw-data-path', type=str,
                        default=str(project_root / 'data' / 'raw' / 'transactions_raw.csv'),
                        help='Path to raw data CSV file')
    parser.add_argument('--output-dir', type=str,
                        default=str(project_root / 'models_store' / 'v1'),
                        help='Directory to save model and evaluation results')
    parser.add_argument('--model-name', type=str, default='fraud_model',
                        help='Name to register the model under')
    parser.add_argument('--register-model', action='store_true',
                        help='Whether to register the model after training')
    parser.add_argument('--promote-to-production', action='store_true',
                        help='Whether to promote the model to production after training')
    parser.add_argument('--force-preprocess', action='store_true',
                        help='Force preprocessing of raw data even if processed data exists')
    
    return parser.parse_args()


def load_config(config_path):
    """
    Load hyperparameter configuration.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary with configuration parameters
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def check_and_preprocess_data(raw_data_path, processed_data_path, force_preprocess=False):
    """
    Check if processed data exists and preprocess if needed.
    
    Args:
        raw_data_path: Path to raw data
        processed_data_path: Path to processed data
        force_preprocess: Force preprocessing even if processed data exists
        
    Returns:
        Path to processed data
    """
    if not force_preprocess and os.path.exists(processed_data_path):
        # Check if processed data is older than raw data
        raw_mtime = os.path.getmtime(raw_data_path) if os.path.exists(raw_data_path) else 0
        processed_mtime = os.path.getmtime(processed_data_path)
        
        if processed_mtime > raw_mtime:
            logger.info(f"Using existing processed data at {processed_data_path}")
            return processed_data_path
    
    # Preprocess data
    if not os.path.exists(raw_data_path):
        logger.error(f"Raw data not found at {raw_data_path}")
        sys.exit(1)
    
    logger.info(f"Preprocessing data from {raw_data_path} to {processed_data_path}")
    try:
        # Create directory for processed data if it doesn't exist
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        
        # Load raw data
        raw_df = pd.read_csv(raw_data_path)
        
        # Preprocess data
        processed_df = preprocess_transactions(raw_df)
        
        # Save processed data
        processed_df.to_csv(processed_data_path, index=False)
        
        logger.info(f"Data preprocessed and saved to {processed_data_path}")
        return processed_data_path
    
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        sys.exit(1)


def setup_output_directory(output_dir):
    """
    Setup output directory for model and evaluation results.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        Paths to model file and evaluation directory
    """
    # Create directories
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Create evaluation directory
    eval_dir = output_path / 'evaluation'
    os.makedirs(eval_dir, exist_ok=True)
    
    # Define file paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = output_path / f'fraud_model_{timestamp}.pkl'
    
    return str(model_path), str(eval_dir)


def run_training_pipeline(config, data_path, model_path):
    """
    Run the training pipeline.
    
    Args:
        config: Dictionary with configuration parameters
        data_path: Path to processed data
        model_path: Path to save the trained model
        
    Returns:
        Trained model instance
    """
    from src.models.train import train_model
    
    logger.info("Running training pipeline")
    
    # Extract training parameters from config
    lightgbm_params = config.get('lightgbm', {})
    training_params = config.get('training', {})
    
    # Run training
    model = train_model(
        data_path=data_path,
        model_path=model_path,
        model_params=lightgbm_params,
        test_size=training_params.get('test_size', 0.2),
        random_state=training_params.get('random_state', 42)
    )
    
    logger.info(f"Model trained and saved to {model_path}")
    
    return model


def run_evaluation_pipeline(model, model_path, data_path, eval_dir, config):
    """
    Run the evaluation pipeline.
    
    Args:
        model: Trained model instance
        model_path: Path to trained model
        data_path: Path to processed data
        eval_dir: Directory to save evaluation results
        config: Dictionary with configuration parameters
        
    Returns:
        Path to evaluation metrics file
    """
    import subprocess
    
    logger.info("Running evaluation pipeline")
    
    # Extract evaluation parameters from config
    eval_params = config.get('evaluation', {})
    training_params = config.get('training', {})
    
    # Build command for evaluate.py
    cmd = [
        sys.executable,
        str(project_root / 'src' / 'models' / 'evaluate.py'),
        f'--model-path={model_path}',
        f'--data-path={data_path}',
        f'--output-dir={eval_dir}',
        f'--test-size={training_params.get("test_size", 0.2)}',
        f'--random-state={training_params.get("random_state", 42)}',
    ]
    
    if eval_params.get('threshold_search', False):
        cmd.append('--threshold-search')
    else:
        cmd.append(f'--threshold={eval_params.get("default_threshold", 0.5)}')
    
    # Run evaluation process
    logger.info(f"Running evaluation command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info("Evaluation completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
    
    # Return path to evaluation metrics
    return str(Path(eval_dir) / 'evaluation_metrics.json')


def register_trained_model(model_path, model_name, eval_metrics_path, promote_to_production):
    """
    Register trained model.
    
    Args:
        model_path: Path to trained model
        model_name: Name to register the model under
        eval_metrics_path: Path to evaluation metrics
        promote_to_production: Whether to promote to production
    """
    import subprocess
    
    logger.info(f"Registering model {model_name}")
    
    # Build command for register_model.py
    cmd = [
        sys.executable,
        str(project_root / 'src' / 'models' / 'register_model.py'),
        f'--model-path={model_path}',
        f'--model-name={model_name}',
    ]
    
    if eval_metrics_path:
        cmd.append(f'--evaluation-path={eval_metrics_path}')
    
    description = f"Model retrained on {datetime.now().strftime('%Y-%m-%d')} with updated data"
    cmd.append(f'--description={description}')
    
    if promote_to_production:
        cmd.append('--promote-to-production')
    
    # Run registration process
    logger.info(f"Running registration command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info("Model registered successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Model registration failed: {e}")
        sys.exit(1)


def main():
    """Main function for model retraining."""
    args = parse_args()
    
    logger.info("Starting model retraining process")
    
    # Load configuration
    config = load_config(args.config)
    
    # Check and preprocess data if needed
    data_path = check_and_preprocess_data(
        args.raw_data_path,
        args.data_path,
        args.force_preprocess
    )
    
    # Setup output directory
    model_path, eval_dir = setup_output_directory(args.output_dir)
    
    # Run training pipeline
    model = run_training_pipeline(config, data_path, model_path)
    
    # Run evaluation pipeline
    eval_metrics_path = run_evaluation_pipeline(model, model_path, data_path, eval_dir, config)
    
    # Register model if requested
    if args.register_model:
        register_trained_model(
            model_path,
            args.model_name,
            eval_metrics_path,
            args.promote_to_production
        )
    
    logger.info("Model retraining completed successfully")


if __name__ == "__main__":
    main()
