#!/usr/bin/env python
"""
Fraud Detection Model Training Script

This script loads preprocessed transaction data, trains a fraud detection model,
and saves the trained model to disk.

Usage:
    python train.py [--config CONFIG_PATH] [--data-path DATA_PATH] [--model-output MODEL_OUTPUT]

Example:
    python train.py --config hyperparams.yml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Add the project root to system path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.model import FraudDetectionModel, load_hyperparameters
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger('train')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to hyperparameters configuration file')
    parser.add_argument('--data-path', type=str, 
                        default=str(project_root / 'data' / 'processed' / 'transactions_processed.csv'),
                        help='Path to processed data CSV file')
    parser.add_argument('--model-output', type=str,
                        default=str(project_root / 'models_store' / 'v1' / 'fraud_v1.pkl'),
                        help='Path to save the trained model')
    parser.add_argument('--plots-dir', type=str,
                        default=str(project_root / 'models_store' / 'v1' / 'plots'),
                        help='Directory to save model evaluation plots')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for test set')
    parser.add_argument('--val-size', type=float, default=0.25,
                        help='Proportion of train data to use for validation set')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--undersample', action='store_true',
                        help='Whether to use undersampling to balance classes')
    parser.add_argument('--undersample-ratio', type=float, default=0.05,
                        help='Ratio of negative to positive samples when undersampling')
    
    return parser.parse_args()


def load_data(data_path):
    """
    Load preprocessed transaction data.
    
    Args:
        data_path: Path to processed data CSV file
        
    Returns:
        Pandas DataFrame with transaction data
    """
    logger.info(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def preprocess_for_training(df, target_column='Class'):
    """
    Prepare data for model training.
    
    Args:
        df: DataFrame with transaction data
        target_column: Name of the target column
        
    Returns:
        Features DataFrame and target Series
    """
    logger.info("Preprocessing data for training")
    
    # Check if target column exists
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in data")
        sys.exit(1)
    
    # Extract target
    y = df[target_column]
    
    # Drop target and any non-feature columns
    non_feature_cols = [
        target_column,
        'transaction_datetime',  # datetime object not usable for training
        'day_name',              # categorical string, already encoded as day_of_week
    ]
    
    X = df.drop([col for col in non_feature_cols if col in df.columns], axis=1)
    
    # Handle any remaining non-numeric columns
    object_columns = X.select_dtypes(include=['object']).columns
    if len(object_columns) > 0:
        logger.warning(f"Dropping non-numeric columns: {list(object_columns)}")
        X = X.drop(object_columns, axis=1)
    
    logger.info(f"Processed features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Class distribution: {dict(y.value_counts())}")
    
    return X, y


def perform_train_test_split(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion to use for test set
        val_size: Proportion of train data to use for validation
        random_state: Random state for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Splitting data into train, validation, and test sets")
    
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Then split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, 
        random_state=random_state, stratify=y_train_val
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, "
                f"Validation set: {X_val.shape[0]} samples, "
                f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def undersample_majority_class(X_train, y_train, ratio=0.1, random_state=42):
    """
    Undersample the majority class to reduce class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        ratio: Ratio of negative to positive samples
        random_state: Random state for reproducibility
        
    Returns:
        Resampled X_train and y_train
    """
    logger.info("Undersampling majority class")
    
    # Separate majority and minority classes
    X_majority = X_train[y_train == 0]
    X_minority = X_train[y_train == 1]
    y_majority = y_train[y_train == 0]
    y_minority = y_train[y_train == 1]
    
    # Calculate number of majority samples to keep
    n_minority = len(X_minority)
    n_majority = int(n_minority / ratio) if ratio > 0 else len(X_majority)
    n_majority = min(n_majority, len(X_majority))
    
    logger.info(f"Original distribution: {len(X_majority)} majority, {n_minority} minority")
    logger.info(f"After undersampling: {n_majority} majority, {n_minority} minority")
    
    # Undersample majority class
    X_majority_undersampled = resample(
        X_majority, 
        n_samples=n_majority,
        random_state=random_state,
        replace=False
    )
    y_majority_undersampled = resample(
        y_majority, 
        n_samples=n_majority,
        random_state=random_state,
        replace=False
    )
    
    # Combine minority class with undersampled majority class
    X_train_resampled = pd.concat([X_majority_undersampled, X_minority])
    y_train_resampled = pd.concat([y_majority_undersampled, y_minority])
    
    # Shuffle the data
    temp_data = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    temp_data = temp_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split back into features and target
    target_col = y_train.name
    X_train_resampled = temp_data.drop(target_col, axis=1)
    y_train_resampled = temp_data[target_col]
    
    logger.info(f"Resampled training data shape: {X_train_resampled.shape}")
    
    return X_train_resampled, y_train_resampled


def identify_categorical_features(X, threshold=10):
    """
    Identify categorical features based on unique value counts.
    
    Args:
        X: Features DataFrame
        threshold: Maximum unique values to consider a column categorical
        
    Returns:
        List of categorical feature names
    """
    categorical_features = []
    
    for col in X.columns:
        # Skip ID columns which might have low cardinality but aren't truly categorical
        if 'id' in col.lower() and X[col].nunique() > threshold:
            continue
        
        # Consider columns with few unique values as categorical
        if X[col].nunique() <= threshold:
            categorical_features.append(col)
    
    logger.info(f"Identified {len(categorical_features)} categorical features: {categorical_features}")
    return categorical_features


def save_model_card(model, metrics, file_path):
    """
    Save model card with metadata and evaluation metrics.
    
    Args:
        model: Trained FraudDetectionModel instance
        metrics: Dictionary of evaluation metrics
        file_path: Path to save model card JSON
    """
    model_card = model.get_model_card(metrics)
    
    # Add additional metadata
    model_card['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_card['data_split'] = {
        'train_size': metrics.get('train_size'),
        'val_size': metrics.get('val_size'),
        'test_size': metrics.get('test_size'),
    }
    
    # Save as JSON
    with open(file_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    
    logger.info(f"Model card saved to {file_path}")


def save_plots(model, X_test, y_test, plots_dir):
    """
    Generate and save model evaluation plots.
    
    Args:
        model: Trained FraudDetectionModel instance
        X_test: Test features
        y_test: Test labels
        plots_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plots
    try:
        # Feature importance plot
        fig_importance = model.plot_feature_importance(top_n=20)
        fig_importance.savefig(os.path.join(plots_dir, 'feature_importance.png'), bbox_inches='tight')
        
        # ROC curve
        fig_roc = model.plot_roc_curve(X_test, y_test)
        fig_roc.savefig(os.path.join(plots_dir, 'roc_curve.png'), bbox_inches='tight')
        
        # Precision-Recall curve
        fig_pr = model.plot_precision_recall_curve(X_test, y_test)
        fig_pr.savefig(os.path.join(plots_dir, 'precision_recall_curve.png'), bbox_inches='tight')
        
        logger.info(f"Evaluation plots saved to {plots_dir}")
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
    
    # Close all plot windows
    plt.close('all')


def update_registry(model_info):
    """
    Update the model registry with information about the newly trained model.
    
    Args:
        model_info: Dictionary with model metadata
    """
    registry_path = project_root / 'models_store' / 'registry.json'
    
    # Initialize or load registry
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"models": []}
    
    # Add new model info
    registry["models"].append(model_info)
    
    # Save updated registry
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Model registry updated at {registry_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("Starting model training process")
    
    # Load data
    data_path = Path(args.data_path)
    df = load_data(data_path)
    
    # Preprocess data
    X, y = preprocess_for_training(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = perform_train_test_split(
        X, y, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state
    )
    
    # Undersample majority class if requested
    if args.undersample:
        X_train, y_train = undersample_majority_class(
            X_train, y_train, ratio=args.undersample_ratio, random_state=args.random_state
        )
    
    # Identify categorical features
    categorical_features = identify_categorical_features(X_train)
    
    # Load hyperparameters
    config_path = args.config
    hyperparams = load_hyperparameters(config_path)
    logger.info(f"Using hyperparameters: {hyperparams}")
    
    # Initialize and train model
    model = FraudDetectionModel(config=hyperparams)
    training_history = model.train(
        X_train, y_train, X_val, y_val, categorical_features=categorical_features
    )
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    metrics = model.evaluate(X_test, y_test)
    metrics['train_size'] = X_train.shape[0]
    metrics['val_size'] = X_val.shape[0]
    metrics['test_size'] = X_test.shape[0]
    
    # Log key metrics
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    
    # Save model
    model_output = Path(args.model_output)
    os.makedirs(model_output.parent, exist_ok=True)
    model.save(str(model_output))
    
    # Save model card
    model_card_path = model_output.parent / 'model_card.json'
    save_model_card(model, metrics, model_card_path)
    
    # Save evaluation plots
    save_plots(model, X_test, y_test, args.plots_dir)
    
    # Update model registry
    model_info = {
        'version': 'v1',
        'path': str(model_output.relative_to(project_root)),
        'metrics': {
            'roc_auc': metrics['roc_auc'],
            'pr_auc': metrics['pr_auc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    update_registry(model_info)
    
    logger.info("Model training and evaluation completed successfully")


if __name__ == "__main__":
    main()
