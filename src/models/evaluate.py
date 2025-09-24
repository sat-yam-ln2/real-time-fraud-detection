#!/usr/bin/env python
"""
Fraud Detection Model Evaluation Script

This script evaluates a trained fraud detection model on test data and
generates comprehensive evaluation metrics and visualizations.

Usage:
    python evaluate.py [--model-path MODEL_PATH] [--data-path DATA_PATH] [--output-dir OUTPUT_DIR]

Example:
    python evaluate.py --model-path models_store/v1/fraud_v1.pkl
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

# Add the project root to system path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.model import FraudDetectionModel
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger('evaluate')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate fraud detection model')
    parser.add_argument('--model-path', type=str,
                        default=str(project_root / 'models_store' / 'v1' / 'fraud_v1.pkl'),
                        help='Path to trained model')
    parser.add_argument('--data-path', type=str,
                        default=str(project_root / 'data' / 'processed' / 'transactions_processed.csv'),
                        help='Path to processed data CSV file')
    parser.add_argument('--output-dir', type=str,
                        default=str(project_root / 'models_store' / 'v1' / 'evaluation'),
                        help='Directory to save evaluation results')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for test set')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--threshold-search', action='store_true',
                        help='Whether to search for optimal threshold')
    
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


def preprocess_for_evaluation(df, target_column='Class'):
    """
    Prepare data for model evaluation.
    
    Args:
        df: DataFrame with transaction data
        target_column: Name of the target column
        
    Returns:
        Features DataFrame and target Series
    """
    logger.info("Preprocessing data for evaluation")
    
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


def get_test_data(X, y, test_size=0.2, random_state=42):
    """
    Get test data using consistent splitting.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion to use for test set
        random_state: Random state for reproducibility
        
    Returns:
        X_test, y_test
    """
    from sklearn.model_selection import train_test_split
    
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    # Split to get test data with same random state as used in training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_test, y_test


def find_optimal_threshold(model, X, y):
    """
    Find the optimal classification threshold.
    
    Args:
        model: Trained FraudDetectionModel instance
        X: Features
        y: Target
        
    Returns:
        Dictionary with optimal thresholds for different metrics
    """
    logger.info("Finding optimal classification thresholds")
    
    # Get predicted probabilities
    y_probs = model.predict(X)
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y, y_probs)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold_f1 = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else pr_thresholds[-1]
    
    # Find threshold that gives balanced precision and recall
    balanced_idx = np.argmin(np.abs(precision - recall))
    balanced_threshold = pr_thresholds[balanced_idx] if balanced_idx < len(pr_thresholds) else pr_thresholds[-1]
    
    # Find threshold for high precision (>0.9) with best recall
    high_precision_idx = np.where(precision > 0.9)[0]
    if len(high_precision_idx) > 0:
        high_precision_recall = recall[high_precision_idx]
        best_high_precision_idx = high_precision_idx[np.argmax(high_precision_recall)]
        high_precision_threshold = pr_thresholds[best_high_precision_idx] if best_high_precision_idx < len(pr_thresholds) else pr_thresholds[-1]
    else:
        high_precision_threshold = 0.95  # Default if no threshold gives >0.9 precision
    
    # Find threshold for high recall (>0.9) with best precision
    high_recall_idx = np.where(recall > 0.9)[0]
    if len(high_recall_idx) > 0:
        high_recall_precision = precision[high_recall_idx]
        best_high_recall_idx = high_recall_idx[np.argmax(high_recall_precision)]
        high_recall_threshold = pr_thresholds[best_high_recall_idx] if best_high_recall_idx < len(pr_thresholds) else pr_thresholds[-1]
    else:
        high_recall_threshold = 0.05  # Default if no threshold gives >0.9 recall
    
    optimal_thresholds = {
        'max_f1': float(optimal_threshold_f1),
        'balanced': float(balanced_threshold),
        'high_precision': float(high_precision_threshold),
        'high_recall': float(high_recall_threshold),
        'default': 0.5
    }
    
    logger.info(f"Optimal thresholds: {optimal_thresholds}")
    
    return optimal_thresholds


def evaluate_model(model, X, y, threshold=0.5):
    """
    Evaluate model performance.
    
    Args:
        model: Trained FraudDetectionModel instance
        X: Features
        y: Target
        threshold: Classification threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating model with threshold={threshold}")
    
    # Get predictions
    y_probs = model.predict(X)
    y_pred = (y_probs >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_probs),
        'pr_auc': average_precision_score(y, y_probs),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'threshold': threshold
    }
    
    # Log key metrics
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, figsize=(10, 8), output_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        figsize: Figure size as (width, height)
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        cbar=False,
        square=True
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['Legitimate (0)', 'Fraud (1)'])
    plt.yticks([0.5, 1.5], ['Legitimate (0)', 'Fraud (1)'], rotation=0)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {output_path}")
    
    return plt.gcf()


def plot_threshold_impact(model, X, y, figsize=(15, 10), output_path=None):
    """
    Plot impact of different thresholds on precision, recall, and F1.
    
    Args:
        model: Trained FraudDetectionModel instance
        X: Features
        y: Target
        figsize: Figure size as (width, height)
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Get predicted probabilities
    y_probs = model.predict(X)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, y_probs)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    # Calculate false positive rate for each threshold
    fpr_list = []
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_list.append(fpr)
    
    # Plot impact of threshold
    plt.figure(figsize=figsize)
    
    plt.subplot(2, 1, 1)
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Impact of Threshold on Precision, Recall, and F1')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(thresholds, fpr_list, label='False Positive Rate')
    plt.xlabel('Threshold')
    plt.ylabel('False Positive Rate')
    plt.title('Impact of Threshold on False Positive Rate')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Threshold impact plot saved to {output_path}")
    
    return plt.gcf()


def evaluate_model_at_different_thresholds(model, X, y, thresholds_dict):
    """
    Evaluate model at different thresholds.
    
    Args:
        model: Trained FraudDetectionModel instance
        X: Features
        y: Target
        thresholds_dict: Dictionary with threshold names and values
        
    Returns:
        Dictionary with evaluation metrics for each threshold
    """
    results = {}
    
    for threshold_name, threshold_value in thresholds_dict.items():
        logger.info(f"Evaluating with {threshold_name} threshold ({threshold_value:.4f})")
        metrics = evaluate_model(model, X, y, threshold=threshold_value)
        results[threshold_name] = metrics
    
    return results


def generate_class_probability_distribution(model, X, y, figsize=(12, 6), output_path=None):
    """
    Generate probability distribution plot for each class.
    
    Args:
        model: Trained FraudDetectionModel instance
        X: Features
        y: Target
        figsize: Figure size as (width, height)
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Get predicted probabilities
    y_probs = model.predict(X)
    
    # Create DataFrame with probabilities and true labels
    df = pd.DataFrame({
        'Probability': y_probs,
        'True_Class': y
    })
    
    plt.figure(figsize=figsize)
    
    # Plot probability distribution for each class
    sns.histplot(
        data=df, x='Probability', hue='True_Class', 
        element='step', stat='probability',
        common_norm=False, bins=50,
        palette=['#2ecc71', '#e74c3c']
    )
    
    plt.title('Probability Distribution by Class')
    plt.xlabel('Predicted Probability of Fraud')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.legend(['Legitimate (0)', 'Fraud (1)'])
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Probability distribution plot saved to {output_path}")
    
    return plt.gcf()


def plot_precision_recall_curve(y_true, y_probs, figsize=(10, 8), output_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        figsize: Figure size as (width, height)
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {output_path}")
    
    return plt.gcf()


def plot_roc_curve(y_true, y_probs, figsize=(10, 8), output_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        figsize: Figure size as (width, height)
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"ROC curve saved to {output_path}")
    
    return plt.gcf()


def save_evaluation_results(results, output_path):
    """
    Save evaluation results to JSON.
    
    Args:
        results: Dictionary with evaluation results
        output_path: Path to save the JSON file
    """
    # Convert any numpy values to Python native types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_json = convert_numpy_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    logger.info("Starting model evaluation process")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trained model
    model_path = Path(args.model_path)
    try:
        logger.info(f"Loading model from {model_path}")
        model = FraudDetectionModel.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load and preprocess data
    data_path = Path(args.data_path)
    df = load_data(data_path)
    X, y = preprocess_for_evaluation(df)
    
    # Get test data using same split as training
    X_test, y_test = get_test_data(X, y, test_size=args.test_size, random_state=args.random_state)
    
    # Perform threshold search if requested
    if args.threshold_search:
        optimal_thresholds = find_optimal_threshold(model, X_test, y_test)
        evaluation_thresholds = optimal_thresholds
    else:
        evaluation_thresholds = {'default': args.threshold}
    
    # Evaluate model at different thresholds
    results = evaluate_model_at_different_thresholds(model, X_test, y_test, evaluation_thresholds)
    
    # Save evaluation results
    save_evaluation_results(results, output_dir / 'evaluation_metrics.json')
    
    # Get probabilities for plots
    y_probs = model.predict(X_test)
    
    # Generate plots
    try:
        # Confusion matrix (using default threshold)
        y_pred = (y_probs >= args.threshold).astype(int)
        plot_confusion_matrix(
            y_test, y_pred, 
            output_path=output_dir / 'confusion_matrix.png'
        )
        
        # Threshold impact
        plot_threshold_impact(
            model, X_test, y_test, 
            output_path=output_dir / 'threshold_impact.png'
        )
        
        # Class probability distribution
        generate_class_probability_distribution(
            model, X_test, y_test, 
            output_path=output_dir / 'probability_distribution.png'
        )
        
        # Precision-Recall curve
        plot_precision_recall_curve(
            y_test, y_probs, 
            output_path=output_dir / 'precision_recall_curve.png'
        )
        
        # ROC curve
        plot_roc_curve(
            y_test, y_probs, 
            output_path=output_dir / 'roc_curve.png'
        )
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
    
    # Close all plot windows
    plt.close('all')
    
    logger.info(f"Model evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
