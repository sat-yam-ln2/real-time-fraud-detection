#!/usr/bin/env python
"""
Fraud Detection Model Definition

This module defines the fraud detection model using LightGBM and provides
functions for model initialization, training, prediction, and evaluation.
"""

import os
import time
import logging
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Tuple, List, Any, Optional, Union
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Get logger
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """Fraud Detection model based on LightGBM."""

    def __init__(self, config: Dict = None):
        """
        Initialize the fraud detection model.
        
        Args:
            config: Dictionary containing model configuration parameters.
                   If None, default parameters will be used.
        """
        self.model = None
        self.feature_importance = None
        self.training_history = None
        
        # Default model parameters
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'verbose': -1,
            'class_weight': 'balanced',
            'random_state': 42,
        }
        
        # Update with provided config if available
        if config:
            self.params.update(config)

        # Store hyperparameters used
        self.hyperparameters = self.params.copy()
        
        # Keep track of feature names
        self.feature_names = None
        
        # Record model version
        self.model_version = "v1"
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              categorical_features: List[str] = None) -> Dict:
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            categorical_features: List of categorical feature names
            
        Returns:
            Dictionary with training history
        """
        start_time = time.time()
        logger.info("Starting model training")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Prepare validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Prepare categorical features if provided
        cat_features = None
        if categorical_features:
            cat_features = [X_train.columns.get_loc(col) for col in categorical_features if col in X_train.columns]
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            categorical_feature=cat_features,
            free_raw_data=False
        )
        
        # Create validation dataset if provided
        valid_data = None
        if eval_set:
            valid_data = lgb.Dataset(
                X_val,
                label=y_val,
                categorical_feature=cat_features,
                reference=train_data,
                free_raw_data=False
            )
        
        # Train model
        evals_result = {}
        
        # Extract relevant parameters
        num_boost_round = self.params.get('n_estimators', 1000)
        early_stopping_rounds = self.params.get('early_stopping_rounds', 50)
        
        # Set up callbacks
        callbacks = [lgb.record_evaluation(evals_result)]
        
        # Add early stopping callback
        if early_stopping_rounds > 0 and valid_data:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=True))
            
        # Add log evaluation callback for verbose output
        callbacks.append(lgb.log_evaluation(period=100))
        
        # Set up parameters that shouldn't be passed directly to train
        params_for_train = {k: v for k, v in self.params.items() if k not in [
            'n_estimators', 'early_stopping_rounds', 'verbose', 'verbose_eval',
            'class_weight', 'scale_pos_weight', 'n_jobs'
        ]}
        
        # Handle class weights and other special parameters
        if 'class_weight' in self.params and self.params['class_weight'] == 'balanced':
            # Use scale_pos_weight instead for LightGBM
            if 'scale_pos_weight' not in params_for_train:
                pos_weight = len(y_train) / (2 * np.sum(y_train))
                params_for_train['scale_pos_weight'] = pos_weight
        
        # Handle n_jobs parameter
        if 'n_jobs' in self.params:
            params_for_train['num_threads'] = self.params['n_jobs']
        
        self.model = lgb.train(
            params=params_for_train,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data] if valid_data else [train_data],
            valid_names=['train', 'valid'] if valid_data else ['train'],
            callbacks=callbacks
        )
        
        # Save feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importance(importance_type='gain')
        }).sort_values(by='Importance', ascending=False)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Save training history
        self.training_history = {
            'training_time': training_time,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None,
            'evals_result': evals_result,
            'params': self.params,
        }
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions with the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of fraud probability predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        return self.model.predict(X)
    
    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions with the trained model.
        
        Args:
            X: Features to predict on
            threshold: Classification threshold
            
        Returns:
            Array of binary predictions (0 = legitimate, 1 = fraud)
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Make predictions
        y_prob = self.predict(X)
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate evaluation metrics
        metrics = {
            'roc_auc': roc_auc_score(y, y_prob),
            'pr_auc': average_precision_score(y, y_prob),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'threshold': threshold
        }
        
        return metrics
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train the model first.")
        
        plt.figure(figsize=figsize)
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(
            x='Importance',
            y='Feature',
            data=top_features,
            palette='viridis'
        )
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curve(self, X: pd.DataFrame, y: pd.Series, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            X: Test features
            y: Test labels
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        y_prob = self.predict(X)
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = roc_auc_score(y, y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, X: pd.DataFrame, y: pd.Series, 
                                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            X: Test features
            y: Test labels
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        y_prob = self.predict(X)
        precision, recall, _ = precision_recall_curve(y, y_prob)
        pr_auc = average_precision_score(y, y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        
        return plt.gcf()
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save model with metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'model_version': self.model_version,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FraudDetectionModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded FraudDetectionModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        # Create a new instance
        instance = cls()
        
        # Restore model attributes
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.hyperparameters = model_data['hyperparameters']
        instance.feature_importance = model_data['feature_importance']
        instance.training_history = model_data['training_history']
        instance.model_version = model_data.get('model_version', 'v1')
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance
    
    def get_model_card(self, metrics: Dict = None) -> Dict:
        """
        Generate a model card with metadata and metrics.
        
        Args:
            metrics: Optional evaluation metrics to include
            
        Returns:
            Dictionary containing model card information
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        model_card = {
            'model_name': 'Fraud Detection LightGBM',
            'model_version': self.model_version,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': self.hyperparameters,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'top_features': self.feature_importance.head(10).to_dict('records') if self.feature_importance is not None else None,
            'training_info': self.training_history,
        }
        
        if metrics:
            model_card['metrics'] = metrics
            
        return model_card
        

def load_hyperparameters(config_path: str = None) -> Dict:
    """
    Load hyperparameters from configuration file.
    
    Args:
        config_path: Path to hyperparameter YAML file
        
    Returns:
        Dictionary with hyperparameters
    """
    # Default hyperparameters
    default_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'class_weight': 'balanced',
        'random_state': 42,
    }
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as file:
                file_params = yaml.safe_load(file)
                if file_params and isinstance(file_params, dict):
                    # If there's a lightgbm section, use those parameters
                    if 'lightgbm' in file_params:
                        default_params.update(file_params['lightgbm'])
                    else:
                        default_params.update(file_params)
                    logger.info(f"Hyperparameters loaded from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load hyperparameters from {config_path}: {e}")
            logger.info("Using default hyperparameters")
    
    return default_params
