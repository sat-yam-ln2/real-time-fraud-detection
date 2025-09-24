#!/usr/bin/env python
"""
Logging Utility

This module provides consistent logging functionality across the fraud detection system.
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_level=logging.INFO, log_file=None):
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    # Get a reference to the root logger and remove all handlers
    # This prevents duplicate logging
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter with consistent timestamp format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_log_file_path(component_name):
    """
    Generate a log file path based on component name and timestamp.
    
    Args:
        component_name: Name of the component/module
        
    Returns:
        Path to log file
    """
    # Get project root directory
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Create logs directory if it doesn't exist
    logs_dir = project_root / 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{component_name}_{timestamp}.log"
    
    return str(logs_dir / filename)


class ModelLogger:
    """Class for logging model training and inference events."""
    
    def __init__(self, model_name, model_version=None, log_level=logging.INFO):
        """
        Initialize a logger for model operations.
        
        Args:
            model_name: Name of the model
            model_version: Optional version of the model
            log_level: Logging level
        """
        self.model_name = model_name
        self.model_version = model_version
        
        # Set up logger
        logger_name = f"model.{model_name}"
        if model_version:
            logger_name = f"{logger_name}.{model_version}"
            
        self.logger = setup_logger(
            logger_name,
            log_level=log_level,
            log_file=get_log_file_path(logger_name)
        )
    
    def log_training_start(self, config=None):
        """Log start of model training with configuration."""
        self.logger.info(f"Starting training for model {self.model_name}")
        if config:
            self.logger.info(f"Training configuration: {config}")
    
    def log_training_complete(self, metrics=None, training_time=None):
        """Log completion of model training with metrics."""
        msg = f"Training completed for model {self.model_name}"
        if training_time:
            msg += f" in {training_time:.2f} seconds"
        self.logger.info(msg)
        
        if metrics:
            self.logger.info(f"Training metrics: {metrics}")
    
    def log_inference_event(self, num_samples, prediction_time=None):
        """Log inference event."""
        msg = f"Inference completed for {num_samples} samples"
        if prediction_time:
            msg += f" in {prediction_time:.4f} seconds"
        self.logger.info(msg)
    
    def log_error(self, error_msg, exc_info=False):
        """Log error message."""
        self.logger.error(f"Error in model {self.model_name}: {error_msg}", exc_info=exc_info)


class APILogger:
    """Class for logging API events."""
    
    def __init__(self, api_name, log_level=logging.INFO):
        """
        Initialize a logger for API operations.
        
        Args:
            api_name: Name of the API component
            log_level: Logging level
        """
        self.api_name = api_name
        
        # Set up logger
        logger_name = f"api.{api_name}"
        self.logger = setup_logger(
            logger_name,
            log_level=log_level,
            log_file=get_log_file_path(logger_name)
        )
    
    def log_request(self, endpoint, method, status_code=None, duration=None):
        """Log API request."""
        msg = f"{method} request to {endpoint}"
        if status_code:
            msg += f" (status: {status_code})"
        if duration:
            msg += f" in {duration:.4f}s"
        self.logger.info(msg)
    
    def log_error(self, endpoint, method, error_msg, status_code=None, exc_info=False):
        """Log API error."""
        msg = f"Error in {method} request to {endpoint}"
        if status_code:
            msg += f" (status: {status_code})"
        msg += f": {error_msg}"
        self.logger.error(msg, exc_info=exc_info)
