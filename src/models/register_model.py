#!/usr/bin/env python
"""
Fraud Detection Model Registration Script

This script handles the registration of trained models into the model registry
with proper versioning and metadata tracking.

Usage:
    python register_model.py [--model-path MODEL_PATH] [--model-name MODEL_NAME] [--model-version MODEL_VERSION]

Example:
    python register_model.py --model-path models_store/v1/fraud_v1.pkl --model-name fraud_model --model-version v2
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import pickle
import hashlib

# Add the project root to system path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger('register_model')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Register fraud detection model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model pickle file')
    parser.add_argument('--model-name', type=str, default='fraud_model',
                        help='Name to register the model under')
    parser.add_argument('--model-version', type=str,
                        help='Version for the model (e.g., v1, v2). If not provided, it will be auto-generated.')
    parser.add_argument('--registry-path', type=str, 
                        default=str(project_root / 'models_store' / 'registry.json'),
                        help='Path to the model registry JSON file')
    parser.add_argument('--evaluation-path', type=str,
                        help='Path to evaluation metrics JSON file to include in registration')
    parser.add_argument('--description', type=str, default='',
                        help='Description of the model')
    parser.add_argument('--promote-to-production', action='store_true',
                        help='Whether to mark this model as the production model')
    
    return parser.parse_args()


def read_registry(registry_path):
    """
    Read the model registry file.
    
    Args:
        registry_path: Path to registry JSON file
        
    Returns:
        Registry as a dictionary
    """
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            try:
                registry = json.load(f)
                logger.info(f"Loaded existing registry from {registry_path}")
                return registry
            except json.JSONDecodeError:
                logger.warning(f"Error reading registry at {registry_path}, creating new registry")
                return {
                    "models": {},
                    "last_updated": datetime.now().isoformat(),
                    "production_models": {}
                }
    else:
        logger.info(f"Registry not found at {registry_path}, creating new registry")
        return {
            "models": {},
            "last_updated": datetime.now().isoformat(),
            "production_models": {}
        }


def write_registry(registry, registry_path):
    """
    Write registry to file.
    
    Args:
        registry: Registry dictionary
        registry_path: Path to registry JSON file
    """
    registry['last_updated'] = datetime.now().isoformat()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Updated registry saved to {registry_path}")


def generate_model_version(model_name, registry):
    """
    Generate a new version for a model.
    
    Args:
        model_name: Name of the model
        registry: Registry dictionary
        
    Returns:
        New version string
    """
    if model_name not in registry['models']:
        return 'v1'
    
    versions = list(registry['models'][model_name].keys())
    if not versions:
        return 'v1'
    
    # Extract numeric part of versions and find the highest
    numeric_versions = []
    for version in versions:
        if version.startswith('v') and version[1:].isdigit():
            numeric_versions.append(int(version[1:]))
    
    if not numeric_versions:
        return 'v1'
    
    highest_version = max(numeric_versions)
    return f'v{highest_version + 1}'


def calculate_model_hash(model_path):
    """
    Calculate a hash of the model file for integrity verification.
    
    Args:
        model_path: Path to model file
        
    Returns:
        SHA-256 hash of the file
    """
    sha256_hash = hashlib.sha256()
    
    with open(model_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def get_model_metadata(model_path, evaluation_path=None):
    """
    Get metadata about the model.
    
    Args:
        model_path: Path to model file
        evaluation_path: Optional path to evaluation metrics
        
    Returns:
        Dictionary with model metadata
    """
    model_file = Path(model_path)
    
    # Load model to get its properties
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Extract model parameters if available
        model_params = getattr(model, 'params', {})
        if hasattr(model, 'model') and hasattr(model.model, 'params'):
            model_params = model.model.params
        
        # Get feature names if available
        feature_names = getattr(model, 'feature_names', [])
        if not feature_names and hasattr(model, 'model'):
            feature_names = getattr(model.model, 'feature_names', [])
    except Exception as e:
        logger.warning(f"Could not extract model parameters: {e}")
        model_params = {}
        feature_names = []
    
    # Get file stats
    file_stats = model_file.stat()
    
    # Calculate file hash
    file_hash = calculate_model_hash(model_path)
    
    # Get evaluation metrics if available
    evaluation_metrics = {}
    if evaluation_path and os.path.exists(evaluation_path):
        try:
            with open(evaluation_path, 'r') as f:
                evaluation_metrics = json.load(f)
            logger.info(f"Loaded evaluation metrics from {evaluation_path}")
        except Exception as e:
            logger.warning(f"Could not load evaluation metrics: {e}")
    
    metadata = {
        "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        "file_size_bytes": file_stats.st_size,
        "file_path": str(model_file.absolute()),
        "file_hash": file_hash,
        "model_parameters": model_params,
        "feature_names": feature_names[:20] if len(feature_names) > 20 else feature_names,  # Limit feature names list
        "feature_count": len(feature_names) if feature_names else None,
        "evaluation_metrics": evaluation_metrics
    }
    
    return metadata


def register_model(model_path, model_name, model_version, registry, registry_path, 
                   evaluation_path=None, description="", promote_to_production=False):
    """
    Register a model in the registry.
    
    Args:
        model_path: Path to model file
        model_name: Name to register the model under
        model_version: Version for the model
        registry: Registry dictionary
        registry_path: Path to registry JSON file
        evaluation_path: Optional path to evaluation metrics
        description: Optional description of the model
        promote_to_production: Whether to mark this model as the production model
        
    Returns:
        Updated registry dictionary
    """
    logger.info(f"Registering model '{model_name}' with version '{model_version}'")
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        sys.exit(1)
    
    # Get model metadata
    metadata = get_model_metadata(model_path, evaluation_path)
    
    # Add description and registration time
    metadata['description'] = description
    metadata['registered_at'] = datetime.now().isoformat()
    
    # Initialize model in registry if it doesn't exist
    if model_name not in registry['models']:
        registry['models'][model_name] = {}
    
    # Add model version to registry
    registry['models'][model_name][model_version] = metadata
    
    # Update production model if specified
    if promote_to_production:
        registry['production_models'][model_name] = model_version
        logger.info(f"Promoted '{model_name}:{model_version}' to production")
    
    # Write updated registry
    write_registry(registry, registry_path)
    
    return registry


def create_versioned_copy(model_path, model_name, model_version, models_store_path):
    """
    Create a versioned copy of the model in the models store.
    
    Args:
        model_path: Path to source model file
        model_name: Name of the model
        model_version: Version of the model
        models_store_path: Path to models store directory
        
    Returns:
        Path to the versioned copy
    """
    # Create versioned directory
    version_dir = os.path.join(models_store_path, model_version)
    os.makedirs(version_dir, exist_ok=True)
    
    # Define destination path
    dest_path = os.path.join(version_dir, f"{model_name}_{model_version}.pkl")
    
    # Copy model file
    shutil.copy2(model_path, dest_path)
    logger.info(f"Created versioned copy of model at {dest_path}")
    
    return dest_path


def main():
    """Main function to register a model."""
    args = parse_args()
    
    logger.info("Starting model registration process")
    
    # Read registry
    registry = read_registry(args.registry_path)
    
    # Generate model version if not provided
    model_version = args.model_version
    if not model_version:
        model_version = generate_model_version(args.model_name, registry)
        logger.info(f"Auto-generated model version: {model_version}")
    
    # Create versioned copy in models store
    models_store_path = os.path.join(project_root, 'models_store')
    versioned_model_path = create_versioned_copy(
        args.model_path, args.model_name, model_version, models_store_path
    )
    
    # Register model
    registry = register_model(
        model_path=versioned_model_path,
        model_name=args.model_name,
        model_version=model_version,
        registry=registry,
        registry_path=args.registry_path,
        evaluation_path=args.evaluation_path,
        description=args.description,
        promote_to_production=args.promote_to_production
    )
    
    logger.info(f"Model '{args.model_name}:{model_version}' registered successfully")


if __name__ == "__main__":
    main()
