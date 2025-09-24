# Real-Time Fraud Detection System

A comprehensive fraud detection system that processes transactions in real-time, using machine learning to identify potentially fraudulent activities.

## Implementation Summary

This implementation provides a robust fraud detection pipeline with the following key components:

- **LightGBM Model**: Gradient boosting framework optimized for fraud detection with imbalanced data
- **Comprehensive Evaluation**: Precision-recall curves, ROC curves, confusion matrices, and threshold optimization
- **Model Registry**: Version control and metadata tracking for model deployments
- **Automated Retraining**: Scheduled model retraining with data processing pipeline
- **Cross-platform Support**: Both Bash and PowerShell scripts for Unix/Linux/macOS and Windows environments

## Project Overview

This project implements an end-to-end fraud detection system with the following components:

- **Data Processing Pipeline**: Handles raw transaction data and preprocesses it for model training
- **Machine Learning Model**: LightGBM-based fraud detection with comprehensive evaluation metrics
- **Real-Time Processing**: Stream processing for real-time fraud detection
- **Model Monitoring**: Detects data drift and model performance degradation
- **API Interface**: REST API for transaction validation and monitoring
- **Retraining Pipeline**: Automated model retraining and evaluation

## Project Structure

```
├── Makefile              # Build automation
├── README.md             # Project documentation
├── data/                 # Data directory
│   ├── generate_synthetic.py  # Script to generate synthetic transaction data
│   ├── processed/        # Processed data
│   ├── raw/              # Raw data
│   └── sample_payloads/  # Sample API payloads
├── models_store/         # Model storage
│   ├── registry.json     # Model registry
│   └── v1/               # Model versions
├── notebooks/            # Jupyter notebooks
│   └── exploratory.ipynb # Data exploration notebook
├── scripts/              # Utility scripts
│   ├── deploy.sh         # Deployment script
│   ├── run_locally.sh    # Local execution script (Unix)
│   ├── run_locally.ps1   # Local execution script (Windows)
│   ├── schedule_retrain.sh # Retraining scheduler (Unix)
│   └── schedule_retrain.ps1 # Retraining scheduler (Windows)
├── src/                  # Source code
│   ├── api/              # API server
│   ├── features/         # Feature processing
│   ├── infra/            # Infrastructure configuration
│   ├── models/           # Model code
│   │   ├── evaluate.py   # Model evaluation
│   │   ├── hyperparams.yml # Hyperparameters configuration
│   │   ├── model.py      # Core model class
│   │   ├── register_model.py # Model registration
│   │   ├── retrain.py    # Model retraining
│   │   └── train.py      # Model training
│   ├── monitoring/       # Monitoring and alerting
│   ├── streaming/        # Stream processing
│   └── utils/            # Utility functions
└── tests/                # Tests
    ├── test_api.py       # API tests
    ├── test_model.py     # Model tests
    └── test_preprocess.py # Preprocessing tests
```

## Model Architecture

The fraud detection model uses LightGBM, a gradient boosting framework that is designed for high performance and large-scale data. The key components of the model are:

- **Feature Engineering**: Transaction amount, time-based features, and behavioral patterns
- **Model Training**: Gradient boosted trees with hyperparameter optimization
- **Evaluation**: Precision-recall focused metrics suitable for imbalanced datasets
- **Versioning**: Model registry with metadata and performance tracking

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (see requirements below)
- For Unix/Linux/macOS: Bash shell
- For Windows: PowerShell

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/real-time-fraud-detection.git
   cd real-time-fraud-detection
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   ```

   On Unix/Linux/macOS:
   ```
   source venv/bin/activate
   ```

   On Windows:
   ```
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the System Locally

#### On Unix/Linux/macOS:

```
bash scripts/run_locally.sh
```

#### On Windows:

```
powershell -ExecutionPolicy Bypass -File scripts\run_locally.ps1
```

This script will:
1. Generate synthetic data if needed
2. Preprocess the data
3. Train and evaluate the model if not already present
4. Start all system components (API, streaming, monitoring)

### Training a Model

To train a model with custom parameters:

```
python src/models/train.py --data-path data/processed/transactions_processed.csv --model-path models_store/v1/fraud_v1.pkl
```

### Evaluating a Model

To evaluate an existing model:

```
python src/models/evaluate.py --model-path models_store/v1/fraud_v1.pkl --data-path data/processed/transactions_processed.csv
```

### Scheduled Retraining

To set up scheduled retraining:

#### On Unix/Linux/macOS:

```
# Add to crontab (runs daily at 2 AM)
0 2 * * * /path/to/project/scripts/schedule_retrain.sh
```

#### On Windows:

Create a scheduled task that runs:
```
powershell -ExecutionPolicy Bypass -File C:\path\to\project\scripts\schedule_retrain.ps1
```

## API Usage

The API server provides endpoints for transaction validation:

```
POST /api/v1/predict

{
  "transaction_id": "TX123456",
  "amount": 250.0,
  "timestamp": "2023-05-15T14:23:10",
  "account_id": "ACC987654",
  "merchant_category": 5411
}
```

## Monitoring

The system includes:

- **Drift Detection**: Monitors for changes in transaction patterns
- **Performance Metrics**: Tracks model accuracy, precision, recall in production
- **Logging**: Comprehensive logging for troubleshooting

## Required Python Packages

Main dependencies:
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn
- fastapi (for API)
- pydantic (for data validation)
- uvicorn (for API server)
- prometheus-client (for metrics)
- pyyaml (for configuration)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Credit card fraud detection dataset from Kaggle
- LightGBM team for the gradient boosting framework
