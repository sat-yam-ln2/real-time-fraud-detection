# Real-Time Fraud Detection System Makefile

.PHONY: setup data train evaluate run_local test clean

# Default target
all: setup data train evaluate

# Setup environment
setup:
	@echo "Setting up environment..."
	python -m venv venv || echo "Virtual environment already exists"
	. venv/bin/activate && pip install -r requirements.txt

# Generate and process data
data:
	@echo "Generating and processing data..."
	python data/generate_synthetic.py
	python src/features/preprocess.py \
		--input data/raw/transactions_raw.csv \
		--output data/processed/transactions_processed.csv

# Train model
train:
	@echo "Training fraud detection model..."
	python src/models/train.py \
		--data-path data/processed/transactions_processed.csv \
		--model-path models_store/v1/fraud_v1.pkl

# Evaluate model
evaluate:
	@echo "Evaluating fraud detection model..."
	python src/models/evaluate.py \
		--model-path models_store/v1/fraud_v1.pkl \
		--data-path data/processed/transactions_processed.csv \
		--output-dir models_store/v1/evaluation \
		--threshold-search

# Register model
register:
	@echo "Registering model in registry..."
	python src/models/register_model.py \
		--model-path models_store/v1/fraud_v1.pkl \
		--model-name fraud_model \
		--model-version v1 \
		--evaluation-path models_store/v1/evaluation/evaluation_metrics.json \
		--description "Fraud detection model" \
		--promote-to-production

# Run system locally
run_local:
	@echo "Running system locally..."
	bash scripts/run_locally.sh

# Run tests
test:
	@echo "Running tests..."
	pytest tests/

# Clean generated artifacts
clean:
	@echo "Cleaning generated artifacts..."
	rm -rf models_store/v1/fraud_v1.pkl
	rm -rf models_store/v1/evaluation/*
	rm -rf data/processed/*

# Windows compatibility targets
setup-win:
	@echo "Setting up environment on Windows..."
	python -m venv venv
	.\venv\Scripts\activate && pip install -r requirements.txt

run_local-win:
	@echo "Running system locally on Windows..."
	powershell -ExecutionPolicy Bypass -File scripts\run_locally.ps1
