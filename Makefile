# Makefile for DPA project
# Usage: make <target>

.PHONY: install test lint examples clean help

# Default target
help:
	@echo "DPA - Density Peak Advanced Clustering"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make install-dev - Install with dev dependencies"
	@echo "  make test        - Run all tests"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo "  make lint        - Run linter (flake8)"
	@echo "  make examples    - Run example scripts"
	@echo "  make notebooks   - Start Jupyter notebook server"
	@echo "  make clean       - Remove generated files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Install with dev dependencies
install-dev:
	pip install -e ".[dev,notebooks]"

# Run tests
test:
	python -m pytest tests/ -v

# Run tests with coverage
test-cov:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Run linter
lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503

# Run examples
examples:
	@echo "Running basic example..."
	python examples/example_basic.py
	@echo ""
	@echo "Running comparison example..."
	python examples/example_comparison.py

# Run basic example only (faster)
example-basic:
	python examples/example_basic.py

# Start Jupyter notebooks
notebooks:
	jupyter notebook notebooks/

# Clean generated files
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Verify installation
verify:
	python -c "from src import DPA, TwoNN, PAk, load_optdigits; print('All imports OK')"
	python -c "from src import DensityPeaks; print('DensityPeaks OK')"
	@echo "Installation verified successfully!"
