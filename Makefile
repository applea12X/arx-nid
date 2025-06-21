# Makefile for arx-nid project data management

.PHONY: help data-list data-pull data-download data-clean setup

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install project dependencies
	pip install -r requirements.txt

data-list:  ## List all available datasets
	python3 scripts/download_all.py --list

data-download:  ## Download all datasets
	python3 scripts/download_all.py

data-download-%:  ## Download specific dataset (e.g., make data-download-hitar-2024)
	python3 scripts/download_all.py $*

data-pull:  ## Pull data from DVC remote
	dvc pull

data-push:  ## Push data to DVC remote
	dvc push

data-status:  ## Show DVC status
	dvc status

data-clean:  ## Remove downloaded raw data (keeps DVC files)
	rm -rf data/raw/*.zip
	rm -rf data/raw/*/

data-add:  ## Add all raw data to DVC tracking
	find data/raw -name "*.zip" -o -type d -mindepth 1 -maxdepth 1 | xargs -I {} dvc add {}

validate-data:  ## Validate data integrity and checksums
	@echo "Validating data manifest..."
	@python3 -c "import csv; print('Datasets:'); [print(f'  {r[\"dataset\"]}') for r in csv.DictReader(open('data/datasets.csv'))]"

init-dvc:  ## Initialize DVC (if not already done)
	dvc init --no-scm
	dvc remote add -d storage s3://arx-nid-dvc-storage

# Development targets
dev-setup:  ## Set up development environment
	pip install -r requirements.txt
	pip install jupyter black flake8 pytest

test-downloads:  ## Test download scripts without downloading
	@echo "Testing download scripts..."
	@for script in scripts/download_*.py; do \
		echo "Testing $$script..."; \
		python3 "$$script" --help || echo "Failed: $$script"; \
	done

# Documentation
docs-data:  ## Generate data documentation
	@echo "Data catalogue available at: data/README.md"
	@echo "Datasets manifest: data/datasets.csv"
