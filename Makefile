# =============================================================================
# Cudara — Developer Makefile
# =============================================================================
.PHONY: help install install-dev lint format typecheck test test-cov \
        run serve clean docker-build docker-run docker-stop \
        build publish pre-commit

SHELL     := /bin/bash
PYTHON    := python3
UV        := uv
APP       := cudara
PORT      := 8000
HOST      := 0.0.0.0
IMAGE     := cudara
TAG       := latest

# -----------------------------------------------------------------------------
# Help (default target)
# -----------------------------------------------------------------------------
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
install: ## Install production dependencies
	$(UV) sync --no-dev --extra vlm

install-dev: ## Install all dependencies (dev + vlm)
	$(UV) sync --dev --extra vlm

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------
lint: ## Run linter (ruff check)
	$(UV) run ruff check src/ tests/

format: ## Auto-format code (ruff)
	$(UV) run ruff format src/ tests/
	$(UV) run ruff check --fix src/ tests/

format-check: ## Check formatting without changing files
	$(UV) run ruff format --check src/ tests/

typecheck: ## Run type checker (mypy)
	$(UV) run mypy src/cudara/ --ignore-missing-imports

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
test: ## Run all tests
	$(UV) run pytest

test-v: ## Run tests with verbose output
	$(UV) run pytest -v --tb=long

test-cov: ## Run tests with coverage report
	$(UV) run pytest --cov=src/cudara --cov-report=term-missing --cov-report=html

test-unit: ## Run only unit tests
	$(UV) run pytest -m unit

test-fast: ## Run tests excluding slow markers
	$(UV) run pytest -m "not slow"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
serve: ## Start the dev server (with reload)
	$(UV) run cudara serve --host $(HOST) --port $(PORT) --reload

run: ## Start the production server
	$(UV) run cudara serve --host $(HOST) --port $(PORT)

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker-build: ## Build Docker image
	docker build -t $(IMAGE):$(TAG) .

docker-run: ## Run Docker container (requires NVIDIA runtime)
	docker run --rm -it \
		--gpus all \
		-p $(PORT):8000 \
		-v $$(pwd)/models:/app/models \
		-v $$(pwd)/registry.json:/app/registry.json \
		$(IMAGE):$(TAG)

docker-stop: ## Stop all running Cudara containers
	docker ps -q --filter ancestor=$(IMAGE):$(TAG) | xargs -r docker stop

docker-shell: ## Open a shell in the container
	docker run --rm -it --gpus all $(IMAGE):$(TAG) /bin/bash

docker-logs: ## Tail logs from running container
	docker logs -f $$(docker ps -q --filter ancestor=$(IMAGE):$(TAG) | head -1)

# -----------------------------------------------------------------------------
# Build & Publish
# -----------------------------------------------------------------------------
build: clean-dist ## Build wheel and sdist
	$(UV) build

publish-check: build ## Validate built package
	$(UV) tool run twine check dist/*

# -----------------------------------------------------------------------------
# Pre-commit (run all checks)
# -----------------------------------------------------------------------------
pre-commit: lint format-check typecheck test ## Run all CI checks locally

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean: ## Remove all build/cache artifacts
	rm -rf dist/ build/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-dist: ## Remove only dist artifacts
	rm -rf dist/ build/

clean-models: ## Remove downloaded models (DESTRUCTIVE)
	@echo "\033[31mThis will delete all downloaded models. Press Ctrl+C to cancel.\033[0m"
	@sleep 3
	rm -rf models/*
	echo "{}" > registry.json