#* Variables
SHELL := /bin/bash
PYTHON := .venv/bin/python

.PHONY: help
help:
	@echo "Commands:"
	@echo "uv-download     : downloads and installs the uv package manager"
	@echo "uv-activate	   : activates the uv python environment"
	@echo "install         : installs required dependencies"
	@echo "install-dev     : installs the dev dependencies for the project"
	@echo "check-style     : run checks on all files without fixing them."
	@echo "fix-style       : run checks on files and potentially modifies them."
	@echo "test            : run all tests."
	@echo "clean           : cleans all unecessary files."

#* UV
uv-download:
	@echo "Downloading uv package manager..."
	@if [[ $OS == "Windows_NT" ]]; then \
		irm https://astral.sh/uv/install.ps1 | iex; \
	else \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv venv


.PHONY: uv-activate
uv-activate:
	@if [[ "$(OS)" == "Windows_NT" ]]; then \
		./uv/Scripts/activate.ps1 \
	else \
		source .venv/bin/activate; \
	fi

#* Installation

.PHONY: install
install:
	uv sync

.PHONY: install-dev
install-dev:
	uv sync --all-extras && pre-commit install && pre-commit autoupdate

#* Linting
.PHONY: check-style
check-style:
	$(PYTHON) -m ruff format --check --config pyproject.toml ./
	$(PYTHON) -m ruff check --no-fix --config pyproject.toml ./

.PHONY: fix-style
fix-style:
	$(PYTHON) -m ruff format --config pyproject.toml ./
	$(PYTHON) -m ruff check --config pyproject.toml ./

#* Linting
.PHONY: test
test:
	$(PYTHON) -m pytest -c pyproject.toml -vv

#* Remove
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: clean
clean: pycache-remove build-remove
