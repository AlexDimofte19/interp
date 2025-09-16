# Contributing to telos-interp

## Prerequisites

- Python 3.10 or higher (recommended: 3.12)
- [UV package manager](https://docs.astral.sh/uv/) (install with `make uv-download`, activate environment with `source .venv/bin/activate`)

## Development Setup

### Option 1: Using UV (Recommended)

1. **Install UV package manager**:
   ```bash
   make uv-download
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/SPAR-Telos/interp
   cd interp
   ```

3. **Install development dependencies**:
   ```bash
   source .venv/bin/activate
   make install-dev
   ```

   This will:
   - Activate the UV virtual environment
   - Install all development dependencies
   - Set up pre-commit hooks
   - Install the package in editable mode

## Project Structure

```
├── telos_interp/          # Main package source code
│   └── commands/          # CLI commands
│       └── cli.py         # Main CLI application using Typer
├── tests/                 # Test files
├── pyproject.toml         # Project configuration and dependencies
├── Makefile              # Development commands
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── requirements.txt       # User dependencies
└── requirements-dev.txt   # Development dependencies
```

## Development Workflow

### Available Commands

The project uses a Makefile for common development tasks:

```bash
make help           # Show all available commands
make install        # Install production dependencies only
make install-dev    # Install development dependencies and setup
make test           # Run all tests with pytest
make check-style    # Check code style without fixing
make fix-style      # Fix code style issues automatically
make clean          # Clean up temporary files
make update-deps    # Update requirements files from pyproject.toml
```

### CLI Usage

After installation, the CLI is available as `interp-cli`:

```bash
interp-cli --help           # Show available CLI commands
interp-cli count-params openai-community/gpt2
```

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Ruff**: For linting and code formatting (configured in `pyproject.toml`)
- **Pre-commit hooks**: Automatically run checks before commits
- **Pytest**: For running tests with parallel execution support

### Testing

Run tests using:
```bash
make test
```

This runs pytest with:
- Parallel execution (`-n auto`)
- Verbose output (`-vv`)
- Configuration from `pyproject.toml`

Test markers available:
- `slow`: For time-intensive tests
- `require_cuda_gpu`: For tests requiring CUDA GPU

### Code Style

The project follows these style guidelines:
- Line length: 119 characters
- Python 3.10+ syntax
- Google-style docstrings
- Import sorting with isort

Before committing, always run:
```bash
make fix-style
```

### Pre-commit Hooks

Pre-commit hooks are automatically installed with `make install-dev`. They will:
- Check and fix trailing whitespace
- Validate TOML files
- Check for merge conflicts
- Run Ruff formatting and linting
- Run tests
- Clean temporary files

## Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality in the `tests/` directory

4. **Run the test suite**:
   ```bash
   make test
   ```

5. **Check code style**:
   ```bash
   make check-style
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   ```
   Pre-commit hooks will automatically run and may modify files.

7. **Push and create a pull request**

## Dependencies

### Core Dependencies
- `transformers>=4.42.0`: Hugging Face transformers library
- `nnsight>=0.5.3`: Neural network interpretability toolkit
- `torch>=2.0.1`: PyTorch deep learning framework
- `typer>=0.17.4`: CLI framework for building command-line interfaces
- `toml>=0.10.2`: TOML file parsing

### Optional Dependencies
- `data`: For data processing (`datasets`, `pandas`)
- `lint`: For development tools (`pytest`, `ruff`, `pre-commit`)
- `notebook`: For Jupyter notebook support (`ipykernel`, `ipywidgets`)

## Package Management

To update dependencies:

1. **Modify `pyproject.toml`** with new dependencies
2. **Update requirement files**:
   ```bash
   make update-deps
   ```
3. **Install updated dependencies**:
   ```bash
   make install-dev
   ```
