# Contributing to telos-interp

## Prerequisites

- Python 3.10 or higher (recommended: 3.12)
- [UV package manager](https://docs.astral.sh/uv/) (install via the official installer, e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh` and follow the prompts)

## Development Setup

1. **Install uv** following the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

2. **Clone the repository**:
   ```bash
   git clone https://github.com/SPAR-Telos/interp
   cd interp
   ```

3. **Create the virtual environment and install project dependencies**:
   ```bash
   uv sync
   ```

   This will:
   - Resolve dependencies declared in `pyproject.toml` / `uv.lock`
   - Create a `.venv` folder at the project root (managed by uv)

4. **Activate the environment (optional)**:
   ```bash
   source .venv/bin/activate
   ```

5. **Install pre-commit hooks**:
   ```bash
   uvx pre-commit install
   ```

`uvx` is shorthand for `uv tool run`; it installs and caches development tools without modifying the project's dependency lists.

## Editor Configuration

### VS Code Setup with Ruff

To configure VS Code to format your Python code on save using Ruff via `uvx ruff format`:

1. **Install Ruff and uvx**:
   ```bash
   uv tool install ruff@latest
   ```

2. **Install the Ruff VS Code Extension**:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Ruff" and install the extension by `charliermarsh`

3. **Configure VS Code Settings**:
   Open your VS Code settings (Ctrl+Shift+P → "Preferences: Open Settings (JSON)") and add:
   ```json
   {
     "[python]": {
       "editor.formatOnSave": true,
       "editor.defaultFormatter": "charliermarsh.ruff",
       "editor.codeActionsOnSave": {
         "source.fixAll.ruff": "always",
         "source.organizeImports.ruff": "always"
       }
     },
     "ruff.path": ["uvx", "ruff"]
   }
   ```

This configuration will automatically format your Python code and organize imports every time you save a file, using Ruff via `uvx ruff format`.

## Project Structure

```
├── telos_interp/          # Main package source code
│   └── commands/          # CLI commands
│       └── cli.py         # Main CLI application using Tyro
├── tests/                 # Test files
├── configs/               # Probe training configuration files
├── pyproject.toml         # Project configuration and dependencies
├── Makefile              # Development commands
└── .pre-commit-config.yaml # Pre-commit hooks configuration
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

1. **Modify `pyproject.toml`** with new dependencies or version constraints
2. **Refresh the lockfile**:
   ```bash
   uv lock --upgrade
   ```
   Use `uv lock --upgrade-package <name>` for targeted upgrades.
3. **Install the updated dependencies locally**:
   ```bash
   make install-dev
   ```
4. Commit both `pyproject.toml` and `uv.lock`.
