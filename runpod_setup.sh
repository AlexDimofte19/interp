#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for a fresh RunPod instance.
# Everything below is idempotent -- safe to re-run on an existing pod.

REPO_DIR="${REPO_DIR:-/workspace/repo/interp}"
HF_CACHE="${HF_CACHE:-/workspace/shared/hf_cache}"

# Install uv (Python package manager)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Install Claude Code
#curl -fsSL https://claude.ai/install.sh | sh

# Install the Hugging Face CLI system-wide, so plain `hf ...` works in any
# shell. (The project env also gets `hf` transitively via huggingface_hub;
# this is for interactive use outside `uv run`.)
if ! command -v hf >/dev/null 2>&1; then
    curl -LsSf https://hf.co/cli/install.sh | bash
fi

# Use the shared HF cache
export HF_HOME="$HF_CACHE"
if ! grep -q "export HF_HOME=\"$HF_CACHE\"" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=\"$HF_CACHE\"" >> ~/.bashrc
    echo "Added HF_HOME to ~/.bashrc"
fi

# Build the environment from uv.lock. This installs the project itself in
# editable mode, so no separate `pip install -e .` is needed.
cd "$REPO_DIR"
uv sync --extra gpu --extra notebook

# Log in to Hugging Face
hf auth login

echo "Setup complete. Run commands with 'uv run ...' from $REPO_DIR."
