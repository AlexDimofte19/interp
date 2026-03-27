#!/usr/bin/env bash

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Claude Code
#curl -fsSL https://claude.ai/install.sh | sh

# Install Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Use the shared HF cache
export HF_HOME="/workspace/shared/hf_cache"
if ! grep -q 'export HF_HOME="/workspace/shared/hf_cache"' ~/.bashrc 2>/dev/null; then
    echo 'export HF_HOME="/workspace/shared/hf_cache"' >> ~/.bashrc
    echo "Added HF_HOME to ~/.bashrc"
fi

# Log in to Hugging Face
hf auth login

echo "Setup complete."
