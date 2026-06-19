#!/usr/bin/env bash

apt update

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


pip install -e .
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "kernels==0.12.0"
pip install seaborn



read -rp "Enter your email for the SSH key: " GIT_EMAIL
while [ -z "$GIT_EMAIL" ]; do
    read -rp "Enter your email for the SSH key: " GIT_EMAIL
done
ssh-keygen -t ed25519 -C "$GIT_EMAIL"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
echo ""
echo "go to https://github.com/settings/ssh/new and add the following key to your github keys list:"
cat ~/.ssh/id_ed25519.pub

echo "Setup complete."