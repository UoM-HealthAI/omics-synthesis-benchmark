#!/bin/bash
# Fix PyTorch for A100 GPU compatibility

echo "Fixing PyTorch for A100 GPU compatibility..."

# Activate conda environment
source activate activa

# Uninstall current PyTorch
pip uninstall torch -y

# Install A100-compatible PyTorch
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

echo "PyTorch updated for A100 compatibility!"

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"