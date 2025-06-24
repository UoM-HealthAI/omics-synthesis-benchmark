#!/bin/bash

# ACTIVA Environment Setup Script
# This script only sets up the ACTIVA conda environment based on requirements.txt

set -e

ENV_NAME="activa"
PYTHON_VERSION="3.8"

setup_environment() {
    echo "=================================================="
    echo "Setting up ACTIVA Environment"
    echo "=================================================="
    
    echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
    
    # Create conda environment with Python 3.8
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    
    echo "Installing core dependencies from requirements.txt..."
    
    # Install dependencies from requirements.txt
    conda run -n $ENV_NAME pip install tqdm==4.47.0
    conda run -n $ENV_NAME pip install pandas==1.2.0
    conda run -n $ENV_NAME pip install numpy==1.18.5
    conda run -n $ENV_NAME pip install scanpy==1.7.0
    conda run -n $ENV_NAME pip install tensorboardX==2.1
    
    # Install PyTorch for A100 GPU compatibility
    echo "Installing PyTorch 1.9.1 for A100 GPU..."
    conda run -n $ENV_NAME pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    
    # Install ACTINN from GitHub
    echo "Installing ACTINN from GitHub repository..."
    conda run -n $ENV_NAME pip install git+http://github.com/SindiLab/ACTINN-PyTorch.git#egg=ACTINN
    
    # Install additional dependencies that might be needed
    echo "Installing additional scientific computing dependencies..."
    conda run -n $ENV_NAME pip install scipy scikit-learn matplotlib seaborn
    conda run -n $ENV_NAME pip install h5py tables anndata
    
    echo ""
    echo "✅ Environment setup completed with ACTIVA dependencies!"
    echo ""
}

echo "🚀 Starting ACTIVA Environment Setup"
setup_environment
echo "🎉 ACTIVA environment setup completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "conda activate $ENV_NAME"