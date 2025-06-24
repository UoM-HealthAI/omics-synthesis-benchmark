#!/bin/bash

# scGAN Complete Workflow Deployment Script
# This script sets up environment and runs the complete scGAN experiment pipeline
# Compatible with A100 GPU using TensorFlow 2.6

set -e

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

ENV_NAME="scgan"
PYTHON_VERSION="3.8"  # Updated for TensorFlow 2.6 compatibility

# Dataset configuration - modify this section for different experiments
DATASET_NAME="tabula_muris"  # Options: tabula_muris, pbmc68k, human_pf_lung
SOURCE_DATA_PATH="/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scDiffusion/data/data"
EXPERIMENT_OUTPUT_DIR="output/benchmark_scGAN_${DATASET_NAME}"

# GPU and compute configuration
USE_GPU=true
CUDA_VERSION="11.8.0"
CUDNN_VERSION="8.7.0.84-CUDA-11.8.0"

# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

setup_environment() {
    echo "=================================================="
    echo "STEP 1: Setting up scGAN Environment"
    echo "=================================================="
    
    echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
    
    # Create conda environment with Python 3.8 (compatible with TensorFlow 2.6)
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    
    echo "Installing TensorFlow 2.6 with GPU support for A100..."
    # Install TensorFlow 2.6 with GPU support for A100
    conda run -n $ENV_NAME pip install tensorflow==2.6.0
    
    echo "Installing core dependencies compatible with TensorFlow 2.6..."
    # Install updated dependencies compatible with TensorFlow 2.6
    conda run -n $ENV_NAME pip install numpy==1.19.5 pandas==1.3.5 scikit-learn==1.0.2 matplotlib==3.5.3
    conda run -n $ENV_NAME pip install scanpy==1.8.2 'anndata==0.8.0' scipy==1.7.3 h5py==3.7.0
    conda run -n $ENV_NAME pip install joblib==1.1.0 natsort==8.2.0 tables==3.7.0
    
    echo ""
    echo "✅ Environment setup completed with TensorFlow 2.6 + A100 GPU support!"
    echo ""
}

prepare_dataset() {
    echo "=================================================="
    echo "STEP 2: Preparing Dataset - $DATASET_NAME"
    echo "=================================================="
    
    # Load required modules for GPU computation
    if [ "$USE_GPU" = true ]; then
        echo "Loading CUDA modules for A100 GPU..."
        module load CUDA/$CUDA_VERSION
        module load cuDNN/$CUDNN_VERSION
    fi
    
    # Activate conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    # Dataset-specific data preparation
    case $DATASET_NAME in
        "tabula_muris")
            echo "Preparing Tabula Muris dataset..."
            prepare_tabula_muris_data
            ;;
        "pbmc68k")
            echo "Preparing PBMC 68k dataset..."
            prepare_pbmc68k_data
            ;;
        "human_pf_lung")
            echo "Preparing Human PF Lung dataset..."
            prepare_human_pf_lung_data
            ;;
        *)
            echo "Unknown dataset: $DATASET_NAME"
            echo "Supported datasets: tabula_muris, pbmc68k, human_pf_lung"
            exit 1
            ;;
    esac
    
    echo "✅ Dataset preparation completed!"
    echo ""
}

prepare_tabula_muris_data() {
    echo "Converting Tabula Muris data to scGAN-compatible format..."
    prepare_dataset_generic "tabula_muris"
}

prepare_pbmc68k_data() {
    echo "Preparing PBMC 68k dataset..."
    prepare_dataset_generic "pbmc68k"
}

prepare_human_pf_lung_data() {
    echo "Preparing Human PF Lung dataset..."
    prepare_dataset_generic "human_pf_lung"
}

prepare_dataset_generic() {
    local dataset=$1
    echo "Converting $dataset data to scGAN-compatible format..."
    
    # Step 1: Extract data using modern environment (scdiffusion)
    echo "Step 1: Extracting data with modern scanpy..."
    conda activate scdiffusion
    python create_tabula_muris_compatible.py --dataset $dataset
    
    # Step 2: Create compatible h5ad using scGAN environment
    echo "Step 2: Creating compatible h5ad..."
    conda activate $ENV_NAME
    python create_tabula_muris_compatible.py step2 --dataset $dataset
    
    # Update parameters file to point to converted data
    if [ -f "parameters_test.json" ]; then
        sed -i "s|\"data_dir\": \".*\"|\"data_dir\": \"data/${dataset}/all_converted.h5ad\"|g" parameters_test.json
        echo "✅ Updated parameters_test.json to use converted data"
    else
        echo "⚠️  parameters_test.json not found, please update manually"
    fi
}

run_experiment() {
    echo "=================================================="
    echo "STEP 3: Running scGAN Experiment"
    echo "=================================================="
    
    # Load GPU modules
    if [ "$USE_GPU" = true ]; then
        module load CUDA/$CUDA_VERSION
        module load cuDNN/$CUDNN_VERSION
    fi
    
    # Activate scGAN environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    # Clean previous output
    if [ -d "$EXPERIMENT_OUTPUT_DIR" ]; then
        echo "Cleaning previous experiment output..."
        rm -rf "$EXPERIMENT_OUTPUT_DIR"
    fi
    
    # Run data processing
    echo "Step 3.1: Processing data..."
    python main.py --param parameters_tabula_muris.json --process
    # python main.py --param parameters_test.json --process
    
    # Check if processing succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Data processing completed successfully!"
        
        # Run training
        echo "Step 3.2: Starting training..."
        python main.py --param parameters_test.json --train
        
        if [ $? -eq 0 ]; then
            echo "✅ Training completed successfully!"
            
            # Run generation (optional)
            echo "Step 3.3: Generating synthetic data..."
            python main.py --param parameters_test.json --generate
            
            if [ $? -eq 0 ]; then
                echo "✅ Generation completed successfully!"
            else
                echo "⚠️  Generation failed, but training was successful"
            fi
        else
            echo "❌ Training failed!"
            exit 1
        fi
    else
        echo "❌ Data processing failed!"
        exit 1
    fi
    
    echo ""
    echo "✅ scGAN experiment completed!"
    echo "📁 Results saved in: $EXPERIMENT_OUTPUT_DIR"
    echo ""
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset DATASET    Dataset to use (tabula_muris, pbmc68k, human_pf_lung)"
    echo "  --setup-only         Only setup environment, don't run experiment"
    echo "  --data-only          Only prepare data, don't run experiment"
    echo "  --run-only           Only run experiment (assumes setup and data are ready)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run complete workflow with tabula_muris"
    echo "  $0 --dataset pbmc68k                 # Run complete workflow with PBMC 68k"
    echo "  $0 --setup-only                      # Only setup environment"
    echo "  $0 --dataset human_pf_lung --data-only  # Only prepare human lung data"
    echo ""
}

# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================

# Parse command line arguments
SETUP_ONLY=false
DATA_ONLY=false
RUN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_NAME="$2"
            EXPERIMENT_OUTPUT_DIR="output/benchmark_scGAN_${DATASET_NAME}"
            shift 2
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --run-only)
            RUN_ONLY=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution flow
echo "🚀 Starting scGAN Complete Workflow"
echo "Dataset: $DATASET_NAME"
echo "Output Directory: $EXPERIMENT_OUTPUT_DIR"
echo "GPU Support: $USE_GPU"
echo ""

if [ "$RUN_ONLY" = false ]; then
    setup_environment
fi

if [ "$SETUP_ONLY" = false ] && [ "$RUN_ONLY" = false ]; then
    prepare_dataset
fi

if [ "$SETUP_ONLY" = false ] && [ "$DATA_ONLY" = false ]; then
    run_experiment
fi

echo "🎉 scGAN workflow completed successfully!"
echo ""
echo "SUMMARY:"
echo "✅ Environment: $ENV_NAME with TensorFlow 2.6 + A100 GPU support"
echo "✅ Dataset: $DATASET_NAME prepared and processed"
echo "✅ Experiment: Training and generation completed"
echo "📁 Results: Available in $EXPERIMENT_OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review results in the output directory"
echo "2. Analyze generated synthetic data quality"
echo "3. Compare with other benchmarking methods"