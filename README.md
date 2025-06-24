# Modular Fine-tuning Pipeline for Scam Detection

This repository contains a modular Python pipeline for fine-tuning large language models for scam detection tasks. The pipeline features a user-friendly command-line interface and comprehensive comparison tools for evaluating model performance.

## Architecture

The pipeline consists of the following modular components:

- **`main.py`**: Main CLI interface with interactive menu system
- **`config.py`**: Configuration classes for all pipeline parameters
- **`model_loader.py`**: Model loading and LoRA adapter setup
- **`data_processor.py`**: Data loading, conversation formatting, and sample extraction
- **`trainer.py`**: Model training with SFTTrainer
- **`inference.py`**: Model inference, text generation, and sample comparison
- **`model_saver.py`**: Model saving in various formats (LoRA, merged, GGUF)
- **`train_pipeline.py`**: Core orchestration pipeline (used by CLI)

## Prerequisites

- Python 3.11+
- CUDA-compatible GPU with sufficient [VRAM](https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements) (click the link for more details) 
- Linux/WSL environment
- Conda/Miniconda (recommended for CUDA management)

PS: I am not sure if Unsloth works with MPS/MLX of Mac, feel free to try it out and let me know if it works.


## Installation (Using Conda)

1. Clone the repository:
```bash
git clone https://github.com/Chen-zexi/Scam-SFT.git
cd Scam-SFT
```

2. Create a .env file:

Put your HuggingFace token in the .env file. You can get it from [here](https://huggingface.co/settings/tokens).
```bash
cp .env.example .env
```

3. Create and activate a conda environment:
```bash
# Create environment with Python 3.12
conda create -n scam-tuning python=3.12 -y
conda activate scam-tuning
```

4. Install CUDA toolkit (choose version based on your GPU driver):
```bash
# For CUDA 12.8 (latest and recommended)
conda install nvidia/label/cuda-12.8.0::cuda-toolkit -y

# Note: Use nvcc --version to check your CUDA version or use nvidia-smi to check your GPU driver version.
```

5. Install PyTorch from official website:
   
   Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select your configuration:
   - **Compute Platform**: Choose your CUDA version (e.g., CUDA 12.8)
   - **OS**: Linux
   - **Package**: Conda
   - **Language**: Python

   Example for CUDA 12.8:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
   ```

6. Install other dependencies:
```bash
pip install -r requirements.txt
```

### Verify Installation

After installation, verify your setup:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output should show:
- PyTorch version (e.g., 2.7.0+cu128)
- CUDA available: True
- CUDA version matching your installation

## Data Format

The pipeline expects a CSV file with the following columns:
- `original_content`: The text content to classify
- `class`: The classification label ("Scam" or "Legitimate")
- `explanation`: Explanation for the classification

Example:
```csv
original_content,class,explanation
"Click here to claim your prize!",Scam,"This is a typical phishing attempt asking users to click suspicious links"
"Your order has been confirmed",Legitimate,"This is a standard order confirmation message"
```

## Usage

### Main CLI Interface

Run the main application:

```bash
python main.py
```

This will present you with an interactive menu:

```
======================================================================
           SCAM DETECTION MODEL - TRAINING & INFERENCE
======================================================================

Please select an option:
1. Run Supervised Fine-Tuning (SFT)
2. Inference with Base Model
3. Inference with Fine-tuned Model
4. Check Environment
5. Exit
```

### Option 1: Supervised Fine-Tuning (SFT)

Runs the complete fine-tuning pipeline:
- Loads the base model defined in config.py
- Processes training data
- Performs fine-tuning with LoRA adapters
- Saves the trained model with timestamps
- Optionally runs sample comparison if enabled

### Option 2: Inference with Base Model

Tests the original model without fine-tuning:
- Loads the base model (DeepSeek-R1-Distill-Qwen-14B)
- Runs sample comparison on dataset examples (if enabled)
- Provides interactive classification interface

### Option 3: Inference with Fine-tuned Model

Tests fine-tuned models:
- Automatically discovers available fine-tuned models in output directory
- Displays model information (timestamp, training steps, final loss)
- Allows user selection of specific models
- Runs sample comparison using the same samples as base model
- Provides interactive classification interface

### Option 4: Check Environment

Comprehensive environment diagnostics:
- **System Information**: Python version, platform
- **GPU & CUDA Information**: CUDA availability, versions, GPU specifications
- **Library Versions**: torch, unsloth, transformers, bitsandbytes, triton, etc.
- **Attention Mechanisms**: Flash Attention, xformers, PyTorch native SDPA
- **PyTorch Configuration**: CUDA build status, cuDNN, MPS, XPU support

## Sample Comparison Feature

The pipeline includes an advanced sample comparison system:

### Configuration
```python
# In config.py InferenceConfig
run_sample_comparison: bool = True    # Enable/disable sample comparison
sample_random_state: int = 42         # Random seed for reproducible sampling
```

### How It Works
1. **Data Sampling**: Automatically samples one "Scam" and one "Legitimate" example from your dataset
2. **Reproducible Results**: Uses controlled random state for consistent sampling
3. **Base Model Testing**: Tests base model performance on sampled data
4. **Fine-tuned Comparison**: Tests fine-tuned models on the same samples
5. **Detailed Output**: Shows content, true class, explanation, and model predictions

### Sample Output
```
============================================================
SAMPLE COMPARISON - BASE MODEL
============================================================

TESTING ON SCAM SAMPLE
============================================================
SAMPLE CLASSIFICATION - TRUE CLASS: Scam
============================================================
Content: Subject: letter of intent Body: letter of intent...
True Class: Scam
Explanation: This content is clearly a scam following the classic...

MODEL PREDICTION:
--------------------------------------------------
[Model response here]

TESTING ON LEGITIMATE SAMPLE
[Similar format for legitimate sample]

============================================================
COMPARISON SUMMARY - BASE MODEL
============================================================
Scam Sample - True: Scam | Predicted: [prediction summary]
Legitimate Sample - True: Legitimate | Predicted: [prediction summary]
============================================================
```

## Configuration

All configurations are centralized in `config.py`. Key configuration classes:

### ModelConfig
- `model_name`: Base model to use (default: "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit")
- `max_seq_length`: Maximum sequence length (default: 2048)
- `load_in_4bit`: Enable 4-bit quantization (default: True)

### LoRAConfig
- `r`: LoRA rank (default: 32)
- `lora_alpha`: LoRA alpha parameter (default: 32)
- `target_modules`: Modules to apply LoRA to

### TrainingConfig
- `max_steps`: Maximum training steps (default: 30)
- `learning_rate`: Learning rate (default: 2e-4)
- `per_device_train_batch_size`: Batch size per device (default: 2)

### DataConfig
- `data_path`: Path to training data (default: "data/unified_error_dataset_annotated.csv")
- `shuffle_seed`: Random seed for shuffling (default: 3407)

### InferenceConfig
- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Nucleus sampling parameter (default: 0.8)
- `max_new_tokens`: Maximum tokens to generate (default: 1024)
- `run_sample_comparison`: Enable sample comparison (default: True)
- `sample_random_state`: Random seed for sampling (default: 42)

### SaveConfig
- `base_output_dir`: Base directory for all outputs (default: "output")
- `use_timestamp`: Whether to create timestamped subdirectories (default: True)
- `timestamp_format`: Format for timestamp directories (default: "%Y%m%d_%H%M%S")
- `save_training_config`: Whether to save training configuration JSON (default: True)
- `save_training_summary`: Whether to save training summary JSON (default: True)

## Output Structure

After training, the pipeline generates a timestamped directory structure:

```
output/
└── 20240115_143022/                    # Timestamp directory
    ├── training_config.json            # Complete training configuration
    ├── training_summary.json           # Training statistics and summary
    └── lora_unsloth_DeepSeek_R1_Distill_Qwen_14B_unsloth_bnb_4bit/
        ├── adapter_config.json         # LoRA adapter configuration
        ├── adapter_model.safetensors   # LoRA adapter weights
        ├── tokenizer_config.json       # Tokenizer configuration
        ├── tokenizer.json              # Tokenizer files
        ├── special_tokens_map.json     # Special tokens mapping
        └── README.md                   # Model card and usage instructions
```

### Metadata Files

- **`training_config.json`**: Contains all configuration parameters used for training
- **`training_summary.json`**: Contains training statistics, memory usage, and run metadata

## Advanced Usage

### Direct Pipeline Access

For advanced users, the pipeline can be accessed directly:

```python
from src.train_pipeline import FineTuningPipeline, create_default_configs
from src.config import *

# Create configurations
configs = create_default_configs()
model_config, lora_config, training_config, data_config, inference_config, save_config = configs

# Customize configurations as needed
training_config.max_steps = 100
data_config.data_path = "path/to/custom/data.csv"

# Create and run pipeline
pipeline = FineTuningPipeline(
    model_config=model_config,
    lora_config=lora_config,
    training_config=training_config,
    data_config=data_config,
    inference_config=inference_config,
    save_config=save_config
)

results = pipeline.run_full_pipeline()
```

### Batch Inference

```python
from src.inference import ModelInference
from src.model_loader import ModelLoader

# Load model
model_loader = ModelLoader(model_config, lora_config)
model, tokenizer = model_loader.load_pretrained_lora("path/to/fine_tuned_model")

# Create inference handler
inference = ModelInference(model, tokenizer, inference_config)

# Batch classify
contents = ["Text 1", "Text 2", "Text 3"]
results = inference.batch_classify(contents)
```

## Environment Verification

Before training, use the environment check (Option 4) to verify your setup:

- CUDA availability and version compatibility
- Required library installations and versions
- GPU specifications and memory availability
- Attention mechanism optimizations (Flash Attention, xformers)

Environment I am using (wsl2 ubuntu 24.04 in windows 11):

```
============================================================
ENVIRONMENT CHECK
============================================================

SYSTEM INFORMATION
------------------------------
Python Version: 3.12.11
Platform: linux

GPU & CUDA INFORMATION
------------------------------
CUDA Available: True
CUDA Version (nvcc): 12.9
PyTorch CUDA Version: 12.8
Number of GPUs: 2

LIBRARY VERSIONS
------------------------------
torch          : 2.8.0.dev20250621+cu128
unsloth        : 2025.6.4
transformers   : 4.51.3
bitsandbytes   : 0.47.0.dev0
triton         : 3.3.1
accelerate     : 1.8.1
peft           : 0.15.2
datasets       : 3.6.0
trl            : 0.15.2
xformers       : Not installed
flash_attn     : 2.8.0.post2

ATTENTION MECHANISMS
------------------------------
Flash Attention: Available
  Version: 2.8.0.post2
  CUDA Compatible: Yes
xformers: Not Available
PyTorch Native SDPA: Available (PyTorch >= 2.0)

PYTORCH CONFIGURATION
------------------------------
PyTorch Built with CUDA: True
cuDNN Available: True
cuDNN Version: 91002
MPS Available (Apple): False
XPU Available (Intel): False
```

## Performance Tips

1. **Memory Optimization**:
   - Use 4-bit quantization for large models
   - Adjust batch size based on available VRAM
   - Enable gradient checkpointing
   - Install Flash Attention or xformers for memory-efficient attention

2. **Training Speed**:
   - Use appropriate LoRA rank (16-32 for most cases)
   - Optimize gradient accumulation steps

3. **Model Quality**:
   - Ensure diverse, high-quality training data
   - Monitor training loss through sample comparison
   - Experiment with learning rates and training steps

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch size in TrainingConfig
   - Enable gradient checkpointing
   - Use smaller LoRA rank
   - Install memory-efficient attention (Flash Attention/xformers)

2. **Slow Training**:
   - Check GPU utilization with environment check
   - Increase batch size if memory allows
   - Verify CUDA installation and version
   - Do not use MOE models as Unsloth has poor support at the moment

3. **Poor Model Performance**:
   - Check data quality and format
   - Use sample comparison to evaluate performance
   - Adjust learning rate and training steps
   - Ensure balanced dataset with both classes

### CUDA Debug

Enable more detailed debug by setting environment variables:

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
```

## Model Evaluation

The pipeline provides comprehensive model evaluation through:

1. **Sample Comparison**: Systematic testing on dataset samples
2. **Interactive Testing**: Real-time classification testing
3. **Performance Metrics**: Training loss tracking and comparison
4. **Before/After Analysis**: Direct comparison of base vs fine-tuned models