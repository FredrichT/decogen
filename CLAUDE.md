# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a living room interior design style transfer system using CycleGAN architecture. The system transforms living rooms between Coastal and Eclectic design styles while preserving structural elements.

## Key Commands

### Training
```bash
# Start training with default settings
python main.py train

# Train with custom parameters
python main.py train --batch_size 2 --epochs 100 --source_style coastal --target_style eclectic

# Alternative training (direct script)
python train.py
```

### Evaluation
```bash
# Evaluate trained model
python main.py evaluate --checkpoint_dir checkpoints --epoch 100 --output_dir evaluation_results

# Direct evaluation script
python evaluate.py --checkpoint_dir checkpoints --epoch 100
```

### Inference
```bash
# Style transfer on single image
python main.py inference --input path/to/image.jpg --output_dir inference_results --direction AtoB

# Batch processing directory
python main.py inference --input path/to/directory --output_dir inference_results --direction AtoB

# Direct inference script
python inference.py --input path/to/image.jpg --direction AtoB
```

### Environment Setup
```bash
# Create conda environment
conda create -n style-transfer python=3.9
conda activate style-transfer

# Install dependencies
pip install -r requirements.txt

# Optional: Configure Weights & Biases
wandb login
```

## Code Architecture

### Core Components
- **main.py**: Entry point with command-line interface for train/evaluate/inference
- **config.py**: Central configuration file with all hyperparameters and paths
- **models/**: Neural network implementations
  - `cycle_gan.py`: Main CycleGAN model implementation
  - `networks.py`: Generator and discriminator architectures
  - `losses.py`: Custom loss functions (GAN, cycle, identity, structural)
- **data_preparation.py**: Dataset loading and preprocessing
- **utils.py**: Utility functions for image processing and visualization

### Model Architecture
- **Enhanced ResNet Generators**: A→B and B→A transformations with residual blocks
- **Multi-scale Discriminators**: Patch-based discriminators for better texture preservation
- **Custom Loss Functions**:
  - Cycle consistency loss (λ=10.0)
  - ~~Identity loss~~ (removed - was too constraining)
  - ~~Structural preservation loss~~ (removed - was too constraining)
- **Image Buffer**: Reduces training oscillation by using historical fake images
- **Asymmetric Learning Rate Scheduling**:
  - Generator: CosineAnnealingWarmRestarts (T_0=50, T_mult=2, restarts at epochs 50, 150, 350)
  - Discriminator: CosineAnnealingLR (steady decay to prevent over-training)
- **Configuration Integration**: All hyperparameters centralized in config.py

### Data Structure
```
dataset/
├── coastal/        # Coastal style living room images (1,049 images)
└── eclectic/       # Eclectic style living room images (1,020 images)
```

### Key Configuration Variables
- `DEVICE`: Auto-detects MPS (Apple Silicon), CUDA, or CPU
- `BATCH_SIZE`: 8 (increased from 4)
- `IMAGE_SIZE`: 256x256 pixels
- `NUM_EPOCHS`: 500 epochs (increased from 100)
- `LEARNING_RATE_G`: 0.0002 (generator learning rate)
- `LEARNING_RATE_D`: 0.0001 (discriminator learning rate - lower to prevent dominance)
- `LAMBDA_CYCLE`: 10.0 (cycle consistency weight)
- `BETA1`: 0.5, `BETA2`: 0.999 (Adam optimizer parameters)
- `STYLES`: ["coastal", "eclectic"]
- `DEFAULT_SOURCE_STYLE`: "coastal"
- `DEFAULT_TARGET_STYLE`: "eclectic"

### Training Pipeline
1. **Data Preparation**: Creates train/val/test splits and data loaders
2. **Model Initialization**: Sets up generators, discriminators, and loss functions
3. **Training Loop**: Alternating generator and discriminator updates
4. **Checkpointing**: Saves models every 10 epochs
5. **Visualization**: Generates sample images every 5 epochs
6. **Logging**: Weights & Biases integration for experiment tracking

### Evaluation Metrics
- **LPIPS**: Perceptual similarity for cycle consistency and structural preservation
- **FID**: Fréchet Inception Distance for image quality assessment
- **Visual Samples**: Generated comparison grids with original, transferred, and reconstructed images

### Device Optimization
- **Apple Silicon**: Uses MPS (Metal Performance Shaders) for M1/M2 acceleration
- **CUDA**: Automatic GPU detection and usage
- **CPU Fallback**: Warning shown for slow CPU-only training

### File Organization
- `checkpoints/`: Model weights saved as `{G_A/G_B}_net_{epoch}.pth`
- `samples/`: Training progress images saved as `epoch_{epoch}.png`
- `evaluation_results/`: LPIPS/FID scores and test samples
- `inference_results/`: Style transfer outputs with comparison grids
- `data/cyclegan_data/`: Processed dataset splits (train/val/test)

## Development Notes

### Testing
No specific test framework detected. Evaluate model performance using:
- `python main.py evaluate` for quantitative metrics
- `python main.py inference` for qualitative assessment

### Memory Management
- Batch size increased to 8 (adjust based on GPU memory)
- Image buffer size (50) helps stabilize training
- Model checkpoints are saved for generators only to save space

### Error Handling
- Automatic device detection with fallback hierarchy
- StopIteration handling for different dataset sizes
- Directory creation for all output paths

### Recent Configuration Improvements (2025-01-24)
- **Centralized Configuration**: All hyperparameters moved from hard-coded values in `cycle_gan.py` to `config.py`
- **Asymmetric Learning Rates**: Generator (0.0002) vs Discriminator (0.0001) to address discriminator dominance
- **Advanced Scheduling**: 
  - Generator uses warm restarts for periodic learning rate boosts
  - Discriminator uses steady decay to prevent over-training
- **Extended Training**: Increased from 100 to 500 epochs for better convergence
- **Simplified Loss**: Removed constraining identity and structural losses, keeping only GAN + cycle consistency

### Style Transfer Directions
- `AtoB`: Source style → Target style (e.g., coastal → eclectic)
- `BtoA`: Target style → Source style (e.g., eclectic → coastal)

This implementation emphasizes architectural preservation while enabling style transformation, making it suitable for interior design applications.

## Claude Code Workflow

When working on this project, follow these 7 rules:

1. **Think and Plan**: First think through the problem, read the codebase for relevant files, and write a plan to `tasks/todo.md`.

2. **Create Todo List**: The plan should have a list of todo items that you can check off as you complete them.

3. **Get Approval**: Before you begin working, check in with me and I will verify the plan.

4. **Execute and Track**: Then, begin working on the todo items, marking them as complete as you go.

5. **Communicate Changes**: Please every step of the way just give me a high level explanation of what changes you made.

6. **Keep It Simple**: Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.

7. **Review and Summarize**: Finally, add a review section to the `todo.md` file with a summary of the changes you made and any other relevant information.

8. **Track Feedback Progress**: For each feedback section below, update the status and maintain a history of what has been done to address the feedback.

## Teacher Feedback & Improvements

**Grade Received**: 12/20

**Positive Feedback**:
- Clean code quality
- Well-written and complete report

**Areas for Improvement**:

### 1. Dataset Size Issue
**Problem**: Current dataset (~100 images per style) is insufficient for training generative models.

**Required Actions**:
- Find larger datasets (typical examples: landscapes and artwork)
- Increase dataset size significantly for better generative model training

**Status**: ✅ **COMPLETED**
**History**: 
- Initial dataset created with ~100 images per bathroom style
- **2025-01-24**: Successfully migrated to MMIS dataset with 2,069 living room images:
  - Coastal style: 1,049 images
  - Eclectic style: 1,020 images
- Dataset structure updated to `dataset/coastal/` and `dataset/eclectic/`
- All documentation (README.md, config.py, CLAUDE.md) updated to reflect new dataset

### 2. Loss Function Configuration
**Problem**: The Fu (2022) paper reference uses overly constraining auxiliary objectives (Structural Loss and Identity Loss) that force generators to behave like identity functions, preventing meaningful style transfer.

**Evidence of Problem**:
- GAN losses are small compared to auxiliary losses
- Discriminator losses are small (easy to discriminate real vs fake)
- Models only modify image tint slightly instead of true style transfer

**Required Actions**:
- Use more classical CycleGAN loss configuration
- Reduce or remove overly constraining identity regularization
- Rebalance loss weights to allow more generative freedom

**Status**: ✅ **COMPLETED**
**History**:
- Current implementation uses λ_identity=0.5 and λ_structure=10.0
- These values are too constraining and prevent style transfer
- **2025-01-24**: Completely removed identity and structural losses:
  - Removed `LAMBDA_IDENTITY` and `LAMBDA_STRUCTURE` from config.py
  - Removed `IdentityLoss` and `StructuralLoss` implementations from CycleGAN model
  - Removed all identity mapping computations from forward pass
  - Updated train.py, inference.py, and evaluate.py to remove references
  - `LAMBDA_CYCLE = 10.0` (kept standard cycle consistency weight)
- Configuration now uses pure classical CycleGAN: GAN loss + cycle consistency only
- Should allow much more meaningful style transfer between domains

### 3. Model Evaluation & Discussion
**Problem**: Current evaluation and discussion are insufficient.

**Required Actions**:
- Improve evaluation methodology
- Provide better analysis of results
- Discuss limitations and improvements more thoroughly

**Status**: Not Started
**History**:
- Current evaluation uses LPIPS and FID metrics
- Need more comprehensive analysis of style transfer quality
- Discussion section needs enhancement

### 4. True Style Transfer Achievement
**Problem**: Current results show only tint modifications rather than genuine style transfer.

**Required Actions**:
- Achieve meaningful style transformation between different artistic/architectural styles
- Demonstrate clear visual differences in generated images
- Move beyond simple color/tint adjustments

**Status**: Not Started
**History**:
- Current bathroom style transfer shows minimal visual changes
- Need to target more distinct style domains (e.g., landscapes to artwork)
- Requires both dataset and loss function improvements