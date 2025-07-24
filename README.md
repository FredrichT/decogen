# Living Room Interior Design Style Transfer

This project implements a living room interior design style transfer system using CycleGAN architecture. The system can transform living rooms between different design styles while preserving structural elements.

## Features

- Style transfer between Coastal and Eclectic living room design styles
- Preservation of structural elements (walls, furniture placement, architectural features)
- Computationally efficient implementation with support for Apple M1/M2 (MPS)
- Weights & Biases integration for experiment tracking
- Evaluation metrics for style transfer quality (LPIPS, FID)

## Project Structure

```
decogen/
├── config.py                  # Configuration parameters
├── data_preparation.py        # Dataset preparation and loading
├── evaluate.py                # Model evaluation script
├── inference.py               # Inference on new images
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── train.py                   # Training script
├── utils.py                   # Utility functions
├── dataset/                   # Raw living room images dataset
│   ├── coastal/               # Coastal style living room images (1,049 images)
│   └── eclectic/              # Eclectic style living room images (1,020 images)
└── models/                    # Model implementations
    ├── __init__.py
    ├── cycle_gan.py           # CycleGAN model implementation
    ├── losses.py              # Custom loss functions
    ├── networks.py            # Network architecture implementations
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/FredrichT/decogen.git
   cd decogen
   ```

2. Create and activate a conda environment:

   ```bash
   conda create -n style-transfer python=3.9
   conda activate style-transfer
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Configure Weights & Biases:
   ```bash
   wandb login
   ```

## Dataset

This project uses the MMIS (Multimodal Dataset for Interior Scene Visual Generation and Recognition) dataset, specifically focusing on living room images from two distinct design styles.

### MMIS Dataset Overview

The MMIS dataset is a comprehensive multimodal dataset containing nearly 160,000 interior scene images with corresponding textual descriptions and audio recordings. Each image captures various interior spaces with different styles, layouts, and furnishings.

For this project, we utilize:
- **Coastal style living rooms**: 1,049 images featuring bright, blue-toned interiors with beach-inspired elements
- **Eclectic style living rooms**: 1,020 images showcasing vibrant, colorful spaces with diverse design influences

### Design Styles

- **Coastal**: Bright, airy living rooms with blue color palettes, natural materials, and beach-inspired décor elements
- **Eclectic**: Vibrant, colorful living rooms combining diverse design elements, patterns, and furnishings from various styles

### Dataset Structure

```
dataset/
├── coastal/        # Coastal style living room images (1,049 images)
└── eclectic/       # Eclectic style living room images (1,020 images)
```

### Citation

When using this dataset in your research, please cite:

**MMIS: Multimodal Dataset for Interior Scene Visual Generation and Recognition**
- Authors: Hozaifa Kassab, Ahmed Mahmoud, Mohamed Bahaa, Ammar Mohamed, Ali Hamdi
- arXiv:2407.05980 [cs.CV]
- DOI: https://doi.org/10.48550/arXiv.2407.05980
- Dataset: https://drive.google.com/drive/folders/1FO_sNVZi757I_QBdwibPX--14O24ff0m

## Usage

The project can be used through the `main.py` entry point with three main commands: `train`, `evaluate`, and `inference`.

### Training

Train the model using the default settings:

```bash
python main.py train
```

Customize training parameters:

```bash
python main.py train --batch_size 2 --epochs 100 --source_style coastal --target_style eclectic
```

Available options:

- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--source_style`: Source style for transfer (choices: coastal, eclectic)
- `--target_style`: Target style for transfer (choices: coastal, eclectic)

### Evaluation

Evaluate a trained model:

```bash
python main.py evaluate --checkpoint_dir checkpoints --epoch 100 --output_dir evaluation_results
```

Options:

- `--checkpoint_dir`: Directory containing model checkpoints (default: checkpoints)
- `--epoch`: Epoch of the model to evaluate (default: 100)
- `--output_dir`: Directory to save evaluation results (default: evaluation_results)

### Inference

Apply style transfer to new living room images:

```bash
python main.py inference --input path/to/your/image.jpg --output_dir inference_results --direction AtoB
```

For batch processing a directory of images:

```bash
python main.py inference --input path/to/image/directory --output_dir inference_results --direction AtoB
```

Options:

- `--input`: Path to input image or directory
- `--output_dir`: Directory to save output images (default: inference_results)
- `--direction`: Direction of style transfer (AtoB: source→target, BtoA: target→source)
- `--checkpoint_dir`: Directory containing model checkpoints (default: checkpoints)
- `--epoch`: Epoch of the model to use (default: 100)

## Model Architecture

The implementation uses CycleGAN architecture with several optimizations:

### Enhanced Generator

- Residual blocks for better feature preservation
- Improved downsampling/upsampling to maintain spatial information

### Multi-scale Discriminator

- Focuses on local patches at different scales
- Better preservation of texture and detail

### Custom Loss Functions

- Cycle consistency loss to ensure content preservation
- Structural preservation loss to maintain architectural elements
- Style transfer loss for aesthetic transformation

## Results

Sample results will be saved in the `samples` directory during training. Full evaluation results, including LPIPS and FID scores, will be saved in the `evaluation_results` directory when running the evaluation script.

## Implementation

This CycleGAN implementation focuses on style transfer between living room interior designs, utilizing the large-scale MMIS dataset for training robust generative models.
