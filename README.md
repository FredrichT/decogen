# Bathroom Interior Design Style Transfer

This project implements a bathroom interior design style transfer system using a CycleGAN architecture optimized based on the research paper by Fu (2022). The system can transform bathrooms from one design style to another while preserving structural elements.

## Features

- Style transfer between different bathroom design styles (modern, minimalist, industrial, boho, and scandinavian)
- Preservation of structural elements (walls, fixtures, architectural features)
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
├── dataset/                   # Raw bathroom images dataset
│   └── bathroom/              # Contains subdirectories for each style
│       ├── boho/
│       ├── industrial/
│       ├── minimalist/
│       ├── modern/
│       └── scandinavian/
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

This project uses a curated subset of bathroom interior design images originally sourced from the "Interior Design Images and Metadata Dataset from Pinterest" (available on Kaggle as "galinakg/interior-design-images-and-metadata").

### Dataset Modifications

The original dataset contained multiple room types (bathroom, bedroom, kitchen, living_room) with various design styles, along with CSV metadata files. For this implementation:

- We focused exclusively on bathroom images for better model specialization
- The dataset was cleaned to remove:
  - Images containing text/writing
  - Images showing only colors or palettes
  - Images with single objects rather than entire rooms
- CSV metadata files were removed to work directly with image files
- The cleaned images were organized into style-specific directories following the structure outlined in the README

### Design Styles

Each bathroom image is categorized into one of five interior design styles:

- **Boho**: Bohemian style featuring eclectic, global influences with casual and layered aesthetics
- **Industrial**: Raw, utilitarian style with exposed materials, metal fixtures, and minimally finished surfaces
- **Minimalist**: Clean, uncluttered spaces with simple color palettes and essential fixtures
- **Modern**: Contemporary designs with clean lines, updated materials, and functional aesthetics
- **Scandinavian**: Light, airy spaces with natural materials, neutral colors, and functional simplicity

### Dataset Structure

The dataset is organized as follows:

```
dataset/bathroom/
├── boho/           # Boho style bathroom images
├── industrial/     # Industrial style bathroom images
├── minimalist/     # Minimalist style bathroom images
├── modern/         # Modern style bathroom images
└── scandinavian/   # Scandinavian style bathroom images
```

Each image should be a clean, high-quality photograph of a bathroom interior in the corresponding style.

### Citation

When using this dataset approach in your research or projects, please cite the original dataset:
"Interior Design Images and Metadata Dataset from Pinterest" (Kaggle: galinakg/interior-design-images-and-metadata)

## Usage

The project can be used through the `main.py` entry point with three main commands: `train`, `evaluate`, and `inference`.

### Training

Train the model using the default settings:

```bash
python main.py train
```

Customize training parameters:

```bash
python main.py train --batch_size 2 --epochs 100 --source_style modern --target_style minimalist
```

Available options:

- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--source_style`: Source style for transfer (choices: boho, industrial, minimalist, modern, scandinavian)
- `--target_style`: Target style for transfer (choices same as source)

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

Apply style transfer to new bathroom images:

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

The implementation is based on an optimized CycleGAN architecture from Fu (2022) with several improvements:

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

## Research Paper

This implementation is based on the research paper:

**Digital Image Art Style Transfer Algorithm Based on CycleGAN**

- Author: Xuhui Fu
- Published: January 13, 2022
- DOI: [https://doi.org/10.1155/2022/6075398](https://doi.org/10.1155/2022/6075398)
- Part of Special Issue: Computational Intelligence in Image and Video Analysis
