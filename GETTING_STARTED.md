# Getting Started with Bathroom Design Style Transfer

This guide will walk you through the process of setting up and running the Bathroom Design Style Transfer project on your machine.

## Initial Setup

### 1. Prerequisites

Before starting, make sure you have the following installed:

- Python 3.8 or higher
- Conda or Miniconda (recommended for managing environments)

### 2. Environment Setup

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/bathroom-style-transfer.git
cd bathroom-style-transfer
```

Create a new conda environment with Python 3.9:

```bash
conda create -n style-transfer python=3.9
conda activate style-transfer
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Dataset Structure

The project expects a specific dataset structure focused on bathroom interior designs:

```
dataset/bathroom/
├── boho/           # Boho style bathroom images
├── industrial/     # Industrial style bathroom images
├── minimalist/     # Minimalist style bathroom images
├── modern/         # Modern style bathroom images
└── scandinavian/   # Scandinavian style bathroom images
```

Each subdirectory should contain high-quality images of bathrooms in that particular style.

### 4. Project Structure Setup

Run the setup script to create the necessary directory structure:

```bash
python setup_project.py
```

### 5. Weights & Biases Setup (Optional)

If you want to track your experiments with Weights & Biases:

```bash
wandb login
```

Follow the instructions to authenticate your wandb account.

## Running the Project

### Training a Model

To train a model with the default settings:

```bash
python main.py train
```

This will automatically:

1. Split your bathroom images into train, validation, and test sets
2. Train the model and save checkpoints
3. Generate sample images to visualize progress

By default, the model will transfer from "modern" style to "minimalist" style. You can customize this with:

```bash
python main.py train --batch_size 2 --epochs 50 --source_style modern --target_style boho
```

The training progress will be displayed in the terminal, and if you've set up wandb, you can also monitor it in your wandb dashboard.

### Evaluating a Trained Model

After training, you can evaluate the model's performance:

```bash
python main.py evaluate --epoch 50
```

This will generate various metrics including:

- LPIPS scores for perceptual similarity
- FID scores for generated image quality
- Sample visualizations

Results will be saved in the `evaluation_results` directory.

### Applying Style Transfer to New Images

To apply style transfer to your own bathroom images:

```bash
python main.py inference --input path/to/your/image.jpg --output_dir inference_results
```

By default, this will transform from source style (modern) to target style (minimalist). To reverse the direction:

```bash
python main.py inference --input path/to/your/image.jpg --direction BtoA
```

You can also process an entire directory of images:

```bash
python main.py inference --input path/to/image/directory --output_dir my_transformed_bathrooms
```

## Troubleshooting

### Common Issues on Mac M1/M2

1. **PyTorch MPS Issues**: If you encounter errors related to MPS, try updating your PyTorch installation:

   ```bash
   pip install --upgrade torch torchvision
   ```

2. **Memory Errors**: For large images or batch sizes, you might encounter memory errors. Try reducing the batch size:

   ```bash
   python main.py train --batch_size 2
   ```

### Other Common Issues

1. **Missing Directories**: If you encounter errors about missing directories, run:

   ```bash
   python setup_project.py
   ```

2. **CUDA Out of Memory**: If using a CUDA device and encountering memory errors, reduce the batch size or image size in `config.py`.

## Next Steps

- **Experiment with different styles**: Try transferring between different bathroom design styles like industrial, boho, or scandinavian.
- **Fine-tune the model**: Adjust hyperparameters like `lambda_cycle`, `lambda_identity`, and `lambda_structure` in `config.py`.
- **Extend to your own dataset**: Add more bathroom style images to your dataset or adapt the code for other room types.

## Support

If you encounter issues or have questions, please open an issue on the GitHub repository.
