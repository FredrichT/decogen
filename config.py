"""
Configuration parameters for the Bathroom Interior Design Style Transfer project.
"""
import os
import torch

# Dataset configuration
DATASET_PATH = "dataset/bathroom"  # Updated to point to the bathroom dataset at project root
DATA_DIR = "data"
CYCLEGAN_DATA_DIR = os.path.join(DATA_DIR, "cyclegan_data")

# Model configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
BATCH_SIZE = 4
IMAGE_SIZE = 256
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_EPOCHS = 100
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 0.5
LAMBDA_STRUCTURE = 10.0  # Weight for structural preservation loss

# Training configuration
SAVE_MODEL_EVERY = 10  # Save model every n epochs
SAVE_IMAGE_EVERY = 5   # Save sample images every n epochs
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"

# Available bathroom styles
STYLES = ["boho", "industrial", "minimalist", "modern", "scandinavian"]

# Default source and target styles for transfer
DEFAULT_SOURCE_STYLE = "industrial"
DEFAULT_TARGET_STYLE = "scandinavian"

# Weights and Biases configuration
WANDB_PROJECT = "bathroom-design-style-transfer"
WANDB_ENTITY = None  # Change to your wandb username if needed