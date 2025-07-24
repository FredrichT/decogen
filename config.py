"""
Configuration parameters for the Living Room Interior Design Style Transfer project.
"""
import os
import torch

# Dataset configuration
DATASET_PATH = "dataset"  # Updated to point to the dataset at project root
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

# Training configuration
SAVE_MODEL_EVERY = 10  # Save model every n epochs
SAVE_IMAGE_EVERY = 5   # Save sample images every n epochs
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"

# Available living room styles
STYLES = ["coastal", "eclectic"]

# Default source and target styles for transfer
DEFAULT_SOURCE_STYLE = "coastal"
DEFAULT_TARGET_STYLE = "eclectic"

# Weights and Biases configuration
WANDB_PROJECT = "living-room-design-style-transfer"
WANDB_ENTITY = None  # Change to your wandb username if needed