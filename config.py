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
BATCH_SIZE = 8
IMAGE_SIZE = 256

# Separate learning rates for G and D
LEARNING_RATE_G = 0.0002 # Generator learning rate
LEARNING_RATE_D = 0.0001  # Discriminator learning rate (lower)

BETA1 = 0.5
BETA2 = 0.999
NUM_EPOCHS = 500
LAMBDA_CYCLE = 5.0

# Discriminator regularization
DISCRIMINATOR_DROPOUT = 0.2  # Dropout rate for discriminator to prevent overfitting

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