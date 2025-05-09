"""
Data preparation module for Bathroom Interior Design Style Transfer project.
Handles dataset loading, preprocessing, and train/val/test splitting.
"""
import os
import shutil
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from config import *


def create_dataset_splits(dataset_path=DATASET_PATH, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                         source_style=DEFAULT_SOURCE_STYLE, target_style=DEFAULT_TARGET_STYLE):
    """
    Create train/val/test splits from the bathroom dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        train_ratio: Ratio of images to use for training
        val_ratio: Ratio of images to use for validation
        test_ratio: Ratio of images to use for testing
        source_style: Source style for transfer
        target_style: Target style for transfer
        
    Returns:
        Dictionary with information about the dataset splits
    """
    # Create directories for CycleGAN training
    os.makedirs(os.path.join(CYCLEGAN_DATA_DIR, 'trainA'), exist_ok=True)  # Source style
    os.makedirs(os.path.join(CYCLEGAN_DATA_DIR, 'trainB'), exist_ok=True)  # Target style
    os.makedirs(os.path.join(CYCLEGAN_DATA_DIR, 'valA'), exist_ok=True)    # Validation A
    os.makedirs(os.path.join(CYCLEGAN_DATA_DIR, 'valB'), exist_ok=True)    # Validation B
    os.makedirs(os.path.join(CYCLEGAN_DATA_DIR, 'testA'), exist_ok=True)   # Testing A
    os.makedirs(os.path.join(CYCLEGAN_DATA_DIR, 'testB'), exist_ok=True)   # Testing B
    
    # Set the paths for both styles
    source_dir = os.path.join(dataset_path, source_style)
    target_dir = os.path.join(dataset_path, target_style)
    
    # Get all image files in both directories
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    source_images = [f for f in os.listdir(source_dir) 
                    if os.path.isfile(os.path.join(source_dir, f)) and 
                    os.path.splitext(f)[1].lower() in valid_extensions]
    
    target_images = [f for f in os.listdir(target_dir) 
                    if os.path.isfile(os.path.join(target_dir, f)) and 
                    os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"Found {len(source_images)} {source_style} images")
    print(f"Found {len(target_images)} {target_style} images")
    
    # Shuffle the images for random splitting
    random.seed(42)  # For reproducibility
    random.shuffle(source_images)
    random.shuffle(target_images)
    
    # Calculate split indices
    source_train_idx = int(len(source_images) * train_ratio)
    source_val_idx = source_train_idx + int(len(source_images) * val_ratio)
    
    target_train_idx = int(len(target_images) * train_ratio)
    target_val_idx = target_train_idx + int(len(target_images) * val_ratio)
    
    # Split the images
    source_train = source_images[:source_train_idx]
    source_val = source_images[source_train_idx:source_val_idx]
    source_test = source_images[source_val_idx:]
    
    target_train = target_images[:target_train_idx]
    target_val = target_images[target_train_idx:target_val_idx]
    target_test = target_images[target_val_idx:]
    
    print(f"{source_style} split: {len(source_train)} train, {len(source_val)} val, {len(source_test)} test")
    print(f"{target_style} split: {len(target_train)} train, {len(target_val)} val, {len(target_test)} test")
    
    # Copy the files to their respective directories
    def copy_files(file_list, src_dir, dest_dir):
        count = 0
        for filename in file_list:
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(src_path, dest_path)
            count += 1
        return count
    
    # Copy source style images
    trainA_count = copy_files(source_train, source_dir, os.path.join(CYCLEGAN_DATA_DIR, 'trainA'))
    valA_count = copy_files(source_val, source_dir, os.path.join(CYCLEGAN_DATA_DIR, 'valA'))
    testA_count = copy_files(source_test, source_dir, os.path.join(CYCLEGAN_DATA_DIR, 'testA'))
    
    # Copy target style images
    trainB_count = copy_files(target_train, target_dir, os.path.join(CYCLEGAN_DATA_DIR, 'trainB'))
    valB_count = copy_files(target_val, target_dir, os.path.join(CYCLEGAN_DATA_DIR, 'valB'))
    testB_count = copy_files(target_test, target_dir, os.path.join(CYCLEGAN_DATA_DIR, 'testB'))
    
    print(f"Successfully copied:")
    print(f"  {trainA_count} {source_style} training images")
    print(f"  {valA_count} {source_style} validation images")
    print(f"  {testA_count} {source_style} test images")
    print(f"  {trainB_count} {target_style} training images")
    print(f"  {valB_count} {target_style} validation images")
    print(f"  {testB_count} {target_style} test images")
    
    return {
        'trainA_size': trainA_count,
        'trainB_size': trainB_count,
        'valA_size': valA_count,
        'valB_size': valB_count,
        'testA_size': testA_count,
        'testB_size': testB_count
    }


class InteriorDesignDataset(Dataset):
    """
    Dataset class for loading interior design images.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                           if os.path.isfile(os.path.join(root_dir, f)) and 
                           os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


def get_data_loaders(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    """
    Creates data loaders for CycleGAN training.
    
    Returns:
        dataloader_A: DataLoader for domain A (source style)
        dataloader_B: DataLoader for domain B (target style)
        val_dataloader_A: Validation DataLoader for domain A
        val_dataloader_B: Validation DataLoader for domain B
    """
    # Define transforms with normalization
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.12), transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # For validation - no random transforms
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Create datasets
    dataset_A = InteriorDesignDataset(os.path.join(CYCLEGAN_DATA_DIR, 'trainA'), transform=transform)
    dataset_B = InteriorDesignDataset(os.path.join(CYCLEGAN_DATA_DIR, 'trainB'), transform=transform)
    val_dataset_A = InteriorDesignDataset(os.path.join(CYCLEGAN_DATA_DIR, 'valA'), transform=val_transform)
    val_dataset_B = InteriorDesignDataset(os.path.join(CYCLEGAN_DATA_DIR, 'valB'), transform=val_transform)
    
    # Create data loaders
    dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader_A = DataLoader(val_dataset_A, batch_size=batch_size, shuffle=False, num_workers=2)
    val_dataloader_B = DataLoader(val_dataset_B, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return dataloader_A, dataloader_B, val_dataloader_A, val_dataloader_B


def get_test_data_loaders(batch_size=1, image_size=IMAGE_SIZE):
    """
    Creates test data loaders for CycleGAN evaluation.
    
    Returns:
        dataloader_A: DataLoader for domain A (source style)
        dataloader_B: DataLoader for domain B (target style)
    """
    # Define transforms - no random transforms for testing
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Create datasets
    dataset_A = InteriorDesignDataset(os.path.join(CYCLEGAN_DATA_DIR, 'testA'), transform=transform)
    dataset_B = InteriorDesignDataset(os.path.join(CYCLEGAN_DATA_DIR, 'testB'), transform=transform)
    
    # Create data loaders
    dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=False, num_workers=2)
    dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return dataloader_A, dataloader_B


def visualize_dataset_samples(num_samples=5):
    """
    Visualize random samples from each style domain.
    """
    # Load data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    dataset_A = InteriorDesignDataset(os.path.join(CYCLEGAN_DATA_DIR, 'trainA'), transform=transform)
    dataset_B = InteriorDesignDataset(os.path.join(CYCLEGAN_DATA_DIR, 'trainB'), transform=transform)
    
    # Check if datasets are non-empty
    if len(dataset_A) == 0 or len(dataset_B) == 0:
        print("WARNING: One or both datasets are empty! Cannot visualize samples.")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    # Plot samples from domain A
    for i in range(num_samples):
        if len(dataset_A) > 0:
            idx = torch.randint(0, len(dataset_A), (1,)).item()
            img = dataset_A[idx]
            img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC for plotting
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
    axes[0, 0].set_title(f'Domain A: {DEFAULT_SOURCE_STYLE}', fontsize=12)
    
    # Plot samples from domain B
    for i in range(num_samples):
        if len(dataset_B) > 0:
            idx = torch.randint(0, len(dataset_B), (1,)).item()
            img = dataset_B[idx]
            img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC for plotting
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
    axes[1, 0].set_title(f'Domain B: {DEFAULT_TARGET_STYLE}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    plt.close()
    
    return 'dataset_samples.png'


if __name__ == "__main__":
    # Example usage
    print(f"Creating dataset splits for {DEFAULT_SOURCE_STYLE} and {DEFAULT_TARGET_STYLE}...")
    stats = create_dataset_splits(
        source_style=DEFAULT_SOURCE_STYLE,
        target_style=DEFAULT_TARGET_STYLE
    )
    
    print(f"Dataset statistics: {stats}")
    
    # Visualize samples
    sample_img_path = visualize_dataset_samples()
    if sample_img_path:
        print(f"Sample visualization saved to: {sample_img_path}")