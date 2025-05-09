"""
Utility functions for the Interior Design Style Transfer project.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def tensor_to_image(tensor):
    """
    Convert a tensor to a numpy image for visualization.
    
    Parameters:
        tensor (torch.Tensor) -- input tensor of shape (C, H, W)
        
    Returns:
        numpy.ndarray -- image array of shape (H, W, C)
    """
    # Convert tensor to numpy array
    image = tensor.detach().cpu().numpy()
    
    # Reshape from (C, H, W) to (H, W, C)
    image = image.transpose(1, 2, 0)
    
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2.0
    
    # Clip values to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def save_images(samples, save_path):
    """
    Save a grid of images from the samples dictionary.
    
    Parameters:
        samples (dict) -- dictionary containing tensors of images
        save_path (str) -- path to save the grid image
    """
    # Select images to include in the grid
    keys_to_show = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
    
    # Create the figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each image
    for i, key in enumerate(keys_to_show):
        # Get the first image from the batch
        img = tensor_to_image(samples[key][0])
        axes[i].imshow(img)
        axes[i].set_title(key)
        axes[i].axis('off')
    
    # Add overall title
    fig.suptitle('CycleGAN Style Transfer Results', fontsize=16)
    
    # Add description
    plt.figtext(0.5, 0.01, 
               'Top row: A→B→A cycle (real_A, fake_B, rec_A)\nBottom row: B→A→B cycle (real_B, fake_A, rec_B)', 
               ha='center', fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def image_to_tensor(image_path, size=256):
    """
    Load an image and convert it to a tensor for inference.
    
    Parameters:
        image_path (str) -- path to the image file
        size (int) -- size to resize the image to
        
    Returns:
        torch.Tensor -- tensor of shape (1, C, H, W)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Apply transform and add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor


def save_tensor_as_image(tensor, save_path):
    """
    Save a tensor as an image file.
    
    Parameters:
        tensor (torch.Tensor) -- input tensor of shape (1, C, H, W) or (C, H, W)
        save_path (str) -- path to save the image
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to image
    image = tensor_to_image(tensor)
    
    # Convert to uint8 format
    image = (image * 255).astype(np.uint8)
    
    # Save the image
    Image.fromarray(image).save(save_path)


def create_comparison_grid(images_dict, title, save_path):
    """
    Create a visual comparison grid of images.
    
    Parameters:
        images_dict (dict) -- dictionary mapping labels to image tensors
        title (str) -- title for the grid
        save_path (str) -- path to save the grid image
    """
    n_images = len(images_dict)
    
    # Calculate grid size
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each image
    for i, (label, tensor) in enumerate(images_dict.items()):
        if i < len(axes):
            img = tensor_to_image(tensor)
            axes[i].imshow(img)
            axes[i].set_title(label)
            axes[i].axis('off')
    
    # Hide unused axes
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    # Add title
    fig.suptitle(title, fontsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()