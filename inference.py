"""
Inference script for applying style transfer to new images.
"""
import os
import argparse
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from models.cycle_gan import CycleGAN
from utils import image_to_tensor, save_tensor_as_image, create_comparison_grid
from config import *


def load_model(checkpoint_dir, epoch, device):
    """
    Load a trained CycleGAN model.
    
    Parameters:
        checkpoint_dir (str) -- directory containing model checkpoints
        epoch (int) -- epoch of the model to load
        device (torch.device) -- device to load the model on
        
    Returns:
        CycleGAN -- loaded model
    """
    model = CycleGAN(
        device=device,
        lambda_cycle=LAMBDA_CYCLE,
        lambda_identity=LAMBDA_IDENTITY,
        lambda_structure=LAMBDA_STRUCTURE
    )
    
    model.load_networks(epoch, checkpoint_dir)
    model.eval()
    
    return model


def transfer_style(model, input_path, output_dir, direction='AtoB'):
    """
    Apply style transfer to an input image.
    
    Parameters:
        model (CycleGAN) -- trained CycleGAN model
        input_path (str) -- path to input image
        output_dir (str) -- directory to save output images
        direction (str) -- 'AtoB' for source to target style, 'BtoA' for target to source style
        
    Returns:
        dict -- paths to output images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess input image
    image_tensor = image_to_tensor(input_path, size=IMAGE_SIZE).to(DEVICE)
    
    # Get filename without extension for output naming
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Set model to evaluation mode
    model.eval()
    
    # Apply style transfer
    with torch.no_grad():
        if direction == 'AtoB':
            # Source style to target style (A → B)
            output_tensor = model.netG_A(image_tensor)
            # Cycle back to source style (A → B → A)
            cycle_tensor = model.netG_B(output_tensor)
        else:
            # Target style to source style (B → A)
            output_tensor = model.netG_B(image_tensor)
            # Cycle back to target style (B → A → B)
            cycle_tensor = model.netG_A(output_tensor)
    
    # Save output images
    original_path = os.path.join(output_dir, f"{input_filename}_original.png")
    output_path = os.path.join(output_dir, f"{input_filename}_{direction}.png")
    cycle_path = os.path.join(output_dir, f"{input_filename}_{direction}_cycle.png")
    
    # Save individual images
    save_tensor_as_image(image_tensor[0], original_path)
    save_tensor_as_image(output_tensor[0], output_path)
    save_tensor_as_image(cycle_tensor[0], cycle_path)
    
    # Create a comparison grid
    grid_path = os.path.join(output_dir, f"{input_filename}_{direction}_comparison.png")
    
    if direction == 'AtoB':
        grid_title = f"Style Transfer: {DEFAULT_SOURCE_STYLE} to {DEFAULT_TARGET_STYLE}"
        input_label = f"Original ({DEFAULT_SOURCE_STYLE})"
        output_label = f"Transferred ({DEFAULT_TARGET_STYLE})"
        cycle_label = f"Reconstructed ({DEFAULT_SOURCE_STYLE})"
    else:
        grid_title = f"Style Transfer: {DEFAULT_TARGET_STYLE} to {DEFAULT_SOURCE_STYLE}"
        input_label = f"Original ({DEFAULT_TARGET_STYLE})"
        output_label = f"Transferred ({DEFAULT_SOURCE_STYLE})"
        cycle_label = f"Reconstructed ({DEFAULT_TARGET_STYLE})"
    
    images_dict = {
        input_label: image_tensor[0],
        output_label: output_tensor[0],
        cycle_label: cycle_tensor[0]
    }
    
    create_comparison_grid(images_dict, grid_title, grid_path)
    
    return {
        'original': original_path,
        'output': output_path,
        'cycle': cycle_path,
        'grid': grid_path
    }


def process_directory(model, input_dir, output_dir, direction='AtoB'):
    """
    Process all images in a directory.
    
    Parameters:
        model (CycleGAN) -- trained CycleGAN model
        input_dir (str) -- directory containing input images
        output_dir (str) -- directory to save output images
        direction (str) -- 'AtoB' for source to target style, 'BtoA' for target to source style
        
    Returns:
        list -- paths to output comparison grids
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and 
        os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return []
    
    # Process each image
    output_grids = []
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        outputs = transfer_style(model, image_path, output_dir, direction)
        output_grids.append(outputs['grid'])
        print(f"Saved outputs to {output_dir}")
    
    return output_grids


def main():
    """Main function for inference."""
    parser = argparse.ArgumentParser(description='Apply style transfer to new images.')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='directory containing model checkpoints')
    parser.add_argument('--epoch', type=int, default=100,
                        help='epoch of the model to use')
    parser.add_argument('--input', type=str, required=True,
                        help='path to input image or directory of images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='directory to save output images')
    parser.add_argument('--direction', type=str, choices=['AtoB', 'BtoA'], default='AtoB',
                        help='direction of style transfer: AtoB (default) or BtoA')
    
    args = parser.parse_args()
    
    print(f"Loading model from epoch {args.epoch}...")
    model = load_model(args.checkpoint_dir, args.epoch, DEVICE)
    
    if os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        output_grids = process_directory(model, args.input, args.output_dir, args.direction)
        print(f"Processed {len(output_grids)} images. Results saved to {args.output_dir}")
    else:
        print(f"Processing single image: {args.input}")
        outputs = transfer_style(model, args.input, args.output_dir, args.direction)
        print(f"Results saved to {args.output_dir}")
        print(f"Comparison grid: {outputs['grid']}")


if __name__ == "__main__":
    main()