"""
Evaluation script for trained CycleGAN models.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import lpips
from pytorch_fid import fid_score
from models.cycle_gan import CycleGAN
from data_preparation import get_test_data_loaders
from utils import tensor_to_image, save_tensor_as_image, create_comparison_grid
from config import *


def compute_lpips(model, dataloader_A, dataloader_B, lpips_model, device):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) scores.
    
    Parameters:
        model (CycleGAN) -- trained CycleGAN model
        dataloader_A (DataLoader) -- test data from domain A
        dataloader_B (DataLoader) -- test data from domain B
        lpips_model -- LPIPS model for perceptual similarity
        device (torch.device) -- device to run evaluation on
        
    Returns:
        dict -- dictionary of average LPIPS scores
    """
    # Set model to evaluation mode
    model.eval()
    lpips_model.eval()
    
    # Initialize metrics
    lpips_scores = {
        'cycle_consistency_A': 0.0,  # real_A vs rec_A
        'cycle_consistency_B': 0.0,  # real_B vs rec_B
        'structural_A_to_B': 0.0,    # real_A vs fake_B
        'structural_B_to_A': 0.0,    # real_B vs fake_A
    }
    
    # Number of test samples
    n_samples_A = len(dataloader_A.dataset)
    n_samples_B = len(dataloader_B.dataset)
    
    print("Computing LPIPS scores...")
    
    # Evaluate cycle consistency and structural preservation
    with torch.no_grad():
        # Process domain A
        for real_A in tqdm(dataloader_A, desc="Domain A"):
            real_A = real_A.to(device)
            fake_B = model.netG_A(real_A)
            rec_A = model.netG_B(fake_B)
            
            # Compute cycle consistency LPIPS
            lpips_cycle_A = lpips_model(real_A, rec_A).mean()
            lpips_scores['cycle_consistency_A'] += lpips_cycle_A.item() * real_A.size(0)
            
            # Compute structural preservation LPIPS
            lpips_struct_A = lpips_model(real_A, fake_B).mean()
            lpips_scores['structural_A_to_B'] += lpips_struct_A.item() * real_A.size(0)
        
        # Process domain B
        for real_B in tqdm(dataloader_B, desc="Domain B"):
            real_B = real_B.to(device)
            fake_A = model.netG_B(real_B)
            rec_B = model.netG_A(fake_A)
            
            # Compute cycle consistency LPIPS
            lpips_cycle_B = lpips_model(real_B, rec_B).mean()
            lpips_scores['cycle_consistency_B'] += lpips_cycle_B.item() * real_B.size(0)
            
            # Compute structural preservation LPIPS
            lpips_struct_B = lpips_model(real_B, fake_A).mean()
            lpips_scores['structural_B_to_A'] += lpips_struct_B.item() * real_B.size(0)
    
    # Compute averages
    lpips_scores['cycle_consistency_A'] /= n_samples_A
    lpips_scores['structural_A_to_B'] /= n_samples_A
    lpips_scores['cycle_consistency_B'] /= n_samples_B
    lpips_scores['structural_B_to_A'] /= n_samples_B
    
    # Add average scores
    lpips_scores['avg_cycle_consistency'] = (
        lpips_scores['cycle_consistency_A'] + lpips_scores['cycle_consistency_B']
    ) / 2
    
    lpips_scores['avg_structural'] = (
        lpips_scores['structural_A_to_B'] + lpips_scores['structural_B_to_A']
    ) / 2
    
    return lpips_scores


def generate_test_samples(model, dataloader_A, dataloader_B, device, output_dir, num_samples=5):
    """
    Generate and save test samples.
    
    Parameters:
        model (CycleGAN) -- trained CycleGAN model
        dataloader_A (DataLoader) -- test data from domain A
        dataloader_B (DataLoader) -- test data from domain B
        device (torch.device) -- device to run evaluation on
        output_dir (str) -- directory to save samples
        num_samples (int) -- number of samples to generate
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get iterators
    iter_A = iter(dataloader_A)
    iter_B = iter(dataloader_B)
    
    # Generate samples
    with torch.no_grad():
        for i in range(min(num_samples, len(dataloader_A.dataset))):
            # Get a batch (single image) from each domain
            try:
                real_A = next(iter_A).to(device)
                real_B = next(iter_B).to(device)
            except StopIteration:
                break
            
            # Generate translations
            fake_B = model.netG_A(real_A)
            fake_A = model.netG_B(real_B)
            rec_A = model.netG_B(fake_B)
            rec_B = model.netG_A(fake_A)
            
            # Create a grid for this sample
            images_dict = {
                'Real A': real_A[0],
                'Fake B': fake_B[0],
                'Reconstructed A': rec_A[0],
                'Real B': real_B[0],
                'Fake A': fake_A[0],
                'Reconstructed B': rec_B[0],
            }
            
            # Save grid
            grid_path = os.path.join(output_dir, f'sample_{i+1}.png')
            create_comparison_grid(
                images_dict,
                f'CycleGAN Test Sample {i+1}',
                grid_path
            )
            
            # Also save individual images
            for name, img in images_dict.items():
                img_path = os.path.join(output_dir, f'sample_{i+1}_{name.replace(" ", "_")}.png')
                save_tensor_as_image(img, img_path)
            
            print(f"Saved sample {i+1} to {grid_path}")


def prepare_fid_data(model, dataloader_A, dataloader_B, device, output_dir):
    """
    Generate images for FID calculation.
    
    Parameters:
        model (CycleGAN) -- trained CycleGAN model
        dataloader_A (DataLoader) -- test data from domain A
        dataloader_B (DataLoader) -- test data from domain B
        device (torch.device) -- device to run evaluation on
        output_dir (str) -- directory to save generated images
    """
    # Create output directories
    real_A_dir = os.path.join(output_dir, 'real_A')
    fake_B_dir = os.path.join(output_dir, 'fake_B')
    real_B_dir = os.path.join(output_dir, 'real_B')
    fake_A_dir = os.path.join(output_dir, 'fake_A')
    
    os.makedirs(real_A_dir, exist_ok=True)
    os.makedirs(fake_B_dir, exist_ok=True)
    os.makedirs(real_B_dir, exist_ok=True)
    os.makedirs(fake_A_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate images for FID calculation
    with torch.no_grad():
        # Process domain A -> B
        for i, real_A in enumerate(tqdm(dataloader_A, desc="Domain A -> B")):
            real_A = real_A.to(device)
            fake_B = model.netG_A(real_A)
            
            # Save real A
            for j in range(real_A.size(0)):
                save_tensor_as_image(
                    real_A[j],
                    os.path.join(real_A_dir, f'img_{i*dataloader_A.batch_size+j:04d}.png')
                )
                
                # Save fake B
                save_tensor_as_image(
                    fake_B[j],
                    os.path.join(fake_B_dir, f'img_{i*dataloader_A.batch_size+j:04d}.png')
                )
        
        # Process domain B -> A
        for i, real_B in enumerate(tqdm(dataloader_B, desc="Domain B -> A")):
            real_B = real_B.to(device)
            fake_A = model.netG_B(real_B)
            
            # Save real B
            for j in range(real_B.size(0)):
                save_tensor_as_image(
                    real_B[j],
                    os.path.join(real_B_dir, f'img_{i*dataloader_B.batch_size+j:04d}.png')
                )
                
                # Save fake A
                save_tensor_as_image(
                    fake_A[j],
                    os.path.join(fake_A_dir, f'img_{i*dataloader_B.batch_size+j:04d}.png')
                )
    
    return {
        'real_A_dir': real_A_dir,
        'fake_B_dir': fake_B_dir,
        'real_B_dir': real_B_dir,
        'fake_A_dir': fake_A_dir,
    }


def compute_fid(dirs):
    """
    Compute FID (Fréchet Inception Distance) scores.
    
    Parameters:
        dirs (dict) -- directories containing real and fake images
        
    Returns:
        dict -- dictionary of FID scores
    """
    fid_scores = {}
    
    print("Computing FID scores...")
    
    # Compute FID between real_A and fake_B
    fid_A_to_B = fid_score.calculate_fid_given_paths(
        [dirs['real_B_dir'], dirs['fake_B_dir']],
        batch_size=50,
        device=DEVICE,
        dims=2048
    )
    fid_scores['fid_A_to_B'] = fid_A_to_B
    
    # Compute FID between real_B and fake_A
    fid_B_to_A = fid_score.calculate_fid_given_paths(
        [dirs['real_A_dir'], dirs['fake_A_dir']],
        batch_size=50,
        device=DEVICE,
        dims=2048
    )
    fid_scores['fid_B_to_A'] = fid_B_to_A
    
    # Average FID
    fid_scores['avg_fid'] = (fid_A_to_B + fid_B_to_A) / 2
    
    return fid_scores


def evaluate_model(model_path, epoch, output_dir='evaluation_results'):
    """
    Evaluate a trained CycleGAN model.
    
    Parameters:
        model_path (str) -- path to the saved model weights
        epoch (int) -- epoch of the saved model
        output_dir (str) -- directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    test_dataloader_A, test_dataloader_B = get_test_data_loaders(batch_size=1)
    
    # Initialize model
    model = CycleGAN(
        device=DEVICE,
        lambda_cycle=LAMBDA_CYCLE,
        lambda_identity=LAMBDA_IDENTITY,
        lambda_structure=LAMBDA_STRUCTURE
    )
    
    # Load model weights
    model.load_networks(epoch, model_path)
    
    print(f"Loaded model from epoch {epoch}")
    
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
    
    # Generate test samples
    samples_dir = os.path.join(output_dir, 'samples')
    print("Generating test samples...")
    generate_test_samples(
        model, 
        test_dataloader_A, 
        test_dataloader_B, 
        DEVICE, 
        samples_dir
    )
    
    # Compute LPIPS scores
    print("Computing LPIPS scores...")
    lpips_scores = compute_lpips(
        model, 
        test_dataloader_A, 
        test_dataloader_B, 
        lpips_model, 
        DEVICE
    )
    
    # Prepare data for FID calculation
    fid_data_dir = os.path.join(output_dir, 'fid_data')
    print("Preparing data for FID calculation...")
    fid_dirs = prepare_fid_data(
        model, 
        test_dataloader_A, 
        test_dataloader_B, 
        DEVICE, 
        fid_data_dir
    )
    
    # Compute FID scores
    print("Computing FID scores...")
    fid_scores = compute_fid(fid_dirs)
    
    # Print results
    print("\nEvaluation Results:")
    print("\nLPIPS Scores (lower is better for cycle consistency):")
    print(f"Cycle Consistency A: {lpips_scores['cycle_consistency_A']:.4f}")
    print(f"Cycle Consistency B: {lpips_scores['cycle_consistency_B']:.4f}")
    print(f"Average Cycle Consistency: {lpips_scores['avg_cycle_consistency']:.4f}")
    print(f"Structural A to B: {lpips_scores['structural_A_to_B']:.4f}")
    print(f"Structural B to A: {lpips_scores['structural_B_to_A']:.4f}")
    print(f"Average Structural: {lpips_scores['avg_structural']:.4f}")
    
    print("\nFID Scores (lower is better):")
    print(f"FID A to B: {fid_scores['fid_A_to_B']:.4f}")
    print(f"FID B to A: {fid_scores['fid_B_to_A']:.4f}")
    print(f"Average FID: {fid_scores['avg_fid']:.4f}")
    
    # Save results to file
    results_file = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("Evaluation Results:\n\n")
        f.write("LPIPS Scores (lower is better for cycle consistency):\n")
        f.write(f"Cycle Consistency A: {lpips_scores['cycle_consistency_A']:.4f}\n")
        f.write(f"Cycle Consistency B: {lpips_scores['cycle_consistency_B']:.4f}\n")
        f.write(f"Average Cycle Consistency: {lpips_scores['avg_cycle_consistency']:.4f}\n")
        f.write(f"Structural A to B: {lpips_scores['structural_A_to_B']:.4f}\n")
        f.write(f"Structural B to A: {lpips_scores['structural_B_to_A']:.4f}\n")
        f.write(f"Average Structural: {lpips_scores['avg_structural']:.4f}\n\n")
        
        f.write("FID Scores (lower is better):\n")
        f.write(f"FID A to B: {fid_scores['fid_A_to_B']:.4f}\n")
        f.write(f"FID B to A: {fid_scores['fid_B_to_A']:.4f}\n")
        f.write(f"Average FID: {fid_scores['avg_fid']:.4f}\n")
    
    print(f"Results saved to {results_file}")
    
    return {
        'lpips_scores': lpips_scores,
        'fid_scores': fid_scores,
        'samples_dir': samples_dir,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CycleGAN model')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='directory containing model checkpoints')
    parser.add_argument('--epoch', type=int, default=100,
                        help='epoch of the model to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint_dir, args.epoch, args.output_dir)