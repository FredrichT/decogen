"""
Training script for CycleGAN with Weights and Biases integration.
Adapted for bathroom design style transfer.
"""
import os
import time
import numpy as np
import torch
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from models.cycle_gan import CycleGAN
from data_preparation import (
    create_dataset_splits, 
    get_data_loaders,
    visualize_dataset_samples
)
from utils import tensor_to_image, save_images
from config import *


def train(
    dataloader_A, 
    dataloader_B, 
    val_dataloader_A, 
    val_dataloader_B, 
    model, 
    num_epochs=NUM_EPOCHS, 
    save_dir=CHECKPOINT_DIR,
    sample_dir=SAMPLE_DIR,
    use_wandb=True
):
    """
    Train the CycleGAN model.
    
    Parameters:
        dataloader_A (DataLoader) -- training data from domain A
        dataloader_B (DataLoader) -- training data from domain B
        val_dataloader_A (DataLoader) -- validation data from domain A
        val_dataloader_B (DataLoader) -- validation data from domain B
        model (CycleGAN) -- CycleGAN model
        num_epochs (int) -- number of training epochs
        save_dir (str) -- directory to save models
        sample_dir (str) -- directory to save sample images
        use_wandb (bool) -- whether to use Weights and Biases for logging
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Get fixed test samples for visualization
    fixed_A = next(iter(val_dataloader_A))[:4].to(DEVICE)
    fixed_B = next(iter(val_dataloader_B))[:4].to(DEVICE)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        
        # Initialize metrics for this epoch
        epoch_losses = {
            'g_loss': 0.0,
            'gan_loss_A': 0.0,
            'gan_loss_B': 0.0,
            'cycle_loss_A': 0.0,
            'cycle_loss_B': 0.0,
            'identity_loss_A': 0.0,
            'identity_loss_B': 0.0,
            'struct_loss_A': 0.0,
            'struct_loss_B': 0.0,
            'd_loss': 0.0,
            'd_loss_A': 0.0,
            'd_loss_B': 0.0,
        }
        
        # Use the shorter of the two dataloaders for the iteration count
        n_iterations = min(len(dataloader_A), len(dataloader_B))
        
        # Create iterators for cycling
        iter_A = iter(dataloader_A)
        iter_B = iter(dataloader_B)
        
        # Progress bar for this epoch
        progress_bar = tqdm(range(n_iterations), desc=f"Epoch {epoch}/{num_epochs}")
        
        for i in progress_bar:
            # Get data from both domains
            try:
                real_A = next(iter_A).to(DEVICE)
            except StopIteration:
                iter_A = iter(dataloader_A)
                real_A = next(iter_A).to(DEVICE)
                
            try:
                real_B = next(iter_B).to(DEVICE)
            except StopIteration:
                iter_B = iter(dataloader_B)
                real_B = next(iter_B).to(DEVICE)
            
            # Forward pass and optimization
            losses = model.optimize_parameters(real_A, real_B)
            
            # Update metrics
            for k, v in losses.items():
                epoch_losses[k] += v
            
            # Update progress bar
            progress_bar.set_postfix({
                'g_loss': f"{losses['g_loss']:.4f}",
                'd_loss': f"{losses['d_loss']:.4f}"
            })
        
        # Calculate average losses for the epoch
        for k in epoch_losses:
            epoch_losses[k] /= n_iterations
        
        # Validation step
        if val_dataloader_A and val_dataloader_B:
            model.eval()
            val_losses = {
                'val_g_loss': 0.0,
                'val_d_loss': 0.0,
            }
            
            n_val_iterations = min(len(val_dataloader_A), len(val_dataloader_B))
            with torch.no_grad():
                for i in range(n_val_iterations):
                    try:
                        val_real_A = next(iter(val_dataloader_A)).to(DEVICE)
                        val_real_B = next(iter(val_dataloader_B)).to(DEVICE)
                        
                        # Get fake images
                        val_fake_B = model.netG_A(val_real_A)
                        val_fake_A = model.netG_B(val_real_B)
                        
                        # Calculate validation losses (simplified)
                        val_g_loss = model.criterionGAN(model.netD_B(val_fake_B), True).item() + \
                                    model.criterionGAN(model.netD_A(val_fake_A), True).item()
                        
                        val_d_loss = (model.criterionGAN(model.netD_A(val_real_A), True).item() + \
                                     model.criterionGAN(model.netD_A(val_fake_A.detach()), False).item() + \
                                     model.criterionGAN(model.netD_B(val_real_B), True).item() + \
                                     model.criterionGAN(model.netD_B(val_fake_B.detach()), False).item()) / 4.0
                        
                        val_losses['val_g_loss'] += val_g_loss / 2.0
                        val_losses['val_d_loss'] += val_d_loss
                    
                    except StopIteration:
                        break
            
            # Average validation losses
            if n_val_iterations > 0:
                for k in val_losses:
                    val_losses[k] /= n_val_iterations
                
                # Add to metrics
                epoch_losses.update(val_losses)
        
        # Update learning rates
        model.update_learning_rate()
        
        # Print epoch stats
        time_per_epoch = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {time_per_epoch:.2f}s")
        print(f"G Loss: {epoch_losses['g_loss']:.4f}, D Loss: {epoch_losses['d_loss']:.4f}")
        if 'val_g_loss' in epoch_losses:
            print(f"Val G Loss: {epoch_losses['val_g_loss']:.4f}, Val D Loss: {epoch_losses['val_d_loss']:.4f}")
        
        # Save model checkpoints
        if epoch % SAVE_MODEL_EVERY == 0 or epoch == num_epochs:
            model.save_networks(epoch, save_dir)
            print(f"Saved model checkpoint at epoch {epoch}")
        
        # Generate and save sample images
        if epoch % SAVE_IMAGE_EVERY == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                samples = model.forward(fixed_A, fixed_B)
                
                # Save samples to disk
                sample_path = os.path.join(sample_dir, f'epoch_{epoch}.png')
                save_images(samples, sample_path)
                print(f"Saved sample images at epoch {epoch} to {sample_path}")
        
        # Log to wandb if enabled
        if use_wandb:
            # Create images for wandb
            model.eval()
            with torch.no_grad():
                samples = model.forward(fixed_A, fixed_B)
                
                # Convert to numpy for wandb logging
                wandb_images = {
                    'real_A': wandb.Image(tensor_to_image(samples['real_A'][0])),
                    'fake_B': wandb.Image(tensor_to_image(samples['fake_B'][0])),
                    'rec_A': wandb.Image(tensor_to_image(samples['rec_A'][0])),
                    'real_B': wandb.Image(tensor_to_image(samples['real_B'][0])),
                    'fake_A': wandb.Image(tensor_to_image(samples['fake_A'][0])),
                    'rec_B': wandb.Image(tensor_to_image(samples['rec_B'][0])),
                }
                
                # Log metrics and images
                wandb.log({
                    'epoch': epoch,
                    'learning_rate': model.optimizer_G.param_groups[0]['lr'],
                    **epoch_losses,
                    **wandb_images
                })
    
    # Final logs
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Models saved to {save_dir}")
    print(f"Sample images saved to {sample_dir}")


def main():
    """
    Main entry point for training.
    """
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    # Initialize wandb
    if WANDB_ENTITY:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    else:
        wandb.init(project=WANDB_PROJECT)
    
    # Log config to wandb
    wandb.config.update({
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'image_size': IMAGE_SIZE,
        'lambda_cycle': LAMBDA_CYCLE,
        'lambda_identity': LAMBDA_IDENTITY,
        'lambda_structure': LAMBDA_STRUCTURE,
        'device': str(DEVICE),
        'source_style': DEFAULT_SOURCE_STYLE,
        'target_style': DEFAULT_TARGET_STYLE,
    })
    
    # Print system info
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    elif DEVICE.type == 'mps':
        print("Using Apple Metal Performance Shaders (MPS) for acceleration")
    
    # Prepare dataset
    print(f"Preparing CycleGAN data for bathroom style transfer...")
    print(f"Source style: {DEFAULT_SOURCE_STYLE}, Target style: {DEFAULT_TARGET_STYLE}")
    stats = create_dataset_splits(
        dataset_path=DATASET_PATH,
        source_style=DEFAULT_SOURCE_STYLE,
        target_style=DEFAULT_TARGET_STYLE
    )
    
    # Visualize dataset samples
    sample_img_path = visualize_dataset_samples()
    if sample_img_path:
        print(f"Sample visualization saved to: {sample_img_path}")
    else:
        print("Warning: Could not create sample visualization!")
    
    # Log dataset stats to wandb
    wandb.log({
        'trainA_size': stats['trainA_size'],
        'trainB_size': stats['trainB_size'],
        'valA_size': stats['valA_size'],
        'valB_size': stats['valB_size'],
        'testA_size': stats['testA_size'],
        'testB_size': stats['testB_size'],
        'dataset_samples': wandb.Image(sample_img_path) if sample_img_path else None
    })
    
    # Get data loaders
    print("Creating data loaders...")
    dataloader_A, dataloader_B, val_dataloader_A, val_dataloader_B = get_data_loaders(
        batch_size=BATCH_SIZE, 
        image_size=IMAGE_SIZE
    )
    
    # Initialize CycleGAN model
    print("Initializing CycleGAN model...")
    model = CycleGAN(
        device=DEVICE,
        lambda_cycle=LAMBDA_CYCLE,
        lambda_identity=LAMBDA_IDENTITY,
        lambda_structure=LAMBDA_STRUCTURE
    )
    
    # Train the model
    print("Starting training...")
    train(
        dataloader_A, 
        dataloader_B,
        val_dataloader_A,
        val_dataloader_B,
        model, 
        num_epochs=NUM_EPOCHS, 
        save_dir=CHECKPOINT_DIR,
        sample_dir=SAMPLE_DIR,
        use_wandb=True
    )


if __name__ == "__main__":
    main()