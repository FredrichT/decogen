"""
Main entry point for the Bathroom Interior Design Style Transfer project.
"""
import os
import argparse
import sys
from pathlib import Path

from config import *


def print_header():
    """Print the project header."""
    print("\n" + "=" * 80)
    print(" " * 20 + "BATHROOM DESIGN STYLE TRANSFER")
    print(" " * 15 + "Based on Optimized CycleGAN (Fu, 2022)")
    print("=" * 80 + "\n")


def setup_environment():
    """Set up the environment including necessary directories."""
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    # Check for GPU/MPS
    if DEVICE.type == 'cuda':
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif DEVICE.type == 'mps':
        print(f"Using Apple MPS (Metal Performance Shaders)")
    else:
        print(f"Using CPU: This will be slow for training!")
    
    print(f"Device: {DEVICE}")


def train_command(args):
    """Run the training process."""
    # Import here to avoid loading all modules when not needed
    from train import main as train_main
    
    # Override config values if specified
    if args.batch_size:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
        print(f"Using batch size: {BATCH_SIZE}")
    
    if args.epochs:
        global NUM_EPOCHS
        NUM_EPOCHS = args.epochs
        print(f"Training for {NUM_EPOCHS} epochs")
    
    if args.source_style and args.target_style:
        global DEFAULT_SOURCE_STYLE, DEFAULT_TARGET_STYLE
        DEFAULT_SOURCE_STYLE = args.source_style
        DEFAULT_TARGET_STYLE = args.target_style
        print(f"Style transfer: {DEFAULT_SOURCE_STYLE} → {DEFAULT_TARGET_STYLE}")
    
    # Run training
    train_main()


def evaluate_command(args):
    """Run the evaluation process."""
    # Import here to avoid loading all modules when not needed
    from evaluate import evaluate_model
    
    evaluate_model(
        model_path=args.checkpoint_dir,
        epoch=args.epoch,
        output_dir=args.output_dir
    )


def inference_command(args):
    """Run the inference process on new images."""
    # Import here to avoid loading all modules when not needed
    from inference import main as inference_main
    
    # Simply forward to the inference main function
    sys.argv = [sys.argv[0]] + args.args
    inference_main()


def main():
    """Main entry point function."""
    print_header()
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description='Bathroom Interior Design Style Transfer using CycleGAN',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the CycleGAN model')
    train_parser.add_argument('--batch_size', type=int, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--source_style', type=str, choices=STYLES,
                             help='Source style for transfer')
    train_parser.add_argument('--target_style', type=str, choices=STYLES,
                             help='Target style for transfer')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                            help='Directory containing model checkpoints')
    eval_parser.add_argument('--epoch', type=int, default=100,
                           help='Epoch of the model to evaluate')
    eval_parser.add_argument('--output_dir', type=str, default='evaluation_results',
                           help='Directory to save evaluation results')
    
    # Inference command
    infer_parser = subparsers.add_parser('inference', 
                                        help='Apply style transfer to new bathroom images')
    infer_parser.add_argument('args', nargs='*', 
                             help='Arguments to pass to inference.py')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Run the appropriate command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'inference':
        inference_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()