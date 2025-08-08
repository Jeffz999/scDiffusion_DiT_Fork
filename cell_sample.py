import argparse
import os
import random
import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch

from guided_diffusion.dit.diffusion import DiffusionGene
from guided_diffusion.dit.transformer import DiT
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def setup_seed(seed: int):
    """
    Sets the random seed for reproducibility across torch, numpy, and random.

    :param seed: The integer value for the seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_data(all_cells: np.ndarray, data_dir: str):
    """
    Saves the generated cell data to a .npz file.

    :param all_cells: A numpy array of generated cell latent vectors.
    :param data_dir: The base path and filename for the output file.
    """
    output_path = f"{data_dir}.npz"
    
    # Ensure the output directory exists before saving
    output_directory = os.path.dirname(output_path)
    os.makedirs(output_directory, exist_ok=True)
    
    logging.info(f"Saving {len(all_cells)} generated cells to {output_path}")
    np.savez(output_path, cell_gen=all_cells)

def main():
    """
    Main function to run the sampling process using Classifier-Free Guidance.
    This version uses the EMA model for inference and standard Python logging.
    """
    args = create_argparser().parse_args()
    setup_seed(1234)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure logging to write to a file and to the console
    log_dir = os.path.join("output", "sampling_logs", os.path.basename(args.sample_dir).replace('.npz', ''))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "sampling.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Using device: {device}")

    logging.info("creating DiT model and diffusion...")
    model: DiT
    diffusion: DiffusionGene
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    logging.info(f"Loading EMA model from: {args.model_path}")
    # Load the EMA model state dict. For inference, we load the EMA weights 
    # into our main model object for better performance.
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    logging.info("sampling...")
    all_cells = []
    
    while len(all_cells) < args.num_samples:
        model_kwargs: Dict[str, Any] = {}
        
        # Prepare class labels for conditional generation
        if args.class_cond:
            logging.info(f"Generating with guidance for class {args.cell_type} and CFG scale {args.cfg_scale}")
            classes = torch.full((args.batch_size,), args.cell_type, device=device, dtype=torch.long)
            model_kwargs["y"] = classes
        else:
            logging.info("Generating unconditionally.")
            # For unconditional generation, y is None and cfg_scale is implicitly 1.0
            args.cfg_scale = 1.0

        # Use the standard sampling method from DiffusionGene, which internally handles CFG
        sample = diffusion.sample(
            model,
            n=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            y=model_kwargs.get("y"),
            cfg_scale=args.cfg_scale,
        )

        # Squeeze channel dimension: [Batch, Channels, Length] -> [Batch, Length]
        # This matches the expected output shape for evaluation.
        sample_np = sample.squeeze(1).cpu().numpy()
        all_cells.extend(sample_np)
        logging.info(f"created {len(all_cells)}/{args.num_samples} samples")

    # Ensure the final array has exactly num_samples
    arr = np.array(all_cells)[:args.num_samples]
    save_data(arr, args.sample_dir)

    logging.info("sampling complete")

def create_argparser():
    """
    Creates the argument parser for CFG-based sampling.
    """
    defaults = dict(
        clip_denoised=False,
        num_samples=3000,
        batch_size=1000,
        num_inference_steps=25,
        # Updated path to point to an EMA model checkpoint, which is standard for inference.
        model_path="output/diffusion_checkpoint/muris_diffusion/epoch_799/ema_model.pt",
        sample_dir="output/simulated_samples/muris_cfg_1",
        
        # --- Classifier-Free Guidance Arguments ---
        class_cond=False, # Set to True to enable conditional generation
        cell_type=0,      # The target class label for conditional generation
        cfg_scale=4.0,    # The scale for CFG. 1.0 means no guidance.
        
        # --- General Model Arguments ---
        num_classes=12,   # Total number of classes the model was trained on
    )
    defaults.update(model_and_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
