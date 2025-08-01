"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import logger
from guided_diffusion.cell_datasets_loader import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util2 import TrainLoop

import torch
import numpy as np
import random
from torch.utils.data import DataLoader

def main():
    """
    Main training function.
    """
    setup_seed(1234)
    args = create_argparser().parse_args()

    logger.configure(dir='../output/logs/' + args.model_name)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    logger.log(f"Num parameters in model: {sum(p.numel() for p in model.parameters())}")


    logger.log("creating data loader...")
    data: DataLoader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        vae_path=args.vae_path,
        train_vae=False,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval_epochs=args.save_interval_epochs,
        resume_checkpoint=args.resume_checkpoint,
        mixed_precision_type=args.mixed_precision_type,
        weight_decay=args.weight_decay,
        lr_anneal_epochs=args.lr_anneal_epochs,
        model_name=args.model_name,
        save_dir=args.save_dir,
        snr_gamma=args.snr_gamma,
    ).run_loop()


def create_argparser():
    """
    Creates the argument parser for the training script.
    """
    defaults = dict(
        data_dir="data/tabula_muris/all.h5ad",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_epochs=1200,
        batch_size=128,
        ema_rate=0.9999,
        log_interval=50,
        save_interval_epochs=200,
        resume_checkpoint="",
        mixed_precision_type="bf16",
        vae_path='output/AE_checkpoint/muris_AE/model_seed=0_step=199999.pt',
        model_name="muris_diffusion",
        save_dir='output/diffusion_checkpoint',
        snr_gamma=5.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def setup_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()

