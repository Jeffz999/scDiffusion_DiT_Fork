import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim, amp
from settings import args
from typing import List
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from diffusion import DiffusionGene
from copy import deepcopy
from collections import OrderedDict

# Run this file to train your model.
# Change training parameters in settings.py.
use_amp = True
use_amp_scaler = True
max_norm = 1.1

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def save_checkpoint(model, ema_model, optimizer, scaler, epoch_label, run_name):
    """
    Saves a checkpoint of the model and training state.

    Args:
        model (nn.Module): The main model.
        ema_model (nn.Module): The Exponential Moving Average model.
        optimizer (optim.Optimizer): The optimizer.
        scaler (amp.GradScaler): The gradient scaler for AMP.
        epoch_label (Union[int, str]): The label for the checkpoint file (e.g., 99 or "final").
        run_name (str): The name of the current run.
    """
    logging.info(f"Saving checkpoint for label: {epoch_label}...")
    ckpt_dir = os.path.join("ckpts", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict()
    }
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_epoch{epoch_label}.pt")
    torch.save(checkpoint, ckpt_path)
    logging.info(f"Checkpoint saved to {ckpt_path}")

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay=0.999):
    """
    Step the EMA model towards the current model.

    Args:
        ema_model (nn.Module): The Exponential Moving Average model.
        model (nn.Module): The current training model.
        decay (float): The decay rate for the EMA.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def train_ddpm(args):
    """
    Main training function with EMA implementation and weighted loss.

    Args:
        args: An object containing training settings and parameters.
    """
    logging.info("Setting up training environment...")
    run_name: str = args.run_name
    device: str = args.device
    model: nn.Module = args.model.to(device)
    
    # --- EMA Implementation: Create EMA model ---
    ema_model = deepcopy(model).to(device)
    requires_grad(ema_model, False)
    ema_model.eval()  # EMA model is always in evaluation mode
    logging.info("EMA model created.")

    dataloader: DataLoader = cell_dataloader
    lr = args.lr
    optimizer: optim.Optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # --- MODIFIED: Revert to standard MSE loss. ---
    # It will operate on both channels, which is what we want.
    mse: nn.MSELoss = nn.MSELoss()
    
    scaler = amp.GradScaler("cuda", enabled=use_amp_scaler)
    logging.info(f"Automatic Mixed Precision (AMP) {'enabled' if use_amp else 'disabled'}.")
    logging.info(f"Automatic Mixed Precision (AMP) Scaler {'enabled' if use_amp_scaler else 'disabled'}.")
    
    eta_min = lr * 1e-2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)
    
    # --- MODIFIED: Initialize diffusion with 2 channels ---
    diffusion = DiffusionGene(gene_size=args.gene_size, device=device, num_channels=2)

    # --- EMA Implementation: Modified Checkpoint Loading ---
    if args.ckpt:
        ckpt_path = os.path.join("ckpts", args.run_name, f"{run_name}_epoch{args.ckpt_epoch}.pt")
        logging.info(f"Loading checkpoint from {ckpt_path}...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            # Load EMA model state if it exists in the checkpoint
            if 'ema' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema'])
                logging.info("Loaded EMA model weights from checkpoint.")
            else:
                # If no EMA state, initialize it from the main model
                update_ema(ema_model, model, decay=0)
                requires_grad(ema_model, False)
                ema_model.eval()  # EMA model is always in evaluation mode
                logging.warning("EMA weights not found in checkpoint. Initializing from model weights.")

            # Load optimizer state if it exists
            if 'opt' in checkpoint:
                optimizer.load_state_dict(checkpoint['opt'])
                logging.info("Loaded optimizer state from checkpoint.")
            else:
                 logging.warning("Optimizer state not found in checkpoint.")

        except FileNotFoundError:
            logging.error(f"Checkpoint file not found at {ckpt_path}. Starting from scratch.")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. It might be an old format.")
            # Fallback for very old, non-dict checkpoints
            model.load_state_dict(torch.load(ckpt_path, map_location=device))


    logger = SummaryWriter(os.path.join("runs", args.run_name))
    logging.info(f"TensorBoard logs will be saved to: runs/{args.run_name}")
    l = len(dataloader)

    if hasattr(model, 'log_stats'):
        logging.info("Enabling detailed model statistics logging to TensorBoard.")
        model.log_stats = True
        model.writer = logger

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss_list: List[float] = []

        for i, genes in enumerate(pbar):
            genes = genes.to(device)
            t: torch.Tensor = diffusion.sample_timesteps(genes.shape[0])
            x_t, noise = diffusion.noise_genes(genes, t)
            
            # --- NEW: Set global step for detailed logging ---
            if hasattr(model, 'global_step'):
                model.global_step = epoch * l + i

            with amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                predicted_noise = model(x_t, t)
                # --- MODIFIED: Simple MSE loss on the 2-channel noise prediction ---
                loss: torch.Tensor = mse(noise, predicted_noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
            # --- Monitor gradient norm ---
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            scaler.step(optimizer)
            scaler.update()
            
            # --- EMA Implementation: Update EMA model after each step ---
            update_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item(), GradNorm=total_norm)
            logger.add_scalar("Loss/MSE", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("Training/Gradient_Norm", total_norm, global_step=epoch * l + i)
            epoch_loss_list.append(loss.item())

        scheduler.step()
        logger.add_scalar("Training/Learning_Rate", scheduler.get_last_lr()[0], global_step=epoch)
        
        avg_epoch_loss: float = sum(epoch_loss_list) / len(epoch_loss_list)
        logging.info(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.6f}")
        logger.add_scalar("Loss/Epoch_Avg_MSE", avg_epoch_loss, global_step=epoch)

        if (epoch + 1) % args.save_frequency == 0:
            save_checkpoint(model, ema_model, optimizer, scaler, epoch, run_name)
    
    logging.info("Training finished.")
    save_checkpoint(model, ema_model, optimizer, scaler, "final", run_name)
    logger.close()

if __name__ == '__main__':
    train_ddpm(args)
