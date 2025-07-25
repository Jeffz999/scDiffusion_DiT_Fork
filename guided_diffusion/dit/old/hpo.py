import os
import torch
import torch.nn as nn
from torch import optim, amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import optuna
from optuna.exceptions import TrialPruned
import numpy as np
import random
import argparse

# --- Project Imports ---
from diffusion import DiffusionGene
from transformer import DiT
from settings import args

use_amp = True
use_amp_scaler = True

# --- Configure Logging ---
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- DiT Model Configurations ---
# A smaller set of models to test individually
DIT_CONFIGS = {
    "scrdit_match": {"depth": 3, "hidden_size": 768, "patch_size": 10, "num_heads": 8},
    "S_4": {"depth": 8, "hidden_size": 384, "patch_size": 4, "num_heads": 6},
    "S_8": {"depth": 8, "hidden_size": 384, "patch_size": 8, "num_heads": 6},
    "SCan_5": {"depth": 8, "hidden_size": 768, "patch_size": 5, "num_heads": 8},
    "B_4_S": {"depth": 8, "hidden_size": 768, "patch_size": 4, "num_heads": 12},
    "B_4": {"depth": 12, "hidden_size": 768, "patch_size": 4, "num_heads": 12},
    "B_8": {"depth": 12, "hidden_size": 768, "patch_size": 8, "num_heads": 12},
}

# --- Fixed Parameters ---
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
GENE_SIZE = args.gene_size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_FREQUENCY = 200


def objective(trial: optuna.Trial, model_config_name: str) -> float:
    """
    The objective function for Optuna to optimize.
    A 'trial' represents a single run with a specific set of hyperparameters.
    The model configuration is now fixed for the entire study.
    """
    run_name = f"trial_{trial.number}"
    logging.info(f"--- Starting Trial {trial.number} for {model_config_name} ({run_name}) ---")

    # --- 1. Suggest Hyperparameters ---
    # Optuna will now only optimize the learning rate.
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    
    max_norm = 2.0
    
    model_params = DIT_CONFIGS[model_config_name]
    
    logging.info(f"  > Learning Rate: {lr:.2e}")
    logging.info(f"  > Model Config: {model_config_name} (depth={model_params['depth']}, hidden_size={model_params['hidden_size']}, patch_size={model_params['patch_size']})")
    
    assert GENE_SIZE % model_params['patch_size'] == 0, \
        f"Patch size {model_params['patch_size']} must be a divisor of gene size {GENE_SIZE}."

    # --- 2. Setup Model, Optimizer, and Dataloader ---
    model = DiT(
        input_size=GENE_SIZE,
        patch_size=model_params['patch_size'],
        hidden_size=model_params['hidden_size'],
        depth=model_params['depth'],
        num_heads=model_params['num_heads'],
    ).to(DEVICE)
    
    print(f"Num parameters in model: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    scaler = amp.GradScaler("cuda", enabled=use_amp_scaler)
    logging.info("Using Automatic Mixed Precision (AMP).")
    logging.info(f"Automatic Mixed Precision (AMP) Scaler {'enabled' if use_amp_scaler else 'disabled'}.")

    
    eta_min = lr * 1e-2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=eta_min)
    logging.info(f"  > Cosine Annealing LR from {lr:.2e} down to {eta_min:.2e}")
    
    mse = nn.MSELoss()
    
    dataloader = DataLoader(cell_dataloader.dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    diffusion = DiffusionGene(gene_size=GENE_SIZE, device=DEVICE)
    
    log_dir = os.path.join("runs", "hpo", model_config_name, run_name)
    writer = SummaryWriter(log_dir)
    
    # --- 3. Training Loop ---
    l = len(dataloader)
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        epoch_loss_sum = 0.0

        for i, genes in enumerate(pbar):
            genes = genes.to(DEVICE)
            t = diffusion.sample_timesteps(genes.shape[0])
            x_t, noise = diffusion.noise_genes(genes, t)
            
            with amp.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=use_amp):
                predicted_noise = model(x_t, t)
                loss: torch.Tensor = mse(noise, predicted_noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(MSE=loss.item())
            writer.add_scalar("Loss/Step_MSE", loss.item(), global_step=epoch * l + i)
            epoch_loss_sum += loss.item()

        scheduler.step()
        
        avg_epoch_loss = epoch_loss_sum / l
        writer.add_scalar("Loss/Epoch_Avg_MSE", avg_epoch_loss, global_step=epoch)
        writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], global_step=epoch)        
            
        # --- 4. Reporting and Pruning ---
        trial.report(avg_epoch_loss, epoch)
        if trial.should_prune():
            logging.info(f"Trial {trial.number} pruned at epoch {epoch} due to poor performance.")
            writer.close()
            raise TrialPruned()

        # --- 5. Checkpointing ---
        if (epoch + 1) % SAVE_FREQUENCY == 0 or epoch == EPOCHS - 1:
            ckpt_dir = os.path.join("ckpts", "hpo", model_config_name, run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch}.pt")
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }
            torch.save(checkpoint, ckpt_path)
            logging.info(f"Saved resumable checkpoint to {ckpt_path}")

    writer.close()
    logging.info(f"--- Trial {trial.number} Finished. Final Avg Loss: {avg_epoch_loss:.6f} ---")
    
    return avg_epoch_loss


if __name__ == '__main__':
    # --- NEW: Use argparse to select the model configuration ---
    parser = argparse.ArgumentParser(description="Run HPO to find the best learning rate for a specific DiT configuration.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=list(DIT_CONFIGS.keys()),
        help=f"The DiT configuration to optimize. Choose from: {list(DIT_CONFIGS.keys())}"
    )
    cli_args = parser.parse_args()
    selected_config = cli_args.config
    # --- END NEW ---

    logging.info(f"Starting Hyperparameter Optimization study for DiT config: {selected_config}")
    
    # --- NEW: Make study name specific to the selected configuration ---
    study_name = f"scRDiT-{selected_config}-LR-HPO"
    storage_name = f"sqlite:///{study_name}.db"
    # --- END NEW ---
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner
    )

    try:
        # --- NEW: Use a lambda function to pass the fixed config to the objective ---
        study.optimize(
            lambda trial: objective(trial, model_config_name=selected_config),
            n_trials=10,
            timeout=None
        )
        # --- END NEW ---
    except KeyboardInterrupt:
        logging.warning("Study interrupted by user. Results so far have been saved.")

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    logging.info("\n--- HPO Study Summary ---")
    logging.info(f"Study name: {study.study_name}")
    logging.info(f"Number of finished trials: {len(study.trials)}")
    logging.info(f"Number of pruned trials: {len(pruned_trials)}")
    logging.info(f"Number of complete trials: {len(complete_trials)}")

    if study.best_trial:
        logging.info("\n--- Best Trial ---")
        trial = study.best_trial
        logging.info(f"  Value (Min Loss): {trial.value:.6f}")
        logging.info("  Best Parameters:")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")
    else:
        logging.info("\nNo complete trials found in the study.")
