import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from collections import OrderedDict
from copy import deepcopy
import time

from accelerate import Accelerator
from accelerate.utils import set_seed

from . import logger

from diffusers.training_utils import compute_snr

from .dit.diffusion import DiffusionGene
from .dit.transformer import DiT

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

SCHEDULER_MIN_MULTIPLIER = 1e-2
#TODO: https://github.com/huggingface/diffusers/blob/aa73072f1f7014635e3de916cbcf47858f4c37a0/examples/text_to_image/train_text_to_image.py
#prob have to manual cast with acceleate
class TrainLoop:
    def __init__(
        self,
        *,
        model: DiT,
        diffusion: DiffusionGene,
        data: DataLoader,
        batch_size,
        lr,
        ema_rate=0.9999,
        log_interval,
        save_interval,
        resume_checkpoint,
        mixed_precision_type:str ="bf16",
        weight_decay=0.0,
        lr_anneal_epochs=500,
        model_name,
        snr_gamma=5.0,
        save_dir,
    ):
        # Initialize Accelerator
        self.accelerator = Accelerator(mixed_precision=mixed_precision_type)
        self.device = self.accelerator.device

        self.model: DiT = model
        self.diffusion = diffusion
        self.data: DataLoader = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = ema_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = diffusion.scheduler
        self.weight_decay = weight_decay
        self.lr_anneal_epochs = lr_anneal_epochs
        self.model_name = model_name
        self.save_dir = save_dir
        self.snr_gamma = snr_gamma
        self.step = 0
        
        # Setup optimizer and scheduler
        self.opt: torch.optim.Optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Using CosineAnnealingLR from dit/train.py
        self.scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.lr_anneal_epochs, eta_min=self.lr * SCHEDULER_MIN_MULTIPLIER
        )

        # Prepare everything with Accelerate
        self.model, self.opt, self.data, self.scheduler = self.accelerator.prepare(
            self.model, self.opt, self.data, self.scheduler
        )

        # Create and prepare EMA model
        self.ema_model = deepcopy(self.model).to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.ema_model.eval()

        self.timestamp = model_name
        self.checkpoint_dir = os.path.join(self.save_dir, self.timestamp)

        if self.resume_checkpoint:
            logger.log(f"Resuming from checkpoint: {self.resume_checkpoint}")
            self.accelerator.load_state(self.resume_checkpoint)
            # Manually load EMA model state
            ema_path = os.path.join(self.resume_checkpoint, "ema_model.pt")
            if os.path.exists(ema_path):
                self.ema_model.load_state_dict(
                    torch.load(ema_path, map_location=self.device)
                )
                logger.log(f"Loaded EMA model from {ema_path}")
            # The step will be loaded by accelerator, but we need to track it
            # This part might need adjustment based on how you track epochs/steps
            # For simplicity, we'll assume the scheduler's last_epoch gives us a hint
            self.start_epoch = self.scheduler.last_epoch
        else:
            self.start_epoch = 0

    def run_loop(self):
        logger.log("Starting training loop...")
        for epoch in range(self.start_epoch, self.lr_anneal_epochs):
            logger.log(f"Starting epoch {epoch}:")
            for i, (batch, cond) in enumerate(self.data):   
                self.run_step(batch, cond) # ex shape batch:[bs: 128, latent:128] cond:[128]
                if self.step % self.log_interval == 0:
                    self.log_step()
                
                self.step += 1

                self.scheduler.step()
            
                if self.accelerator.is_main_process and (epoch + 1) % self.save_interval == 0:
                    self.save(epoch)

        logger.log("Training finished.")
        if self.accelerator.is_main_process:
            self.save("final")

    def run_step(self, batch, cond=None):
        self.model.train()
        self.opt.zero_grad()
        

        # Determine the target dtype from the accelerator's configuration
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        # Cast the input batch to the correct dtype for mixed precision
        batch = batch.to(self.accelerator.device, dtype=weight_dtype)
        # Add a channel dimension to the batch for model compat reasons
        batch = batch.unsqueeze(1)
        
        # Using the simplified loss calculation from dit/train.py
        t = self.diffusion.sample_timesteps(batch.shape[0])

        x_t, noise = self.diffusion.noise_genes(batch, t)
        
        scheduler_pred_type = self.schedule_sampler.config.prediction_type
        # ------------------ V-Prediction Target Calculation ------------------ #
        # Calculate the appropriate target for the loss function based on the prediction type.
        if scheduler_pred_type == "epsilon":
            target = noise
        elif scheduler_pred_type == "v_prediction":
            target = self.diffusion.get_velocity(batch, noise, t)
        else:
            raise ValueError(f"Unknown prediction type {scheduler_pred_type}")
        
        y = cond.get("y")
        if y is not None:
            y = y.to(self.accelerator.device)
        
        if cond is None:
            predicted_noise = self.model(x_t, t)
        else:
            predicted_noise = self.model(x_t, t, y) 

        # code from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
        if self.snr_gamma is None:
            # Use standard MSE loss if snr_gamma is not provided
            loss = F.mse_loss(target, predicted_noise)
        else:
            # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.schedule_sampler, t)
            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(t)], dim=1).min(
                dim=1
            )[0]
            if scheduler_pred_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif scheduler_pred_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(predicted_noise, target, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        self.accelerator.backward(loss)
        self.opt.step()

        # Update EMA model
        update_ema(self.ema_model, self.accelerator.unwrap_model(self.model), decay=self.ema_rate)

        # Log loss
        self.accelerator.log({"loss": loss.item()}, step=self.step)


    def log_step(self):
        # Logging is handled by accelerator.log, but we can add more here if needed
        if self.accelerator.is_main_process:
            logger.logkv("step", self.step)
            logger.logkv("lr", self.scheduler.get_last_lr()[0])
            logger.dumpkvs()

    def save(self, epoch_label):
        """
        Saves a checkpoint using Accelerate and also saves the EMA model.
        """
        if self.accelerator.is_main_process:
            save_dir = os.path.join(self.checkpoint_dir, f"epoch_{epoch_label}")
            os.makedirs(save_dir, exist_ok=True)
            
            logger.log(f"Saving checkpoint for epoch: {epoch_label} to {save_dir}...")
            
            # Use accelerator to save the main model, optimizer, and scheduler
            self.accelerator.save_state(save_dir)
            
            # Manually save the EMA model's state dictionary
            ema_model_path = os.path.join(save_dir, "ema_model.pt")
            unwrapped_ema_model = self.ema_model
            torch.save(unwrapped_ema_model.state_dict(), ema_model_path)
            
            logger.log(f"Checkpoint saved to {save_dir}")

