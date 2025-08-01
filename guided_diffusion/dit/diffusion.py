import torch
from tqdm import tqdm
import logging
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler
from .transformer import DiT
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class DiffusionGene:
    def __init__(
        self,
        gene_size=2000, 
        device="cuda", 
        num_channels=1,
        num_train_timesteps=1000,
        beta_scheduler: str = "linear",
        prediction_type: str = "epsilon",
        beta_start: float = 0.0001,
        beta_end: float = 0.02
        ):
        
        self.gene_size = gene_size
        self.device = device
        # --- NEW: Number of channels is now a parameter ---
        self.num_channels = num_channels

        # self.scheduler = PNDMScheduler(
        #     beta_start=0.0001,
        #     beta_end=0.02,
        #     beta_schedule="linear",
        #     num_train_timesteps=1000,
        #     prediction_type="epsilon",
        #     trained_betas=None,            
        # )
        
        self.scheduler = UniPCMultistepScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_scheduler,
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
            trained_betas=None,
            thresholding=True,
            sample_max_value=15,
            solver_order=2            
        )
        
        # self.scheduler = DDIMScheduler(
        #     beta_start=0.0001,
        #     beta_end=0.02,
        #     beta_schedule="linear",
        #     num_train_timesteps=1000,
        #     prediction_type="epsilon",
        #     trained_betas=None,
        #     clip_sample=False
        # ) 

    def noise_genes(self, x, t):
        """Add noise to the genes using the scheduler's method."""
        noise = torch.randn_like(x)
        noisy_x = self.scheduler.add_noise(x, noise, t)
        return noisy_x, noise

    def sample_timesteps(self, n):
        """Generate random timesteps for training."""
        return torch.randint(low=0, high=self.scheduler.config.num_train_timesteps, size=(n,), device=self.device)

    def sample(self, model: DiT, n: int, num_inference_steps: int = 25, cfg_scale = 1.0, y = None, clamp: bool = False, eta=1.0):
        """
        Modern sampling method using the diffusers scheduler, now for multi-channel data.
        """
        logging.info(f"Sampling {n} new genes ({self.num_channels} channels) with selected sampler...")
        model.eval()

        self.scheduler.set_timesteps(num_inference_steps)

        # 1. Start with random noise, with the correct number of channels
        x = torch.randn((n, self.num_channels, self.gene_size), device=self.device)

        x *= self.scheduler.init_noise_sigma

        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps, desc="Sampling"):
                timestep_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
                # Use CFG if scale is greater than 1.0
                if cfg_scale > 1.0:
                    predicted_noise = self.model.forward_with_cfg(x, timestep_batch, y=y, cfg_scale=cfg_scale)
                else:
                    predicted_noise = self.model(x, timestep_batch, y=y)
                #x = self.scheduler.step(predicted_noise, t, x, eta=eta).prev_sample
                x = self.scheduler.step(predicted_noise, t, x).prev_sample
        
        if clamp:
            logging.info("Clamping generated samples to [-1, 1] range.")
            x = torch.clamp(x, min=-1.0, max=1.0)

        model.train()
        return x.cpu()
    
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # adapted from https://github.com/huggingface/diffusers/blob/v0.34.0/src/diffusers/schedulers/scheduling_ddpm.py#L129
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        alphas_cumprod = self.scheduler.alphas_cumprod.to(sample.device)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


if __name__ == '__main__':
    # Test code.
    # --- MODIFIED: Test with 2 channels ---
    diffusion = DiffusionGene(num_channels=2)
    model = DiT(depth=3, patch_size=10, in_channels=2).to('cuda')

    logging.info("Testing corrected sample method for 2 channels...")
    X = diffusion.sample(model, n=4, num_inference_steps=10)
    X = X.to('cpu')
    print("Sampled shape:", X.shape) # Should be (4, 2, 2000)
    logging.info("Test complete.")