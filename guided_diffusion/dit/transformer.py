import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional

class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


def modulate(x, shift, scale):
    """
    Applies adaptive instance normalization.
    The shift operation is optional.
    """
    if shift is None:
        # If no shift is provided, create a zero tensor with the same shape as the scale tensor
        shift = torch.zeros_like(scale)
    # The modulation operation: scale the input x and then add the shift
    # scale is expanded to match the dimensions of x for element-wise multiplication
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(nn.Module):
    """
    Patch Embedding for 1D data.
    """
    def __init__(self, img_size=2000, patch_size=10, in_c=1, embed_dim=768, norm_layer=None, bias=False):
        super().__init__()
        # Ensure img_size and patch_size are treated as 1D
        self.img_size = (img_size,)
        self.patch_size = (patch_size,)
        self.grid_size = (img_size // patch_size,)
        self.num_patches = self.grid_size[0]
        # Use Conv1d for 1D data
        self.proj = nn.Conv1d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.img_size[0], f"Input length ({L}) doesn't match model ({self.img_size[0]})."
        x = self.proj(x)
        x = x.transpose(1, 2)  # [B, C, L] -> [B, L, C]
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """
    --- NEW: Attention module with per-head Query-Key Normalization ---
    This version correctly implements QK-Norm by normalizing each head's Q and K vectors
    independently, which is more stable and aligned with modern implementations like SD3.
    """
    def __init__(self, dim, num_heads, qkv_bias=True, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Linear layer to project input to Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Final projection layer
        self.proj = nn.Linear(dim, dim)

        # --- NEW: Per-head QK-Norm using RMSNorm ---
        # These normalization layers operate on the head_dim, not the full model dim.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)

    def forward(self, x, log_stats=False, writer=None, global_step=None, prefix=""):
        B, L, C = x.shape
        # 1. Project to Q, K, V
        qkv = self.qkv(x)

        # 2. Reshape for multi-head attention and split
        # (B, L, 3 * C) -> (B, L, 3, num_heads, head_dim) -> 3x (B, L, num_heads, head_dim)
        q, k, v = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).chunk(3, dim=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)

        # 3. --- NEW: Apply QK-Norm *per head* ---
        # The RMSNorm is applied to the last dimension, which is `head_dim`.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # --- NEW: Extensive monitoring ---
        if log_stats and writer:
            writer.add_scalar(f'{prefix}/q_norm_val', q.norm(dim=-1).mean(), global_step)
            writer.add_scalar(f'{prefix}/k_norm_val', k.norm(dim=-1).mean(), global_step)
            writer.add_scalar(f'{prefix}/v_val_norm', v.norm(dim=-1).mean(), global_step)

        # 4. Transpose for scaled_dot_product_attention's expected input
        # (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v)

        # 6. Reshape back to (B, L, C)
        x = attn_output.transpose(1, 2).reshape(B, L, C)

        # 7. Final projection
        x = self.proj(x)

        if log_stats and writer:
            writer.add_scalar(f'{prefix}/attn_output_norm', x.norm(dim=-1).mean(), global_step)

        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, multiple=256):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Use the new, corrected Attention class
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=multiple)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, log_stats=False, writer=None, global_step=None, prefix=""):
        # Generate modulation parameters from the conditioning signal
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # --- NEW: Extensive monitoring for modulation parameters ---
        if log_stats and writer:
            with torch.no_grad():
                writer.add_scalar(f'{prefix}/gate_msa_norm', gate_msa.norm(dim=-1).mean(), global_step)
                writer.add_scalar(f'{prefix}/gate_mlp_norm', gate_mlp.norm(dim=-1).mean(), global_step)
                writer.add_scalar(f'{prefix}/scale_msa_norm', scale_msa.norm(dim=-1).mean(), global_step)
                writer.add_scalar(f'{prefix}/scale_mlp_norm', scale_mlp.norm(dim=-1).mean(), global_step)
                writer.add_scalar(f'{prefix}/shift_msa_norm', shift_msa.norm(dim=-1).mean(), global_step)
                writer.add_scalar(f'{prefix}/shift_mlp_norm', shift_mlp.norm(dim=-1).mean(), global_step)

        # Attention block with modulation
        modulated_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.attn(modulated_input, log_stats, writer, global_step, prefix=f"{prefix}/Attention")
        x = x + gate_msa.unsqueeze(1) * attn_output

        # MLP block with modulation
        modulated_mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(modulated_mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=2000,
        patch_size=10,
        in_channels=1,
        hidden_size=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_pos_embs=True
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_pos_embs = use_pos_embs

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        # --- NEW: Attributes for logging ---
        self.log_stats = False
        self.writer = None
        self.global_step = 0

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if self.use_pos_embs:
            pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
           nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            # --- NEW: Implement adaLN-Gaussian initialization ---
            # Initialize the final layer of the modulation network to be close to zero,
            # but not exactly zero. This "soft" identity initialization can improve training
            # stability and performance, as suggested by recent research.
            nn.init.normal_(block.adaLN_modulation[-1].weight, mean=0, std=1e-5)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Final layer initialization: initialize modulation to be identity
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size * C)
        imgs: (N, C, L)
        """
        c = self.out_channels
        p = self.patch_size
        l_p = self.x_embedder.num_patches
        assert l_p == x.shape[1]

        x = x.reshape(shape=(x.shape[0], l_p, p, c))
        x = torch.einsum('nlpc->nclp', x)
        imgs = x.reshape(shape=(x.shape[0], c, l_p * p))
        return imgs

    def forward(self, x, t):
        """
        Forward pass of DiT.
        x: (N, C, L) tensor of 1D inputs
        t: (N,) tensor of diffusion timesteps
        """
        if self.use_pos_embs:
            x = self.x_embedder(x) + self.pos_embed
        else:
            x = self.x_embedder(x)
        c = self.t_embedder(t)

        # --- NEW: Log conditioning vector norm ---
        if self.log_stats and self.writer:
            with torch.no_grad():
                self.writer.add_scalar('Stats/Cond_Norm', c.norm(dim=-1).mean(), self.global_step)

        for i, block in enumerate(self.blocks):
            x = block(x, c, self.log_stats, self.writer, self.global_step, prefix=f"Stats/Block_{i}")

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
    
    #TODO: debug
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        #eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Create 1D sin/cos positional embeddings.
    """
    grid = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


if __name__ == '__main__':
    # Test code.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = torch.randint(0, 1000, (16,)).to(device)
    x = torch.randn((16, 1, 2000)).to(device)
    dit = DiT(depth=24, hidden_size=768*2, patch_size=4, num_heads=24).to(device)
    
    # --- Test new logging ---
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter("runs/test_run")
    dit.log_stats = True
    dit.writer = writer
    dit.global_step = 0
    # --- End test ---

    out = dit(x, t)
    print("Output shape:", out.shape)
    
    # Check parameter count
    num_params = sum(p.numel() for p in dit.parameters())
    print(f"Total parameters: {num_params:,}")
    
    writer.close()
    print("Test complete. Check the 'runs/test_run' directory for TensorBoard logs.")