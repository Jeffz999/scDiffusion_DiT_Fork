import argparse
import inspect
from typing import Tuple, Union

import torch.nn as nn
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .cell_model import Cell_classifier, Cell_Unet
from dit.transformer import DiT
from dit.diffusion import DiffusionGene

NUM_CLASSES = 11

# TODO: update defaults
def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        prediction_type="epsilon",
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        class_cond=False,
    )
    
    
IN_CHANNELS=1
USE_POS_EMBS=False
def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        input_dim = 128,
        hidden_dim = 768,
        patch_size = 5,
        IN_CHANNELS=IN_CHANNELS,
        num_heads = 12,
        use_pos_embs=False
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = dict(
        input_dim = 128,
        hidden_dim = [512,512,256,128],
        classifier_use_fp16=False,
        dropout = 0.1,
        num_class = 11,
    )
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    # DiT Model Arguments
    input_dim: int,
    hidden_dim: int,
    patch_size: int,
    depth: int,
    num_heads: int,
    learn_sigma: bool,
    
    # DiffusionGene/UniPC Scheduler Arguments
    diffusion_steps: int,
    noise_schedule: str = "linear",
    prediction_type: str = "epsilon",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> Tuple[nn.Module, DiffusionGene]:
    model = DiT(
        depth=depth,
        input_size=input_dim,
        patch_size=patch_size,
        in_channels=IN_CHANNELS,
        hidden_size=hidden_dim,
        num_heads=num_heads,
        learn_sigma=learn_sigma,
        use_pos_embs=False
    )
    diffusion = DiffusionGene(
        beta_scheduler=noise_schedule,
        num_train_timesteps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        prediction_type=prediction_type,
    )
    return model, diffusion


# def create_model(
#     input_dim,
#     hidden_dim,
#     dropout,
# ):

#     return Cell_Unet(
#         input_dim,
#         hidden_dim,
#         dropout=dropout
#     )


def create_classifier_and_diffusion(
    input_dim,
    hidden_dim,
    classifier_use_fp16,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    dropout,
    num_class,
    class_cond,
):
    classifier = create_classifier(
        input_dim,
        hidden_dim,
        dropout=dropout,
        num_class=num_class
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
    input_dim,
    hidden_dim,
    num_class = NUM_CLASSES,
    dropout = 0.1,
):

    return Cell_classifier(
        input_dim,
        hidden_dim,
        num_class,
        dropout,
    )

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
