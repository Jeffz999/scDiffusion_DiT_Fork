"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import random

from guided_diffusion import logger
from guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return
#TODO: add in own model/ema loading code
def main():
    setup_seed(1234)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='output/checkpoint/sample_logs')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    generated_samples = diffusion.sample(
        model,
        n=args.num_samples,
        num_inference_steps=args.num_inference_steps
    )
    
    # The output has a channel dimension, so we squeeze it out before saving.
    arr = generated_samples.squeeze(1).cpu().numpy()
    save_data(arr, None, args.sample_dir)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=12000,
        batch_size=3000,
        use_ddim=False,
        model_path="output/checkpoint/backbone/open_problem/model800000.pt",
        sample_dir="output/simulated_samples/open_problem"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True # 设置随机数种子


if __name__ == "__main__":
    main()
