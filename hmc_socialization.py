import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
import json
import numpy as np
import random
from text_network import TextNetwork
from pymc_models import run_model
from utils import write1D, write2D, write3D
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="LDA-style topic model configuration")

    parser.add_argument("--input_dir",        type=str,   required=True,  help="Input directory path")
    parser.add_argument("--output_dir",       type=str,   required=True,  help="Output directory path")
    parser.add_argument("--checkpoint_dir",   type=str,   required=True,  help="Checkpoint directory path")
    parser.add_argument("--topics",           type=int,   required=True,  help="Number of topics")
    parser.add_argument("--samples",       type=int,   required=True,  help="Number of samples to collect")
    parser.add_argument("--warmup",     type=int,   required=True,  help="Number of warmup steps")
    parser.add_argument("--alpha_sum_vocab",  type=float, required=True,  help="Alpha sum over vocabulary")
    parser.add_argument("--alpha_sum_topics", type=float, required=True,  help="Alpha sum over topics")
    parser.add_argument("--alpha_edges",      type=float, required=True,  help="Alpha for edges")
    parser.add_argument("--model_name",      type=str, required=True,  help="Either 'gammas_pooled', 'gammas_unpooled', 'no_topics', or 'no_topics_multinomial'")
    parser.add_argument("--device",      type=str, required=True,  help="Either 'cpu' or 'cuda'")
    parser.add_argument("--num_chains",      type=int, required=True,  help="How many times to run the sampler")
    return parser.parse_args()
    
if __name__ == "__main__":
    opt = parse_args()
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    numpyro.set_platform(opt.device)
    text_network = TextNetwork.load(opt.input_dir)
    samples = run_model(text_network,
                        opt.topics,
                        opt.alpha_sum_topics,
                        opt.alpha_sum_vocab,
                        opt.alpha_edges,
                        opt.samples,
                        opt.warmup,
                        opt.num_chains,
                        opt.model_name,
                        opt.checkpoint_dir,
                        checkpoint_interval=100
              )
    with open(opt.output_dir) as f:
        f.write(json.dumps(samples))
