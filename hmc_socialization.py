import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
import json
import numpy as np
import random
from text_network import TextNetwork
from utils import write2D, write3D
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="LDA-style topic model configuration")

    parser.add_argument("--input_dir",        type=str,   required=True,  help="Input directory path")
    parser.add_argument("--output_dir",       type=str,   required=True,  help="Output directory path")
    parser.add_argument("--topics",           type=int,   required=True,  help="Number of topics")
    parser.add_argument("--iterations",       type=int,   required=True,  help="Number of iterations")
    parser.add_argument("--warmup_steps",     type=int,   required=True,  help="Number of warmup steps")
    parser.add_argument("--alpha_sum_vocab",  type=float, required=True,  help="Alpha sum over vocabulary")
    parser.add_argument("--alpha_sum_topics", type=float, required=True,  help="Alpha sum over topics")
    parser.add_argument("--alpha_edges",      type=float, required=True,  help="Alpha for edges")
    parser.add_argument("--model_name",      type=float, required=True,  help="Either 'gammas_pooled' or 'gammas_unpooled'")
    parser.add_argument("--device",      type=float, required=True,  help="Either 'cpu' or 'cuda'")
    return parser.parse_args()
    
if __name__ == "__main__":
    opt = parse_args()
    numpyro.set_device(opt.device)
    text_network = TextNetwork.load(opt.input_dir)
    samples = run_model(text_network, opt.topics, opt.alpha_sum_topics, opt.alpha_sum_vocab, opt.alpha_edges, opt.samples, opt.warmup, opt.model_name)
    write3D("{}/lambda.txt".format(opts.output_dir), samples["lambda"])
    write3D("{}/gamma.txt".format(opts.output_dir), samples["gamma"])        
    write3D("{}/phi.txt".format(opts.output_dir), samples["phi"])
    write3D("{}/theta.txt".format(opts.output_dir), samples["theta"])
    write3D("{}/psi.txt".format(opts.output_dir), samples["psi"])
