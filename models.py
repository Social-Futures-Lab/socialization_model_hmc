import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
import json
import numpy as np
import random
from numpyro.infer import MCMC, NUTS
import time
import numpyro.diagnostics as diagnostics
import json
import pickle
import os
from utils import read2D, read1D, flatten_docs, flatten_docs_ragged, group_by_edge_count, has_edges_bool_mask, build_edge_mask, calc_vocab_size

"""## Dirichlet over edges per tgt subreddit ##"""
#TO-DO: add typing
def model_gammas_pooled(num_topics, vocab_size, num_src_subs, num_tgt_docs, num_tgt_subs, num_src_words,
                           num_tgt_words, src_word_ids, tgt_word_ids, src_doc_ids, tgt_doc_ids, tgt_sub_ids,
                           adjacency_matrix, has_edges_mask, alpha_topics, alpha_vocab, alpha_edges, lambda_theta, lambda_psi):
    """
    Marginalized LDA (integrating out topic assignments z)
    """
    # topic distribution per src document
    with numpyro.plate("src_topics", num_src_subs):
        theta = numpyro.sample(
            "theta",
            dist.Dirichlet(alpha_topics)
        )  # shape: (num_src_subs, num_topics)

    # topic distribution per tgt document
    with numpyro.plate("tgt_topics", num_tgt_subs):
        psi = numpyro.sample(
            "psi",
            dist.Dirichlet(alpha_topics)
        )  # shape: (num_tgt_subs, num_topics)

    # citation distribution per tgt subreddit
    # documents are grouped by num edges and priors are generated in batches
    # then filled in tgt_docs x tgt_subs sized matrix
    with numpyro.plate("tgt_edges", num_tgt_subs):
        gamma = numpyro.sample(
            "gamma",
            dist.Dirichlet(alpha_edges*np.ones(num_src_subs))
        )  # shape: (num_tgt_subs, num_src_subs)
    #Only users w/ edges will have a prior here
    full_edge_mat = gamma[tgt_sub_ids]
    full_edge_mat = full_edge_mat * adjacency_matrix
    #need to normalize rows, but avoid divide by zero issues
    row_sums = full_edge_mat.sum(axis=1, keepdims=True)
    safe_row_sums = jnp.where(row_sums == 0, 1.0, row_sums)
    full_edge_mat = full_edge_mat / safe_row_sums
    full_edge_mat = jnp.where(
        has_edges_mask[:, None],
        full_edge_mat,
        0.0
    )
    # citation probability per tgt document
    with numpyro.plate("coin_flips", num_tgt_subs):
        lambdA = numpyro.sample(
            "lambda",
            dist.Beta(lambda_psi, lambda_theta)
        )  # shape: (num_tgt_subs, 1)

    # word distribution per topic
    with numpyro.plate("word_dists", num_topics):
        phi = numpyro.sample(
            "phi",
            dist.Dirichlet(alpha_vocab)
        )  # shape: (K, V)

    # likelihood (src)
    with numpyro.plate("src_tokens", num_src_words):
        # theta[doc[n]] -> (K,)
        log_theta = jnp.log(theta[src_doc_ids])           # (N, K)
        log_phi   = jnp.log(phi[:, src_word_ids]).T          # (N, K)
        src_doc_factor = log_theta + log_phi               # (N, K)
        numpyro.factor(
            "src_likelihood",
            logsumexp(src_doc_factor, axis=1)
        )
    #likelihood (tgt)
    with numpyro.plate("tgt_tokens", num_tgt_words):
        # theta[doc[n]] -> (K,)
        log_phi2   = jnp.log(phi[:, tgt_word_ids]).T          # (N, K)
        lambda_doc = lambdA[tgt_sub_ids]
        lambda_doc = jnp.where(has_edges_mask,
                               lambda_doc,
                               1.0
                              ) # for users w/ no edges, there's no chance of citation
        weighted_psi = jnp.expand_dims(lambda_doc, axis=-1)*psi[tgt_sub_ids, :]
        weighted_theta = jnp.expand_dims(1-lambda_doc, axis=-1)*(full_edge_mat@theta)
        log_topics = jnp.log((weighted_psi + weighted_theta)[tgt_doc_ids])
        tgt_doc_factor = log_topics + log_phi2               # (N, K)
        numpyro.factor(
            "tgt_likelihood",
            logsumexp(tgt_doc_factor, axis=1)
        )

"""## Base model, dirichlet over edges per user ##"""

#TO-DO: add typing
def model_gammas_unpooled(num_topics, vocab_size, num_src_subs, num_tgt_docs, num_tgt_subs, num_src_words,
                           num_tgt_words, src_word_ids, tgt_word_ids, src_doc_ids, tgt_doc_ids, tgt_sub_ids,
                           doc_edge_lists, has_edges_mask, alpha_topics, alpha_vocab, alpha_edges, lambda_theta, lambda_psi):
    """
    Marginalized LDA (integrating out topic assignments z)
    """
    # topic distribution per src document
    with numpyro.plate("src_topics", num_src_subs):
        theta = numpyro.sample(
            "theta",
            dist.Dirichlet(alpha_topics)
        )  # shape: (num_src_subs, num_topics)

    # topic distribution per tgt document
    with numpyro.plate("tgt_topics", num_tgt_subs):
        psi = numpyro.sample(
            "psi",
            dist.Dirichlet(alpha_topics)
        )  # shape: (num_tgt_subs, num_topics)

    # citation distribution per tgt document
    # documents are grouped by num edges and priors are generated in batches
    # then filled in tgt_docs x tgt_subs sized matrix
    full_edge_mat = jnp.zeros((num_tgt_docs, num_src_subs))
    for edge_obj in doc_edge_lists:
        num_edges= edge_obj["num_edges"]
        edges= edge_obj["edges"]
        doc_ids = edge_obj["doc_ids"]
        num_edge_docs = doc_ids.shape[0]
        with numpyro.plate("tgt_edges_{}".format(num_edges), num_edge_docs):
            gamma = numpyro.sample(
                "gamma_{}".format(num_edges),
                dist.Dirichlet(alpha_edges*np.ones(num_edges))
            )  # shape: (num_tgt_docs, num_src_subs)
            rows = doc_ids[:, None]  #Only users w/ edges will have a prior here
            full_edge_mat = full_edge_mat.at[rows, edges].set(gamma)

    # citation probability per tgt document
    with numpyro.plate("coin_flips", num_tgt_subs):
        lambdA = numpyro.sample(
            "lambda",
            dist.Beta(lambda_theta, lambda_psi)
        )  # shape: (num_tgt_subs, 1)

    # word distribution per topic
    with numpyro.plate("word_dists", num_topics):
        phi = numpyro.sample(
            "phi",
            dist.Dirichlet(alpha_vocab)
        )  # shape: (K, V)

    # likelihood (src)
    with numpyro.plate("src_tokens", num_src_words):
        # theta[doc[n]] -> (K,)
        log_theta = jnp.log(theta[src_doc_ids])           # (N, K)
        log_phi   = jnp.log(phi[:, src_word_ids]).T          # (N, K)
        src_doc_factor = log_theta + log_phi               # (N, K)
        numpyro.factor(
            "src_likelihood",
            logsumexp(src_doc_factor, axis=1)
        )
    #likelihood (tgt)
    with numpyro.plate("tgt_tokens", num_tgt_words):
        # theta[doc[n]] -> (K,)
        log_phi2   = jnp.log(phi[:, tgt_word_ids]).T          # (N, K)
        lambda_doc = lambdA[tgt_sub_ids]
        lambda_doc = jnp.where(has_edges_mask,
                               lambda_doc,
                               1.0
                              ) # for users w/ no edges, there's no chance of citation
        weighted_psi = jnp.expand_dims(lambda_doc, axis=-1)*psi[tgt_sub_ids, :]
        weighted_theta = jnp.expand_dims(1-lambda_doc, axis=-1)*(full_edge_mat@theta)
        log_topics = jnp.log((weighted_psi + weighted_theta)[tgt_doc_ids])
        tgt_doc_factor = log_topics + log_phi2               # (N, K)
        numpyro.factor(
            "tgt_likelihood",
            logsumexp(tgt_doc_factor, axis=1)
        )

MODEL_MAP = {
    "gammas_pooled": model_gammas_pooled,
    "gammas_unpooled": model_gammas_unpooled
}

def gen_model_args(text_network, topics, alpha_sum_topics, alpha_sum_vocab, alpha_edges, samples, warmup, model_name):
    vocab_size = calc_vocab_size(text_network)
    src_word_ids, src_doc_ids = flatten_docs_ragged(text_network.src_blobs)
    tgt_word_ids, tgt_doc_ids = flatten_docs_ragged(text_network.tgt_blobs)
    model_args = {
        "num_topics": topics,
        "vocab_size": vocab_size,
        "num_src_subs": len(text_network.src_blobs),
        "num_tgt_docs": len(text_network.tgt_blobs),
        "num_src_words": sum([len(src_blob) for src_blob in text_network.src_blobs]),
        "num_tgt_words": sum([len(src_blob) for src_blob in text_network.tgt_blobs]),
        "src_word_ids": src_word_ids,
        "tgt_word_ids": tgt_word_ids,
        "src_doc_ids": src_doc_ids,
        "tgt_doc_ids": tgt_doc_ids,
        "has_edges_mask": has_edges_bool_mask(text_network.edges),
        "num_tgt_subs": text_network.num_tgt_subreddits,
        "tgt_sub_ids": np.array(text_network.subreddits),
        "alpha_topics": np.ones(topics) * (alpha_sum_topics/topics),
        "alpha_vocab": np.ones(vocab_size)* (alpha_sum_topics/vocab_size),
        "alpha_edges": alpha_edges,
        "lambda_theta": 1,
        "lambda_psi": 1
    }
    if model_name == "gammas_pooled":
        #List of dicts where the items are ("num_edges" -> int, "doc_ids" ->list, "edges" -> 2d mat)
        model_args["adjacency_matrix"] = build_edge_mask(text_network.edges,
                                                         len(text_network.src_blobs))
    elif model_name == "gammas_unpooled":
        model_args["doc_edge_lists"] = group_by_edge_count(text_network.edges)
        
    return model_args

def run_model(text_network, topics, alpha_sum_topics, alpha_sum_vocab, alpha_edges, samples, warmup, model_name, checkpoint_dir, checkpoint_interval=10):
    if model_name not in MODEL_MAP:
        raise ValueError("Unsupported model type")
    
    model = MODEL_MAP[model_name]
    nuts_kernel = NUTS(model)
    rng_key = jr.PRNGKey(96)
    model_args = gen_model_args(text_network, topics, alpha_sum_topics, alpha_sum_vocab, alpha_edges, samples, warmup, model_name)
    
    checkpoint_path = os.path.join(checkpoint_dir, "mcmc_checkpoint.pkl")
    
    # check if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        samples_so_far = checkpoint["samples"]
        last_state = checkpoint["last_state"]
        samples_remaining = samples - checkpoint["num_samples_collected"]
        print(f"Resuming with {checkpoint['num_samples_collected']} samples already collected, {samples_remaining} remaining")
        
        if samples_remaining <= 0:
            print("Already have enough samples, returning saved results")
            return samples_so_far
        
        # resume from last state, no warmup needed
        mcmc = MCMC(nuts_kernel, num_warmup=0, num_samples=samples_remaining)
        mcmc.post_warmup_state = last_state
    else:
        print("Starting fresh run...")
        samples_so_far = None
        mcmc = MCMC(nuts_kernel, num_warmup=warmup, num_samples=checkpoint_interval)
        mcmc.run(rng_key, **model_args)
        
        # save initial checkpoint
        samples_so_far = mcmc.get_samples()
        last_state = mcmc.last_state
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "samples": samples_so_far,
                "last_state": mcmc.last_state,
                "num_samples_collected": checkpoint_interval
            }, f)
        print(f"Saved checkpoint with {checkpoint_interval} samples")
        samples_remaining = samples - checkpoint_interval

    # continue sampling in chunks
    num_collected = checkpoint["num_samples_collected"] if os.path.exists(checkpoint_path) and samples_so_far is not None else checkpoint_interval
    
    while samples_remaining > 0:
        chunk = min(checkpoint_interval, samples_remaining)
        mcmc = MCMC(nuts_kernel, num_warmup=0, num_samples=chunk)
        mcmc.post_warmup_state = last_state
        mcmc.run(last_state.rng_key, **model_args)
        
        new_samples = mcmc.get_samples()
        last_state = mcmc.last_state
        num_collected += chunk
        samples_remaining -= chunk
        
        # merge samples
        samples_so_far = {k: jnp.concatenate([samples_so_far[k], new_samples[k]], axis=0) 
                         for k in samples_so_far}
        
        # save checkpoint
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "samples": samples_so_far,
                "last_state": last_state,
                "num_samples_collected": num_collected
            }, f)
        print(f"Saved checkpoint with {num_collected}/{samples} samples")
    
    return samples_so_far
