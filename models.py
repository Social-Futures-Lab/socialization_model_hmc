import jax.numpy as jnp
import jax.nn as nn
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
import pandas as pd
import os
from utils import read2D, read1D, flatten_docs, flatten_docs_ragged, group_by_edge_count, has_edges_bool_mask, build_edge_mask, calc_vocab_size
import arviz as az


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
            dist.Beta(concentration1=lambda_psi, concentration0=lambda_theta)
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


def sample_dirichlet_via_gamma(name, alpha):
    """
    Sample a simplex `pi ~ Dirichlet(alpha)` parameterized through Gammas.

    Parameters
    ----------
    name : str
        Site name for the returned simplex (registered as a deterministic).
        Latent gammas are sampled at "{name}_gamma".
    alpha : array_like, shape (K,)
        Dirichlet concentration parameters. May contain entries < 1 (sparse).
    identified : bool, default True
        If True, remove the free overall-scale direction by pinning the last
        gamma primitive to 1 and adding the exact density correction. This is
        the version you almost always want as a prior inside a larger model.
        If False, sample all K gammas and normalize (simpler, but the
        non-identifiability can force tiny NUTS step sizes / low ESS).

    Returns
    -------
    pi : jnp.ndarray, shape (K,)
        A simplex distributed as Dirichlet(alpha).
    """
    alpha = jnp.asarray(alpha, dtype=float)
    K = alpha.shape[-1]

    # Naive route: K independent Gammas, normalize. Exactly Dirichlet(alpha),
    # but the scale sum(g) is unconstrained -> a non-identified ridge.
    g = numpyro.sample(f"{name}_gamma", dist.Gamma(alpha, 1.0).to_event(1))
    pi = g / jnp.sum(g, axis=-1, keepdims=True)   # <-- axis=-1, keepdims=True
    return numpyro.deterministic(name, pi)

#TO-DO: add typing
def model_no_topics(vocab_size, num_src_subs, num_tgt_docs, num_tgt_subs, num_src_words,
                           num_tgt_words, src_word_ids, tgt_word_ids, src_doc_ids, tgt_doc_ids, tgt_sub_ids,
                           doc_edge_lists, has_edges_mask, alpha_vocab, alpha_edges, lambda_theta, lambda_psi):

    """
    Marginalized LDA (integrating out topic assignments z)
    """
    
    # citation probability per tgt document
    #with numpyro.plate("coin_flips", num_tgt_subs):
    #    lambdA = numpyro.sample(
    #        "lambda",
    #        dist.Beta(concentration0=lambda_theta, concentration1=lambda_psi)
    #    )  # shape: (num_tgt_subs, 1)
    with numpyro.plate("coin_flips", num_tgt_subs):
        lambdA = numpyro.sample(
            "lambda_logit",
            dist.Normal(0, 1)
        )  # shape: (num_tgt_subs, 1)
    
    lambdA = numpyro.deterministic("lambda", nn.sigmoid(lambdA))

    num_src_words = len(src_word_ids)

    # Prior over word probabilities for each sub-source
    logits_psi = numpyro.sample(
        "logits_psi",
        dist.Normal(
            jnp.zeros((num_tgt_subs, vocab_size-1)),
            jnp.ones((num_tgt_subs, vocab_size-1))
        )
    )
    logits_psi = jnp.concatenate([logits_psi, jnp.zeros((num_tgt_subs, 1))], axis=1)
    logits_theta = numpyro.sample(
        "logits_theta",
        dist.Normal(
            jnp.zeros((num_src_subs, vocab_size-1)),
            jnp.ones((num_src_subs, vocab_size-1))
        )
    )
    logits_theta = jnp.concatenate([logits_theta, jnp.zeros((num_src_subs, 1))], axis=1)

    # log-probs directly from logits (stable)
    log_theta_full = nn.log_softmax(logits_theta, axis=-1)  # (num_src_subs, vocab_size)
    log_psi_full   = nn.log_softmax(logits_psi,   axis=-1)  # (num_tgt_subs, vocab_size)

    # Accumulate the per-doc theta-mixture log-probs here.
    # Init to -inf so any doc never written (no edges) contributes nothing if added in log space.
    log_theta_mix = jnp.full((num_tgt_docs, vocab_size), -jnp.inf)

    for edge_obj in doc_edge_lists:
        num_edges     = edge_obj["num_edges"]
        edges         = edge_obj["edges"]     # (num_edge_docs, num_edges) src-sub indices
        doc_ids       = edge_obj["doc_ids"]   # (num_edge_docs,) tgt-doc indices
        num_edge_docs = doc_ids.shape[0]

        with numpyro.plate("tgt_edges_{}".format(num_edges), num_edge_docs):
            if num_edges > 1:
               # gamma = numpyro.sample(
               #     "gamma_{}".format(num_edges),
               #     dist.Dirichlet(alpha_edges * np.ones(num_edges))
               # )                                 # (num_edge_docs, num_edges)
                logit_gamma = numpyro.sample(
                                 "logit_gamma_{}".format(num_edges),
                                 dist.Normal(
                                     jnp.zeros(num_edges-1),
                                     jnp.ones(num_edges-1)
                               ).to_event(1)
                )
       #         print("logit_gamma", logit_gamma.shape)
                logit_gamma = jnp.concatenate([logit_gamma, jnp.zeros((num_edge_docs, 1))], axis = 1)
        # log-probs directly from logits (stable)
        if num_edges == 1:
            logit_gamma = jnp.ones((num_edge_docs, 1))  # (num_src_subs, vocab_size)
        log_gamma = numpyro.deterministic("gamma_{}".format(num_edges), nn.log_softmax(logit_gamma, axis=-1))  # (num_src_subs, vocab_size)
        # Gather only the active source-sub log-probs for this group:
        #   log_theta_full[edges] -> (num_edge_docs, num_edges, vocab_size)
        log_theta_active = log_theta_full[edges]                    # fancy-index gather

        # Mixture in log space: log( Σ_k γ_k θ_k ) over the num_edges active comps
        group_mix = logsumexp(
            log_gamma[:, :, None] + log_theta_active,
            axis=1
        )                                                           # (num_edge_docs, vocab_size)

        # Scatter into the per-doc result matrix
        log_theta_mix = log_theta_mix.at[doc_ids, :].set(group_mix)

    # --- now index to token level and apply the lambda gate, all in log space ---
    log_theta_mix_tok = log_theta_mix[tgt_doc_ids, :]               # (num_tgt_words, vocab_size)
    log_psi           = log_psi_full[tgt_sub_ids[tgt_doc_ids], :]   # (num_tgt_words, vocab_size)

    lambda_doc = lambdA[tgt_sub_ids[tgt_doc_ids]]
    lambda_doc = jnp.where(has_edges_mask[tgt_doc_ids], lambda_doc, 1.0)
    log_lam    = jnp.log(lambda_doc)[:, None]                       # (num_tgt_words, 1)
    log_1mlam  = jnp.log1p(-lambda_doc)[:, None]                    # log(1-λ)

    tgt_doc_logits = logsumexp(
        jnp.stack([log_lam + log_psi, log_1mlam + log_theta_mix_tok], axis=0),
        axis=0
    )                                                               # (num_tgt_words, vocab_size)

    with numpyro.plate("src_tokens", num_src_words):
        numpyro.sample("src_likelihood",
            dist.Categorical(logits=log_theta_full[src_doc_ids]),
            obs=src_word_ids
        )

    with numpyro.plate("tgt_tokens", num_tgt_words):
        numpyro.sample("tgt_likelihood",
            dist.Categorical(logits=tgt_doc_logits),
            obs=tgt_word_ids
        )

    # citation distribution per tgt document
    # documents are grouped by num edges and priors are generated in batches
    # then filled in tgt_docs x tgt_subs sized matrix
#    full_edge_mat = jnp.zeros((num_tgt_docs, num_src_subs))
 #   for edge_obj in doc_edge_lists:
  #      num_edges= edge_obj["num_edges"]
  #      edges= edge_obj["edges"]
  #      doc_ids = edge_obj["doc_ids"]
  #      num_edge_docs = doc_ids.shape[0]
  #      with numpyro.plate("tgt_edges_{}".format(num_edges), num_edge_docs):
  #          if num_edges > 1:
  #              gamma = numpyro.sample(
  #                          "gamma_{}".format(num_edges),
  #                          dist.Dirichlet(alpha_edges*np.ones(num_edges))
  #                      )
  #      if num_edges == 1:
  #          gamma = jnp.ones((num_edge_docs, 1))  # shape: (num_tgt_docs, num_src_subs)
  #      rows = doc_ids[:, None]  #Only users w/ edges will have a prior here
  #      full_edge_mat = full_edge_mat.at[rows, edges].set(gamma)

  #  psi = nn.softmax(unconstrained_logits, axis=-1)  # (num_tgt_subs, vocab_size)
 #   theta = nn.softmax(unconstrained_logits, axis=-1)  # (num_src_subs, vocab_size)

  #  psi  = psi[tgt_sub_ids[tgt_doc_ids], :] #(num_tgt_words, vocab_size)
   # lambda_doc = lambdA[tgt_sub_ids[tgt_doc_ids]] #(num_tgt_words, )
    #lambda_doc = jnp.where(has_edges_mask[tgt_doc_ids],
     #                      lambda_doc,
      #                     1.0
       #         )[:, None] # for users w/ no edges, there's no chance of citation
 #   weighted_psi= lambda_doc*psi
#    weighted_theta = (full_edge_mat@theta)[tgt_doc_ids, :]
  #  weighted_theta = (1-lambda_doc)*weighted_theta
   # tgt_doc_probs = weighted_psi + weighted_theta

    # Likelihood
   # with numpyro.plate("src_tokens", num_src_words):
   #     numpyro.sample(
   #         "src_likelihood",
   #         dist.Categorical(probs=theta[src_doc_ids]),  # (num_src_words, vocab_size)
   #         obs=src_word_ids
   #     )

    #with numpyro.plate("tgt_tokens", num_tgt_words):
     #   numpyro.sample(
     #       "tgt_likelihood",
     #       dist.Categorical(probs=tgt_doc_probs),  # (num_src_words, vocab_size)
     #       obs=tgt_word_ids
     #   )


    # word distribution per topic
    #with numpyro.plate("tgt_word_dists", num_tgt_subs):
    #    psi = numpyro.sample(
    #        "psi",
    #        dist.Dirichlet(alpha_vocab)
    #    )  # shape: (num_tgt_subs, V)

    # word distribution per topic
 #   with numpyro.plate("src_word_dists", num_src_subs):
 #   unconstrained_logits = numpyro.sample(
 #       "logits", 
 #       dist.Normal(jnp.zeros((num_src_subs, vocab_size)), jnp.ones((num_src_subs, vocab_size)))
 #   )
 #   prob_transform = dist.transforms.SoftmaxTransform()
 #   theta = numpyro.sample(
 #       "theta", 
  #      dist.TransformedDistribution(dist.Delta(unconstrained_logits), prob_transform)
  #  )

    # likelihood (src)
   # with numpyro.plate("src_tokens", num_src_words):
    #    log_theta  = jnp.log(theta[src_doc_ids, src_word_ids]) # (1, num_src_words)
    #    numpyro.factor(
    #        "src_likelihood",
    #        log_theta.sum()
    #    )
    
    #likelihood (tgt)
 #   with numpyro.plate("tgt_tokens", num_tgt_words):
  #      psi  = psi[tgt_sub_ids[tgt_doc_ids], tgt_word_ids] #(1, num_tgt_words)
   #     lambda_doc = lambdA[tgt_sub_ids[tgt_doc_ids]]
   #     lambda_doc = jnp.where(has_edges_mask[tgt_doc_ids],
    #                           lambda_doc,
    #                           1.0
    #                          ) # for users w/ no edges, there's no chance of citation
    #    weighted_log_psi= jnp.expand_dims(jnp.log(lambda_doc*psi), axis=0)
    #    weighted_log_theta = jnp.log((1-lambda_doc)*(theta[:, tgt_word_ids]))
    #    weighted_log_V = jnp.concatenate([weighted_log_theta, weighted_log_psi], axis=0).T # (N, Docs+1)       
    #    tgt_doc_factor = weighted_log_V + jnp.log(full_edge_mat[tgt_doc_ids, :])               # (N, K)
    #    numpyro.factor(
    #        "tgt_likelihood",
    #        logsumexp(tgt_doc_factor, axis=1)
    #    )

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
        theta = sample_dirichlet_via_gamma(
            "theta",
            alpha_topics
        )  # shape: (num_src_subs, num_topics)

    # topic distribution per tgt document
    with numpyro.plate("tgt_topics", num_tgt_subs):
        psi = sample_dirichlet_via_gamma(
            "psi",
            alpha_topics
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
            if num_edges > 1:
                gamma = sample_dirichlet_via_gamma(
                            "gamma_{}".format(num_edges),
                            alpha_edges*np.ones(num_edges)
                        )
        if num_edges == 1:
            gamma = jnp.ones((num_edge_docs, 1))  # shape: (num_tgt_docs, num_src_subs)
        rows = doc_ids[:, None]  #Only users w/ edges will have a prior here
        full_edge_mat = full_edge_mat.at[rows, edges].set(gamma)
    # citation probability per tgt document
    with numpyro.plate("coin_flips", num_tgt_subs):
        lambdA = numpyro.sample(
            "lambda",
            dist.Beta(concentration0=lambda_theta, concentration1=lambda_psi)
        )  # shape: (num_tgt_subs, 1)

    # word distribution per topic
    with numpyro.plate("word_dists", num_topics):
        phi = sample_dirichlet_via_gamma(
            "phi",
            alpha_vocab
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
    "gammas_unpooled": model_gammas_unpooled,
    "no_topics": model_no_topics
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
        "num_tgt_words": sum([len(tgt_blob) for tgt_blob in text_network.tgt_blobs]),
        "src_word_ids": src_word_ids,
        "tgt_word_ids": tgt_word_ids,
        "src_doc_ids": src_doc_ids,
        "tgt_doc_ids": tgt_doc_ids,
        "has_edges_mask": has_edges_bool_mask(text_network.edges),
        "num_tgt_subs": text_network.num_tgt_subreddits,
        "tgt_sub_ids": np.array(text_network.subreddits),
        "alpha_topics": np.ones(topics) * (alpha_sum_topics/topics),
        "alpha_vocab": np.ones(vocab_size)* (alpha_sum_vocab/vocab_size),
        "alpha_edges": alpha_edges,
        "lambda_theta": 1,
        "lambda_psi": 1
    }
    if model_name == "gammas_pooled":
        #List of dicts where the items are ("num_edges" -> int, "doc_ids" ->list, "edges" -> 2d mat)
        model_args["adjacency_matrix"] = build_edge_mask(text_network.edges,
                                                         len(text_network.src_blobs))
    elif model_name == "gammas_unpooled" or model_name == "no_topics":
        model_args["doc_edge_lists"] = group_by_edge_count(text_network.edges)
    if model_name == "no_topics":
        del model_args["num_topics"]
        del model_args["alpha_topics"]
    return model_args

def run_model(text_network, topics, alpha_sum_topics, alpha_sum_vocab, alpha_edges, samples, warmup, num_chains, model_name, checkpoint_dir, checkpoint_interval=10):

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)

    if model_name not in MODEL_MAP:
        raise ValueError("Unsupported model type")

    model = MODEL_MAP[model_name]
    nuts_kernel = NUTS(model)
    model_args = gen_model_args(text_network, topics, alpha_sum_topics, alpha_sum_vocab, alpha_edges, samples, warmup, model_name)
    checkpoint_path = os.path.join(checkpoint_dir, "mcmc_checkpoint.pkl")

    # load checkpoint or run warmup
 #   if os.path.exists(checkpoint_path):
 #       print("Resuming from checkpoint...")
  #      with open(checkpoint_path, "rb") as f:
  #          checkpoint = pickle.load(f)
  #      samples_so_far = checkpoint["samples"]
  #      last_state = checkpoint["last_state"]
  #      num_collected = checkpoint["num_samples_collected"]
  #  else:
    print("Starting fresh run, running warmup...")
    #mcmc = MCMC(nuts_kernel, num_warmup=warmup, num_samples=checkpoint_interval, num_chains=num_chains)
    mcmc = MCMC(nuts_kernel, num_warmup=warmup, num_samples=samples, num_chains=num_chains)
    mcmc.run(jr.PRNGKey(11), **model_args)
    inf_data = az.from_numpyro(mcmc)
    print(az.summary(inf_data, var_names = ["^lambda*", "^gamma*"], filter_vars="regex"))
    samples_so_far = mcmc.get_samples(group_by_chain=True)
    print(samples_so_far["logits_psi"].shape)
    print(samples_so_far["logits_theta"].shape)

    psi3 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_psi"][0,:,3,:]), axis=0))
    psi8 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_psi"][0,:,8,:]), axis=0))
    theta0 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_theta"][0,:,0,:]), axis=0))
    theta36 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_theta"][0,:,36,:]), axis=0))
    theta47 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_theta"][0,:,47,:]), axis=0))
    theta168 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_theta"][0,:,168,:]), axis=0))
    
    from numpy.linalg import norm

    pairs = {
        "psi3": psi3,
        "psi8": psi8,
    }

    thetas = {
        "theta0": theta0,
        "theta36": theta36,
        "theta47": theta47,
        "theta168": theta168,
    }

    for psi_name, psi_vec in pairs.items():
        for theta_name, theta_vec in thetas.items():
            cos_sim = np.dot(psi_vec, theta_vec) / (norm(psi_vec) * norm(theta_vec))
            print(f"cos_sim({psi_name}, {theta_name}) = {cos_sim:.4f}")    

    #mcmc.print_summary()

   # last_state = mcmc.last_state
   # num_collected = checkpoint_interval
   # with open(checkpoint_path, "wb") as f:
  #      pickle.dump({"samples": samples_so_far, "last_state": last_state, "num_samples_collected": num_collected}, f)
  #  print(f"Warmup done, saved checkpoint with {num_collected} samples")

    # sample in chunks
   # while num_collected < samples:
   #     chunk = min(checkpoint_interval, samples - num_collected)
   #     print(f"Sampling chunk of {chunk}, {num_collected}/{samples} collected so far...")
   #     mcmc = MCMC(nuts_kernel, num_warmup=0, num_chains=num_chains, num_samples=chunk)
   #     mcmc.post_warmup_state = last_state
   #     mcmc.run(mcmc.post_warmup_state.rng_key, **model_args)
   #     new_samples = mcmc.get_samples()
   #     last_state = mcmc.last_state
   #     num_collected += chunk
   #     samples_so_far = {k: jnp.concatenate([samples_so_far[k], new_samples[k]], axis=0)
   #                       for k in samples_so_far}
   #     with open(checkpoint_path, "wb") as f:
   #         pickle.dump({"samples": samples_so_far, "last_state": last_state, "num_samples_collected": num_collected}, f)
   #     print(f"Saved checkpoint with {num_collected}/{samples} samples")
    
    #samples_so_far = {k: jnp.expand_dims(samples_so_far[k], axis=0) for k in samples_so_far}
    #samples_so_far = {k: samples_so_far[k] for k in samples_so_far}
    #samples_so_far["log_prob"] = log_prob
#    print(samples_so_far["lambda"][0, :10, 0])
   # samples_so_far = {k: np.asarray(samples_so_far[k]) for k in samples_so_far}
#    subset_samples = {k: v.reshape(-1, *v.shape[2:]) for k,v in samples_so_far.items() if k in ["lambda", "gamma", "theta", "psi", "phi", "log_prob"]}
 #   return subset_samples
 #   return samples_so_far
