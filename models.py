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

DROP_SITES = {"logits_theta", "logits_psi"}

class FilteredNUTS(NUTS):
    """NUTS that omits large nuisance sites from the collected samples.
    constrain_fn still runs with all values present (no re-sampling, no key
    needed); we only discard the sites from the returned dict, before the
    cross-chain stack that was OOMing."""
    def postprocess_fn(self, args, kwargs):
        base = super().postprocess_fn(args, kwargs)
        def fn(z, *a, **k):
            out = base(z, *a, **k)          # z has every site -> replay is fine
            if isinstance(out, dict):
                return {key: v for key, v in out.items() if key not in DROP_SITES}
            return out
        return fn

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

def collapse_to_counts(doc_ids, word_ids, vocab_size):
    """
    Collapse parallel per-token (doc, word) arrays into unique pairs + counts.
    Returns int32 doc/word index arrays and a float32 count array, all (num_pairs,).
    num_pairs <= len(doc_ids), with equality only if no (doc,word) repeats.
    """
    doc_ids  = np.asarray(doc_ids)
    word_ids = np.asarray(word_ids)
    # encode (doc, word) -> single int64 key; safe as long as
    # max_doc * vocab_size + max_word < 2**63 (true for your sizes).
    keys = doc_ids.astype(np.int64) * np.int64(vocab_size) + word_ids.astype(np.int64)
    uniq_keys, counts = np.unique(keys, return_counts=True)
    uniq_doc  = (uniq_keys // vocab_size).astype(np.int32)
    uniq_word = (uniq_keys %  vocab_size).astype(np.int32)
    return uniq_doc, uniq_word, counts.astype(np.float32)

def model_no_topics_multinomial(
    vocab_size, num_src_subs, num_tgt_docs, num_tgt_subs,
    src_sub_ids, src_word_ids, src_counts,
    tgt_doc_ids, tgt_word_ids, tgt_counts,
    tgt_sub_ids, doc_edge_lists, has_edges_mask,
    edges_per_sub, local_edge_map):

    word_mean = numpyro.sample("mean", dist.Normal(0, 1).expand([1, vocab_size - 1]))
    word_var  = numpyro.sample("variance", dist.HalfNormal(1))

    logits_psi = numpyro.sample("logits_psi",
        dist.Normal(jnp.zeros((num_tgt_subs, vocab_size - 1)),
                    jnp.ones((num_tgt_subs, vocab_size - 1))))
    logits_psi = jnp.concatenate([word_mean + (logits_psi * word_var),
                                  jnp.zeros((num_tgt_subs, 1))], axis=1)

    logits_theta = numpyro.sample("logits_theta",
        dist.Normal(jnp.zeros((num_src_subs, vocab_size - 1)),
                    jnp.ones((num_src_subs, vocab_size - 1))))
    logits_theta = jnp.concatenate([word_mean + (logits_theta * word_var),
                                    jnp.zeros((num_src_subs, 1))], axis=1)

    log_theta_full = nn.log_softmax(logits_theta, axis=-1)
    log_psi_full   = nn.log_softmax(logits_psi,   axis=-1)

    log_mix = jnp.full((num_tgt_docs, vocab_size), -jnp.inf)

    # ---- ragged pair-gammas with per-tgt-sub mean ----
    edges_per_sub_flat = jnp.asarray(edges_per_sub).reshape(-1)            # (num_tgt_subs,)
    total_pairs = int(np.asarray(edges_per_sub).sum())                     # static
    offsets = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32),
                               jnp.cumsum(edges_per_sub_flat)[:-1].astype(jnp.int32)])

    # per-tgt-sub citation-propensity mean (the lambda analog)
    gamma_mean = numpyro.sample("gamma_mean", dist.Normal(0, 1).expand([num_tgt_subs]))
    gamma_var  = numpyro.sample("gamma_var",  dist.HalfNormal(1))
    gamma_z    = numpyro.sample("gammas", dist.Normal(0, 1).expand([total_pairs]))  # deviations

    # map each flat pair-slot to its owning tgt sub, so we can add that sub's mean
    # pair_owner[p] = which tgt sub owns flat slot p
    pair_owner = jnp.repeat(jnp.arange(num_tgt_subs), edges_per_sub_flat,
                            total_repeat_length=total_pairs)               # (total_pairs,)
    gammas = gamma_mean[pair_owner] + gamma_var * gamma_z                  # (total_pairs,)

    for edge_obj in doc_edge_lists:
        num_edges     = edge_obj["num_edges"]
        edges         = edge_obj["edges"]
        doc_ids       = edge_obj["doc_ids"]
        num_edge_docs = doc_ids.shape[0]

        t = tgt_sub_ids[doc_ids]                                           # (num_edge_docs,)
        local    = local_edge_map[t[:, None], edges]                       # (num_edge_docs, num_edges)
        pair_pos = offsets[t][:, None] + local                            # (num_edge_docs, num_edges)
        logit_gamma_src = gammas[pair_pos]                                # (num_edge_docs, num_edges)

        logit_gamma = jnp.concatenate(
            [jnp.zeros((num_edge_docs, 1)), logit_gamma_src], axis=1)
        log_gamma = nn.log_softmax(logit_gamma, axis=-1)

        log_theta_active = log_theta_full[edges]
        log_psi_active   = jnp.expand_dims(log_psi_full[t, :], axis=1)
        log_active = jnp.concatenate([log_psi_active, log_theta_active], axis=1)
        group_mix = logsumexp(log_gamma[:, :, None] + log_active, axis=1)
        log_mix = log_mix.at[doc_ids, :].set(group_mix)

    no_edge = ~has_edges_mask
    log_mix = jnp.where(no_edge[:, None], log_psi_full[tgt_sub_ids, :], log_mix)

    tgt_pair_ll = log_mix[tgt_doc_ids, tgt_word_ids]
    src_pair_ll = log_theta_full[src_sub_ids, src_word_ids]

    with numpyro.plate("src_pairs", src_counts.shape[0]):
        numpyro.factor("src_likelihood", src_counts * src_pair_ll)
    with numpyro.plate("tgt_pairs", tgt_counts.shape[0]):
        numpyro.factor("tgt_likelihood", tgt_counts * tgt_pair_ll)

def model_no_topics(vocab_size, num_src_subs, num_tgt_docs, num_tgt_subs, num_src_words,
                    num_tgt_words, src_word_ids, tgt_word_ids, src_doc_ids, tgt_doc_ids, tgt_sub_ids,
                    doc_edge_lists, has_edges_mask, alpha_vocab, alpha_edges, lambda_theta, lambda_psi):
    """
    Marginalized LDA (integrating out topic assignments z).
    Likelihood via numpyro.factor on observed-word log-probs (no vocab-sized
    token matrices). VALID ONLY because each mixture component is a normalized
    log_softmax distribution, so the convex mixture is normalized and needs no
    per-token vocab normalizer.
    """

    with numpyro.plate("coin_flips", num_tgt_subs):
        lambdA = numpyro.sample("lambda_logit", dist.Normal(0, 1))
    lambdA = numpyro.deterministic("lambda", nn.sigmoid(lambdA))

    num_src_words = len(src_word_ids)
    
    word_mean = numpyro.sample("mean", dist.Normal (0, 1).expand([1, vocab_size-1]))
    word_var = numpyro.sample("variance", dist.HalfNormal(1))

    # Prior over word probabilities (pinned reference logit -> num_*-1 free dims)
    logits_psi = numpyro.sample(
        "logits_psi",
        dist.Normal(jnp.zeros((num_tgt_subs, vocab_size - 1)),
                    jnp.ones((num_tgt_subs, vocab_size - 1)))
    )
    logits_psi = jnp.concatenate([word_mean + (logits_psi*word_var), jnp.zeros((num_tgt_subs, 1))], axis=1)

    logits_theta = numpyro.sample(
        "logits_theta",
        dist.Normal(jnp.zeros((num_src_subs, vocab_size - 1)),
                    jnp.ones((num_src_subs, vocab_size - 1)))
    )
    logits_theta = jnp.concatenate([word_mean + (logits_theta*word_var), jnp.zeros((num_src_subs, 1))], axis=1)

    log_theta_full = nn.log_softmax(logits_theta, axis=-1)  # (num_src_subs, vocab_size)
    log_psi_full   = nn.log_softmax(logits_psi,   axis=-1)  # (num_tgt_subs, vocab_size)

    # Per-doc theta-mixture log-probs over full vocab.
    # (num_tgt_docs, vocab_size) — small (docs, not words); init -inf for no-edge docs.
    log_theta_mix = jnp.full((num_tgt_docs, vocab_size), -jnp.inf)

    gamma_var = numpyro.sample("gamma_var", dist.HalfNormal(1))

    for edge_obj in doc_edge_lists:
        num_edges     = edge_obj["num_edges"]
        edges         = edge_obj["edges"]     # (num_edge_docs, num_edges)
        doc_ids       = edge_obj["doc_ids"]   # (num_edge_docs,)
        num_edge_docs = doc_ids.shape[0]

        with numpyro.plate("tgt_edges_{}".format(num_edges), num_edge_docs):
            if num_edges > 1:
                logit_gamma = numpyro.sample(
                    "logit_gamma_{}".format(num_edges),
                    dist.Normal(jnp.zeros(num_edges-1),
                                jnp.ones(num_edges-1)).to_event(1)
                )
                logit_gamma = jnp.concatenate(
                   [logit_gamma*gamma_var, jnp.zeros((num_edge_docs, 1))], axis=1)
        if num_edges == 1:
            logit_gamma = jnp.ones((num_edge_docs, 1))

        log_gamma = nn.log_softmax(logit_gamma, axis=-1)

        log_theta_active = log_theta_full[edges]   # (num_edge_docs, num_edges, vocab_size)
        group_mix = logsumexp(log_gamma[:, :, None] + log_theta_active, axis=1)  # (num_edge_docs, vocab_size)
        log_theta_mix = log_theta_mix.at[doc_ids, :].set(group_mix)

    # ---- token-level GATHER of observed words only: all 1-D, no vocab axis ----
    log_theta_mix_tok = log_theta_mix[tgt_doc_ids, tgt_word_ids]            # (num_tgt_words,)
    log_psi_tok       = log_psi_full[tgt_sub_ids[tgt_doc_ids], tgt_word_ids] # (num_tgt_words,)

    lambda_doc = lambdA[tgt_sub_ids[tgt_doc_ids]]
    lambda_doc = jnp.where(has_edges_mask[tgt_doc_ids], lambda_doc, 1.0)
    log_lam    = jnp.log(lambda_doc)       # (num_tgt_words,)  no [:, None] now
    log_1mlam  = jnp.log1p(-lambda_doc)    # (num_tgt_words,)

    # top-level mixture, observed word only: (num_tgt_words,)
    tgt_token_ll = logsumexp(
        jnp.stack([log_lam + log_psi_tok, log_1mlam + log_theta_mix_tok], axis=0),
        axis=0
    )

    # src side: pure gather, (num_src_words,)
    src_token_ll = log_theta_full[src_doc_ids, src_word_ids]

    with numpyro.plate("src_tokens", num_src_words):
        numpyro.factor("src_likelihood", src_token_ll)
    with numpyro.plate("tgt_tokens", num_tgt_words):
        numpyro.factor("tgt_likelihood", tgt_token_ll)

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
    "no_topics": model_no_topics,
    "no_topics_multinomial": model_no_topics_multinomial
}


import numpy as np

def build_tgt_sub_edge_maps(edges, subreddits, num_tgt_subs, num_src_subs):
    """
    Collapse tgt docs into per-tgt-subreddit supernodes and build:

    1. distinct_counts : (num_tgt_subs, 1) int
       Number of distinct src subreddits connected to each tgt subreddit
       (after merging all that sub's docs and de-duplicating edges).

    2. local_index_map : (num_tgt_subs, num_src_subs) int
       local_index_map[t, s] = compact local index (0..distinct_counts[t]-1)
       of global src sub `s` within tgt sub `t`'s mapping, or -1 if src sub `s`
       is not connected to any doc in tgt sub `t`.

    Parameters
    ----------
    edges : sequence length num_tgt_docs
        edges[k] = iterable of global src-sub ids connected to tgt doc k.
    subreddits : sequence length num_tgt_docs
        subreddits[k] = global tgt-sub id of tgt doc k.
    """
    # 1. gather the set of distinct src subs per tgt sub (supernode dedup)
    #    use a set per tgt sub so duplicates across docs collapse.
    src_sets = [set() for _ in range(num_tgt_subs)]
    for doc_k, src_list in enumerate(edges):
        t = subreddits[doc_k]
        src_sets[t].update(int(s) for s in src_list)

    # 2. distinct counts -> (num_tgt_subs, 1)
    distinct_counts = np.array([[len(s)] for s in src_sets], dtype=np.int64)

    # 3. local index map -> (num_tgt_subs, num_src_subs), -1 default
    local_index_map = np.full((num_tgt_subs, num_src_subs), -1, dtype=np.int64)
    for t, s_set in enumerate(src_sets):
        # assign compact local indices in a deterministic order (sorted by global id)
        for local_idx, global_src in enumerate(sorted(s_set)):
            local_index_map[t, global_src] = local_idx

    return distinct_counts, local_index_map

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
    elif model_name == "gammas_unpooled" or model_name == "no_topics" or model_name == "no_topics_multinomial":
        model_args["doc_edge_lists"] = group_by_edge_count(text_network.edges)
    if model_name == "no_topics":
        del model_args["num_topics"]
        del model_args["alpha_topics"]
    if model_name == "no_topics_multinomial":
        src_sub_ids_u, src_word_ids_u, src_counts = collapse_to_counts(src_doc_ids, src_word_ids, vocab_size)
        tgt_doc_ids_u, tgt_word_ids_u, tgt_counts = collapse_to_counts(tgt_doc_ids, tgt_word_ids, vocab_size)
        model_args["src_sub_ids"] = src_sub_ids_u
        model_args["src_word_ids"] = src_word_ids_u
        model_args["src_counts"] = src_counts
        model_args["tgt_doc_ids"] = tgt_doc_ids_u
        model_args["tgt_word_ids"] = tgt_word_ids_u
        model_args["tgt_counts"] = tgt_counts
        model_args["edges_per_sub"], model_args["local_edge_map"] = build_tgt_sub_edge_maps(text_network.edges, text_network.subreddits, model_args["num_tgt_subs"], model_args["num_src_subs"])
        del model_args["src_doc_ids"]
        del model_args["num_topics"]
        del model_args["alpha_topics"]
        del model_args["alpha_vocab"]
        del model_args["alpha_edges"]
        del model_args["lambda_theta"]
        del model_args["lambda_psi"]
        del model_args["num_tgt_words"]
        del model_args["num_src_words"]
    return model_args

def run_model(text_network, topics, alpha_sum_topics, alpha_sum_vocab, alpha_edges, samples, warmup, num_chains, model_name, checkpoint_dir, checkpoint_interval=10):
    
    start = time.time()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)

    if model_name not in MODEL_MAP:
        raise ValueError("Unsupported model type")

    model = MODEL_MAP[model_name]
    nuts_kernel = FilteredNUTS(model, dense_mass=[("gamma_mean", "gamma_var")], max_tree_depth=12)
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
    mcmc.run(jr.PRNGKey(11), **model_args, extra_fields=("num_steps", "diverging", "adapt_state.step_size"))
 
    ef = mcmc.get_extra_fields()
    print("final step size per chain:", ef["adapt_state.step_size"][-1])
    print("mean tree depth:", np.log2(np.asarray(ef["num_steps"])+1).mean())
    print("divergences:", np.asarray(ef["diverging"]).sum())

    inf_data = az.from_numpyro(mcmc, log_likelihood=False)
    #print(az.summary(inf_data, var_names = ["^lambda*"], filter_vars="regex"))
    print(az.summary(inf_data, var_names = ["^gamma*"], filter_vars="regex"))
    print("TIME: ", time.time() - start)
    #samples_so_far = mcmc.get_samples(group_by_chain=True)
    
    #psi3 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_psi"][0,:,3,:]), axis=0))
    #psi8 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_psi"][0,:,8,:]), axis=0))
    #theta0 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_theta"][0,:,0,:]), axis=0))
    #theta36 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_theta"][0,:,36,:]), axis=0))
    #theta47 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_theta"][0,:,47,:]), axis=0))
    #theta168 = np.asarray(jnp.mean(nn.softmax(samples_so_far["logits_theta"][0,:,168,:]), axis=0))
    
    #from numpy.linalg import norm

    #pairs = {
    #    "psi3": psi3,
    #    "psi8": psi8,
    #}

    #thetas = {
    #    "theta0": theta0,
    #    "theta36": theta36,
    #    "theta47": theta47,
    #    "theta168": theta168,
    #}

    #for psi_name, psi_vec in pairs.items():
    #    for theta_name, theta_vec in thetas.items():
    #        cos_sim = np.dot(psi_vec, theta_vec) / (norm(psi_vec) * norm(theta_vec))
    #        print(f"cos_sim({psi_name}, {theta_name}) = {cos_sim:.4f}")    

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
