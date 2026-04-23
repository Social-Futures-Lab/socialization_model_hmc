import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
import json
import numpy as np
import random
from utils import read2D, read1D, flatten_docs, flatten_docs_ragged, group_by_edge_count, has_edges_bool_mask, build_edge_mask
from text_network import TextNetwork, Blob, TextBlob
numpyro.set_platform('cuda')

class TextBlob:
    def _preprocess(self, text):
        return text.lower().split()
    def __init__(self, text):
        self.text = self._preprocess(text)

class TextNetwork:
    def compute_vocab_size(self, src_blobs, tgt_blobs):
        num_words = 0
        for blob in src_blobs:
            for word in blob:
                num_words = max(num_words, word)
        for blob in tgt_blobs:
            for word in blob:
                num_words = max(num_words, word)
        return num_words+1
    def __init__(self, src_blobs, tgt_blobs, edges, subreddits):
        """
            src_blobs: List of lists. Each list contains words associated with a particular source subreddit
            tgt_blobs: List of lists. Each list contains words associated with a particular target subreddit,user pair
            edges: List of lists. List i contains the list of src_blob indices that target blob i is connected to
            #subreddits: list of the subreddit associated with each tgt_blob in tgt_blobs
        """
        self.src_blobs = src_blobs
        self.tgt_blobs = tgt_blobs
        self.edges = edges
        self.subreddits = subreddits
        self.vocab_size = self.compute_vocab_size(src_blobs, tgt_blobs)
        self.num_src_subreddits = len(src_blobs)
        self.num_tgt_subreddits = max(subreddits) + 1


"""## Data Reading ##"""

src_blobs_file = "/content/hp/src_blobs.txt"
tgt_blobs_file = "/content/hp/tgt_blobs.txt"
edges_file = "/content/hp/edges.txt"
subreddits_file = "/content/hp/subreddits.txt"

src_blobs = read2D(src_blobs_file)
tgt_blobs = read2D(tgt_blobs_file)
tgt_subreddits = read1D(subreddits_file)
edges = read2D(edges_file)

num_src_subs = len(src_blobs)
num_tgt_subs = max(tgt_subreddits) + 1;
num_tgt_blobs = len(tgt_blobs)
num_src_words = sum([len(blob) for blob in src_blobs])
num_tgt_words = sum([len(blob) for blob in tgt_blobs])

vocab_size = -1;
cur_row = 0;
for row in src_blobs:
    if len(row) > 0:
        row_max = max(row)
        vocab_size = max(row_max, vocab_size)
for row in tgt_blobs:
    if len(row) > 0:
        row_max = max(row)
        vocab_size = max(row_max, vocab_size)

vocab_size += 1
network = TextNetwork(src_blobs, tgt_blobs, edges, tgt_subreddits)
num_topics = 50

"""## Fake Data Simulation ##"""

num_topics = 50
vocab_size = 1000
num_src_subs = 100
src_doc_size = 100
num_tgt_subs = 100
tgt_authors_per_sub = 100
tgt_doc_size = 100
num_edges = 20
num_tgt_blobs = num_tgt_subs*tgt_authors_per_sub
spec = gen_specification(num_src_subs, num_tgt_subs, tgt_authors_per_sub, vocab_size, src_doc_size,
                      tgt_doc_size, num_topics, num_edges)
network = genNetwork(spec)

num_src_words = num_src_subs*src_doc_size
num_tgt_words = num_tgt_subs*tgt_doc_size*tgt_authors_per_sub

"""## Gen ID lists ##"""

src_docs = network.src_blobs
tgt_docs = network.tgt_blobs

#src_word_ids, src_doc_ids = flatten_docs(src_docs)
#tgt_word_ids, tgt_doc_ids = flatten_docs(tgt_docs)

src_word_ids, src_doc_ids = flatten_docs_ragged(src_docs)
tgt_word_ids, tgt_doc_ids = flatten_docs_ragged(tgt_docs)

"""## Dirichlet over edges per tgt subreddit ##"""

#TO-DO: add typing
def lda_marginalized_model_pooled(num_topics, vocab_size, num_src_subs, num_tgt_docs, num_tgt_subs, num_src_words,
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

from numpyro.infer import MCMC, NUTS


import time

start = time.time()

nuts_kernel = NUTS(lda_marginalized_model_pooled)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000)
rng_key = jr.PRNGKey(96)
model_args = {
    "num_topics": num_topics,
    "vocab_size": vocab_size,
    "num_src_subs": num_src_subs,
    "num_tgt_docs": num_tgt_blobs,
    "num_src_words": num_src_words,
    "num_tgt_words": num_tgt_words,
    "src_word_ids": src_word_ids,
    "tgt_word_ids": tgt_word_ids,
    "src_doc_ids": src_doc_ids,
    "tgt_doc_ids": tgt_doc_ids,
    "adjacency_matrix": build_edge_mask(network.edges, num_src_subs), #List of dicts where the items are ("num_edges" -> int, "doc_ids" ->list, "edges" -> 2d mat)
    "has_edges_mask": has_edges_bool_mask(network.edges),
    "num_tgt_subs": network.num_tgt_subreddits,
    "tgt_sub_ids": np.array(network.subreddits),
    "alpha_topics": np.ones(num_topics) / 3,
    "alpha_vocab": np.ones(vocab_size) / 3,
    "alpha_edges": 5,
    "lambda_theta": 1,
    "lambda_psi": 1
}
mcmc.run(rng_key, **model_args)

import numpyro.diagnostics as diagnostics
import json
#mcmc.print_summary(print_vars=["lambda"])
samples = mcmc.get_samples()

subset_samples = {k: v for k, v in samples.items() if k in ["lambda"]}
print(subset_samples["lambda"].mean(axis=0))
with open("hp/idx2tgt_sub.json") as f:
    print(json.load(f))
# 3. Print summary for subset
#diagnostics.print_summary(subset_samples)

"""## Base model, dirichlet over edges per user ##"""

#TO-DO: add typing
def lda_marginalized_model(num_topics, vocab_size, num_src_subs, num_tgt_docs, num_tgt_subs, num_src_words,
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

with open("/content/hp/idx2tgt_sub.json") as f:
    idx2tgt_sub = json.load(f)

with open("/content/hp/idx2src_sub.json") as f:
    idx2src_sub = json.load(f)

for edge_list, sub_id in zip(edges, network.subreddits):
    tgt_sub_name = idx2tgt_sub[str(sub_id)]
    for src_sub in edge_list:
        src_sub_name = idx2src_sub[str(src_sub)]
        if src_sub_name == tgt_sub_name:
            print(src_sub_name, tgt_sub_name)

from numpyro.infer import MCMC, NUTS


import time

start = time.time()
nuts_kernel = NUTS(lda_marginalized_model)
mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=100)
rng_key = jr.PRNGKey(96)
model_args = {
    "num_topics": num_topics,
    "vocab_size": vocab_size,
    "num_src_subs": num_src_subs,
    "num_tgt_docs": num_tgt_blobs,
    "num_src_words": num_src_words,
    "num_tgt_words": num_tgt_words,
    "src_word_ids": src_word_ids,
    "tgt_word_ids": tgt_word_ids,
    "src_doc_ids": src_doc_ids,
    "tgt_doc_ids": tgt_doc_ids,
    "doc_edge_lists": group_by_edge_count(network.edges),
    "has_edges_mask": has_edges_bool_mask(network.edges),
    "num_tgt_subs": network.num_tgt_subreddits,
    "tgt_sub_ids": np.array(network.subreddits),
    "alpha_topics": np.ones(num_topics) / 3,
    "alpha_vocab": np.ones(vocab_size) / 3,
    "alpha_edges": 5,
    "lambda_theta": 1,
    "lambda_psi": 1
}
mcmc.run(rng_key, **model_args)
mcmc.print_summary()

import numpyro.diagnostics as diagnostics
import json
#mcmc.print_summary(print_vars=["lambda"])
samples = mcmc.get_samples()

subset_samples = {k: v for k, v in samples.items() if k in ["lambda"]}
print(subset_samples["lambda"].mean(axis=0))
with open("hp/idx2tgt_sub.json") as f:
    print(json.load(f))
# 3. Print summary for subset
#diagnostics.print_summary(subset_samples)

print(spec.coin_flip_probs)

np.array(spec.tgt_topic_vectors)

"""## Dirichlet over edges, pymc ##"""

!pip install nutpie

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import jax
print(jax.devices())
jax.config.update("jax_default_device", jax.devices("gpu")[0])

def lda_marginalized_model_pooled(
    num_topics, vocab_size, num_src_subs, num_tgt_docs, num_tgt_subs,
    num_src_words, num_tgt_words, src_word_ids, tgt_word_ids,
    src_doc_ids, tgt_doc_ids, tgt_sub_ids, adjacency_matrix,
    has_edges_mask, alpha_topics, alpha_vocab, alpha_edges,
    lambda_theta, lambda_psi
):
    """
    Marginalized LDA (integrating out topic assignments z) — PyMC version
    """
    with pm.Model() as model:

        # topic distribution per src subreddit
        theta = pm.Dirichlet(
            "theta",
            a=alpha_topics,
            shape=(num_src_subs, num_topics)
        )

        # topic distribution per tgt subreddit
        psi = pm.Dirichlet(
            "psi",
            a=alpha_topics,
            shape=(num_tgt_subs, num_topics)
        )

        # citation distribution per tgt subreddit
        gamma = pm.Dirichlet(
            "gamma",
            a=alpha_edges * np.ones(num_src_subs),
            shape=(num_tgt_subs, num_src_subs)
        )

        # mask by adjacency and renormalize per doc
        full_edge_mat = gamma[tgt_sub_ids]  # (num_tgt_docs, num_src_subs)
        full_edge_mat = full_edge_mat * adjacency_matrix
        row_sums = full_edge_mat.sum(axis=1, keepdims=True)
        safe_row_sums = pt.switch(pt.eq(row_sums, 0.0), 1.0, row_sums)
        full_edge_mat = full_edge_mat / safe_row_sums
        full_edge_mat = pt.switch(
            has_edges_mask[:, None],
            full_edge_mat,
            0.0
        )

        # citation probability per tgt subreddit
        lambdA = pm.Beta(
            "lambda",
            alpha=lambda_psi,
            beta=lambda_theta,
            shape=num_tgt_subs
        )

        # word distribution per topic
        phi = pm.Dirichlet(
            "phi",
            a=alpha_vocab,
            shape=(num_topics, vocab_size)
        )

        # --- src likelihood ---
        log_theta_src = pt.log(theta[src_doc_ids])          # (num_src_words, K)
        log_phi_src = pt.log(phi[:, src_word_ids]).T        # (num_src_words, K)
        src_doc_factor = log_theta_src + log_phi_src        # (num_src_words, K)
        src_ll = pm.math.logsumexp(src_doc_factor, axis=1)  # (num_src_words,)
        pm.Potential("src_likelihood", src_ll.sum())

        # --- tgt likelihood ---
        log_phi_tgt = pt.log(phi[:, tgt_word_ids]).T        # (num_tgt_words, K)

        lambda_doc = lambdA[tgt_sub_ids]                    # (num_tgt_docs,)
        lambda_doc = pt.switch(
            has_edges_mask,
            lambda_doc,
            1.0
        )

        weighted_psi = pt.shape_padright(lambda_doc) * psi[tgt_sub_ids, :]          # (num_tgt_docs, K)
        weighted_theta = pt.shape_padright(1 - lambda_doc) * pt.dot(full_edge_mat, theta)  # (num_tgt_docs, K)
        log_topics = pt.log((weighted_psi + weighted_theta)[tgt_doc_ids])           # (num_tgt_words, K)

        tgt_doc_factor = log_topics + log_phi_tgt           # (num_tgt_words, K)
        tgt_ll = pm.math.logsumexp(tgt_doc_factor, axis=1)  # (num_tgt_words,)
        pm.Potential("tgt_likelihood", tgt_ll.sum())

    return model

nuts_sampler="nutpie",
nuts_sampler_kwargs={"backend": "jax"}

model_args = {
    "num_topics": num_topics,
    "vocab_size": vocab_size,
    "num_src_subs": num_src_subs,
    "num_tgt_docs": num_tgt_subs*tgt_authors_per_sub,
    "num_src_words": num_src_subs*src_doc_size,
    "num_tgt_words": num_tgt_subs*tgt_doc_size*tgt_authors_per_sub,
    "src_word_ids": src_word_ids,
    "tgt_word_ids": tgt_word_ids,
    "src_doc_ids": src_doc_ids,
    "tgt_doc_ids": tgt_doc_ids,
    "adjacency_matrix": build_edge_mask(network.edges, num_src_subs), #List of dicts where the items are ("num_edges" -> int, "doc_ids" ->list, "edges" -> 2d mat)
    "has_edges_mask": has_edges_bool_mask(network.edges),
    "num_tgt_subs": network.num_tgt_subreddits,
    "tgt_sub_ids": np.array(network.subreddits),
    "alpha_topics": np.ones(num_topics) / 3,
    "alpha_vocab": np.ones(vocab_size) / 3,
    "alpha_edges": 5,
    "lambda_theta": 1,
    "lambda_psi": 1
}

model = lda_marginalized_model_pooled(**model_args)

with model:
    trace = pm.sample(
        draws=100,
        tune=100,
        chains=1,
        nuts_sampler="nutpie",
        nuts_sampler_kwargs={"backend": "jax"},
    )

