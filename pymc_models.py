import numpy as np
import pytensor
pytensor.config.floatX = "float32"
import pytensor.tensor as pt
import pymc as pm
import nutpie
import arviz as az
import time, os
import random
import time
import json
from utils import read2D, read1D, flatten_docs, flatten_docs_ragged, group_by_edge_count, has_edges_bool_mask, build_edge_mask, calc_vocab_size
from models import gen_model_args

import matplotlib
matplotlib.use("Agg")          # at the very top, before `import matplotlib.pyplot`
import matplotlib.pyplot as plt

import jax
jax.devices() # [CudaDevice(id=0)]

DROP_VARS = ["logits_theta", "logits_psi"]  # large nuisance sites

def _logsumexp(x, axis):
    x_max_keep = pt.max(x, axis=axis, keepdims=True)
    summed = pt.sum(pt.exp(x - x_max_keep), axis=axis, acc_dtype="float32")
    return pt.log(summed) + pt.max(x, axis=axis)

def _logsumexp_keep(x, axis):
    x_max = pt.max(x, axis=axis, keepdims=True)
    summed = pt.sum(pt.exp(x - x_max), axis=axis, keepdims=True, acc_dtype="float32")
    return pt.log(summed) + x_max

def _log_softmax(x, axis):
    return x - _logsumexp_keep(x, axis)

def build_pymc_model(
    vocab_size, num_src_subs, num_tgt_docs, num_tgt_subs,
    src_sub_ids, src_word_ids, src_counts,
    tgt_doc_ids, tgt_word_ids, tgt_counts,
    tgt_sub_ids, doc_edge_lists, has_edges_mask,
    edges_per_sub, local_edge_map):
    """
    PyMC port of model_no_topics_multinomial. All data-derived index arithmetic
    is done in NumPy (constants); only parameter-dependent ops are PyTensor.
    """

    # ---- static (NumPy) precompute: identical math to the NumPyro version ----
    edges_per_sub_flat = np.asarray(edges_per_sub).reshape(-1).astype(np.int64)
    total_pairs = int(edges_per_sub_flat.sum())
    offsets = np.concatenate([[0], np.cumsum(edges_per_sub_flat)[:-1]]).astype(np.int64)
    pair_owner = np.repeat(np.arange(num_tgt_subs), edges_per_sub_flat).astype(np.int64)

    tgt_sub_ids    = np.asarray(tgt_sub_ids)
    has_edges_mask = np.asarray(has_edges_mask).astype(bool)
    local_edge_map = np.asarray(local_edge_map)
    no_edge_ids    = np.where(~has_edges_mask)[0]
    
    src_counts = np.asarray(src_counts, dtype=np.float32)
    tgt_counts = np.asarray(tgt_counts, dtype=np.float32)

    # all the per-group index arrays, precomputed in NumPy so the graph only
    # ever does parameter gathers (never data-dependent index construction):
    groups = []
    for edge_obj in doc_edge_lists:
        edges   = np.asarray(edge_obj["edges"])      # (n, num_edges) global src ids
        doc_ids = np.asarray(edge_obj["doc_ids"])    # (n,)
        t       = tgt_sub_ids[doc_ids]               # (n,)
        local   = local_edge_map[t[:, None], edges]  # (n, num_edges)
        # guard against the -1 silent-corruption case discussed earlier:
        if (local < 0).any():
            raise ValueError("local_edge_map gave -1 for an existing edge — "
                             "edge list and supernode map disagree.")
        pair_pos = offsets[t][:, None] + local       # (n, num_edges)
        groups.append(dict(edges=edges, doc_ids=doc_ids, t=t, pair_pos=pair_pos))

    with pm.Model() as model:
        # ---- word background (non-centered hierarchy) ----
        word_mean = pm.Normal("mean", 0.0, 1.0, shape=(1, vocab_size - 1))
        word_var  = pm.HalfNormal("variance", 1.0)

        logits_psi_raw = pm.Normal("logits_psi", 0.0, 1.0,
                                   shape=(num_tgt_subs, vocab_size - 1))
        logits_psi = word_mean + logits_psi_raw * word_var
        logits_psi = pt.concatenate(
            [logits_psi, pt.zeros((num_tgt_subs, 1))], axis=1)

        logits_theta_raw = pm.Normal("logits_theta", 0.0, 1.0,
                                     shape=(num_src_subs, vocab_size - 1))
        logits_theta = word_mean + logits_theta_raw * word_var
        logits_theta = pt.concatenate(
            [logits_theta, pt.zeros((num_src_subs, 1))], axis=1)

        log_theta_full = _log_softmax(logits_theta, axis=-1)  # (num_src_subs, V)
        log_psi_full   = _log_softmax(logits_psi,   axis=-1)  # (num_tgt_subs, V)

        # ---- ragged pair-gammas with per-tgt-sub mean ----
#        gamma_mean_mean = pm.Normal("gamma_mean_mean", 0.0, 1.0)
        gamma_mean = pm.Normal("gamma_mean", 0.0, 1.0, shape=(num_tgt_subs,))
        gamma_var  = pm.HalfNormal("gamma_var", 1.0)
        gamma_z    = pm.Normal("gammas", 0.0, 1.0, shape=(total_pairs,))
        gammas = gamma_mean[pair_owner] + gamma_var*gamma_z         # (total_pairs,)

        # ---- build per-doc mixture log-probs ----
        log_mix = pt.full((num_tgt_docs, vocab_size), -np.inf, dtype="float32")

        for g in groups:
            edges    = g["edges"]
            doc_ids  = g["doc_ids"]
            t        = g["t"]
            pair_pos = g["pair_pos"]
            n = doc_ids.shape[0]

            logit_gamma_src = gammas[pair_pos]                        # (n, num_edges)
            logit_gamma = pt.concatenate(
                [pt.zeros((n, 1)), logit_gamma_src], axis=1)          # psi ref = 0
            log_gamma = _log_softmax(logit_gamma, axis=-1)     # (n, 1+num_edges)

            log_theta_active = log_theta_full[edges]                  # (n, num_edges, V)
            log_psi_active   = pt.expand_dims(log_psi_full[t], 1)     # (n, 1, V)
            log_active = pt.concatenate(
                [log_psi_active, log_theta_active], axis=1)           # (n, 1+num_edges, V)

            group_mix = _logsumexp(
                log_gamma[:, :, None] + log_active, axis=1)           # (n, V)
            log_mix = pt.set_subtensor(log_mix[doc_ids, :], group_mix)

        # ---- no-edge docs: pure psi ----
        if no_edge_ids.size > 0:
            log_mix = pt.set_subtensor(
                log_mix[no_edge_ids, :],
                log_psi_full[tgt_sub_ids[no_edge_ids], :])

        # ---- sparse-multinomial likelihood via Potentials ----
        tgt_pair_ll = log_mix[tgt_doc_ids, tgt_word_ids]              # (num_tgt_pairs,)
        src_pair_ll = log_theta_full[src_sub_ids, src_word_ids]       # (num_src_pairs,)

        pm.Potential("src_likelihood",
                     pt.sum(pt.as_tensor_variable(src_counts) * src_pair_ll))
        pm.Potential("tgt_likelihood",
                     pt.sum(pt.as_tensor_variable(tgt_counts) * tgt_pair_ll))

    return model

def run_model(text_network, topics, alpha_sum_topics, alpha_sum_vocab,
                     alpha_edges, samples, warmup, num_chains, model_name,
                     checkpoint_dir=None, checkpoint_interval=10):
    start = time.time()

    # reuse the existing arg generator — its no_topics_multinomial branch already
    # produces exactly the keys build_pymc_model expects.
    model_args = gen_model_args(text_network, topics, alpha_sum_topics,
                                alpha_sum_vocab, alpha_edges, samples, warmup,
                                model_name)

    print("Building PyMC model...")
    model = build_pymc_model(**model_args)

    print("Compiling with nutpie (JAX backend -> GPU if JAX sees it)...")
    compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")

    print("Sampling...")


    keep = [rv.name for rv in model.free_RVs if rv.name not in DROP_VARS]
    with model:
        idata = pm.sample(
            draws=samples, tune=warmup, chains=num_chains,
            nuts_sampler="nutpie",
            nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"},   # GPU path
            var_names=keep,
            random_seed=11,
        )

    # --- confirm the real dimension names first (run once, then hardcode) ---
    print("gamma_mean dims:", idata.posterior["gamma_mean"].dims)
    print("gammas dims:    ", idata.posterior["gammas"].dims)
    # expect something like ('chain','draw','gamma_mean_dim_0') and (...,'gammas_dim_0')

    GM_DIM = "gamma_mean_dim_0"   # <- replace with the actual name printed above
    GZ_DIM = "gammas_dim_0"       # <- replace with the actual name printed above

    # ===== PDF 1: gamma_mean[0] vs gamma_var =====
    axes = az.plot_pair(
        idata,
        var_names=["gamma_mean", "gamma_var"],
        coords={GM_DIM: [0]},
        kind="scatter",
        marginals=True,
    )
    # plot_pair returns an array of axes (or a single ax); grab the figure from either
    fig1 = (axes.ravel()[0] if hasattr(axes, "ravel") else axes).figure
    fig1.suptitle("gamma_mean[0] vs gamma_var", y=1.02)
    fig1.savefig("mean_vs_var.pdf", bbox_inches="tight")
    plt.close(fig1)

    # ===== PDF 2: gamma_mean[0] vs a few of sub 0's gamma_z slots =====
    # sub 0's flat gamma_z indices are offsets[0] : offsets[0]+edges_per_sub[0]
    # (offsets[0] == 0, so the first few slots belong to sub 0)
    gz_slots = [0, 1, 2]   # pick a few; add the slot for a heavily-shared source if known

    axes2 = az.plot_pair(
        idata,
        var_names=["gamma_mean", "gammas"],
        coords={GM_DIM: [0], GZ_DIM: gz_slots},
        kind="scatter",
        marginals=True,
    )
    fig2 = (axes2.ravel()[0] if hasattr(axes2, "ravel") else axes2).figure
    fig2.suptitle("gamma_mean[0] vs sub-0 gamma_z slots", y=1.02)
    fig2.savefig("mean_vs_z.pdf", bbox_inches="tight")
    plt.close(fig2)


    axes3 = az.plot_trace(idata, var_names=["gamma_mean"])
    fig3 = (axes3.ravel()[0] if hasattr(axes3, "ravel") else axes3).figure
    fig3.savefig("traces.pdf", bbox_inches="tight")
    plt.close(fig3)

    # ---- diagnostics (nutpie populates arviz sample_stats) ----
    ss = idata.sample_stats
    try:
        # nutpie names: 'step_size' (or 'step_size_bar'), 'depth', 'diverging'
        step = ss["step_size"].values
        print("final step size per chain:", step[:, -1])
    except KeyError:
        print("step_size field not found; available:", list(ss.data_vars))
    for depth_key in ("depth", "tree_depth"):
        if depth_key in ss:
            print("mean tree depth:", float(ss[depth_key].mean()))
            break
    if "diverging" in ss:
        print("divergences:", int(ss["diverging"].sum()))

    # ---- drop the large nuisance sites from the stored trace (memory) ----
    keep = [v for v in idata.posterior.data_vars if v not in DROP_VARS]
    idata.posterior = idata.posterior[keep]

    import pandas as pd
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None); pd.set_option("display.max_rows", None)
    print(az.summary(idata, var_names=["^gamma", "^mean", "^variance"],
                     filter_vars="regex"))
    print("TIME: ", time.time() - start)
    return idata
