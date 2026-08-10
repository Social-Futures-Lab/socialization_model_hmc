import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np
from scipy.special import softmax
from text_network import TextNetwork
from models import build_tgt_sub_edge_maps
import pandas as pd
import sys
import json
import os
from poster_plots import plot_ingroup

input_dir = sys.argv[1]
results_dir = sys.argv[2]
output_file = sys.argv[3]

text_network = TextNetwork.load(input_dir) 

input_dir_path = input_dir
text_network = TextNetwork.load(input_dir_path)
parameters_path = "{}/parameters.npz".format(results_dir)
with open(input_dir_path + "/idx2tgt_sub.json") as f:
    idx2tgt_sub = json.load(f)

with open(input_dir_path + "/idx2src_sub.json") as f:
    idx2src_sub = json.load(f)

tgt2group = {
    "AskHistorians": "High",
    "changemyview": "High",
    "photocritique": "High",
    "lgbt": "Medium",
    "autism": "Medium",
    "nfl": "Medium",
    "CasualConversation": "Low",
    "pics": "Low",
    "AskReddit": "Low"
}

expected_entries = {"High": [], "Medium": [], "Low": []}
perc_by_sub = {}
for k in tgt2group.keys():
    perc_by_sub[k] = {}
    perc_by_sub[k][k] = 0
    for src_sub in idx2src_sub.values():
        perc_by_sub[k][src_sub] = 0
with np.load(parameters_path) as parameters:
    num_tgt_subs = max(text_network.subreddits) + 1
    num_src_subs = len(text_network.src_blobs)
    edges_per_sub, local_edge_map = build_tgt_sub_edge_maps(text_network.edges, text_network.subreddits, num_tgt_subs, num_src_subs)
    edges_per_sub_flat = np.asarray(edges_per_sub).reshape(-1)
    total_pairs = int(np.asarray(edges_per_sub).sum()) 
    _, pair_srcs = np.where(local_edge_map >= 0)   
    pair_owner = np.repeat(np.arange(num_tgt_subs), edges_per_sub_flat)
    offsets = np.concatenate([np.zeros((1,), dtype=np.int32),
                               np.cumsum(edges_per_sub_flat)[:-1].astype(np.int32)]) 
    records = []

    gamma_mean = parameters["gamma_mean"] 
    gamma_var = parameters["gamma_var"] 
    gamma_mean_src = parameters["gamma_mean_src"] 
    gamma_z = parameters["z_gamma"] 

    gamma_mean = gamma_mean.reshape(-1, gamma_mean.shape[2])
    gamma_var = gamma_var.flatten()
    gamma_mean_src = gamma_mean_src.reshape(-1, gamma_mean_src.shape[2])
    gamma_z = gamma_z.reshape(-1, gamma_z.shape[2])
    print(gamma_mean.shape[0])
    for iteration in range(0, gamma_mean.shape[0]):
        if iteration % 100 == 0:
            print("Iteraction: {} out of {}...".format(iteration, gamma_mean.shape[0]))
        gammas = gamma_mean[iteration, pair_owner] + gamma_mean_src[iteration, pair_srcs] +\
                 gamma_var[iteration]*gamma_z[iteration]
         
        subreddit_counts = {}
        for doc in range(len(text_network.tgt_blobs)):
            subreddit = text_network.subreddits[doc]
            if subreddit not in subreddit_counts:
                subreddit_counts[subreddit] = 0
            subreddit_counts[subreddit] += len(text_network.tgt_blobs[doc])

        avg_innovation_prob = {}
        for doc in range(len(text_network.tgt_blobs)):
            subreddit = text_network.subreddits[doc]
            sub_name = idx2tgt_sub[str(subreddit)]
            if subreddit not in avg_innovation_prob:
                avg_innovation_prob[subreddit] = 0
            edges = text_network.edges[doc]
            if len(edges) == 0:
                innovation_prob = 1
                perc_by_sub[sub_name][sub_name] += len(text_network.tgt_blobs[doc])/subreddit_counts[subreddit]
            else:
                local = local_edge_map[subreddit, edges]
                pair_pos = offsets[subreddit] + local
                logit_gamma_src = gammas[pair_pos]
                logit_gamma = np.concatenate([np.zeros(1), logit_gamma_src], axis=0)
                probs = softmax(logit_gamma)
                perc_by_sub[sub_name][sub_name] += probs[0]*len(text_network.tgt_blobs[doc])/subreddit_counts[subreddit]
                for edge, prob in zip(edges, probs[1:]):
                    src_sub = idx2src_sub[str(edge)]
                    perc_by_sub[sub_name][src_sub] += prob*len(text_network.tgt_blobs[doc]) /subreddit_counts[subreddit]
                innovation_prob = probs[0]
            avg_innovation_prob[subreddit] += innovation_prob*(len(text_network.tgt_blobs[doc]) /\
                    subreddit_counts[subreddit])

        expected_entries_mean = {"High": [], "Medium": [], "Low": []}
        for subreddit in avg_innovation_prob:
            #to-do use idx2tgt_sub to remap to subreddit name
            sub_name = idx2tgt_sub[str(subreddit)]
            expected = tgt2group[sub_name]
            expected_entries_mean[expected].append(avg_innovation_prob[subreddit])
            records.append({"% In-Group Language": avg_innovation_prob[subreddit],
                            "Subreddit": sub_name,
                            "Expected": expected,
                            "Iteration": iteration})
        for key, val in expected_entries_mean.items():
            expected_entries[key].append(np.mean(val))

for tgt_sub in perc_by_sub.keys():
    cur_percs = perc_by_sub[tgt_sub]
    print(tgt_sub)
    print(sorted(list(cur_percs.items()), key=lambda x: x[1], reverse=True)[:20])
    print("~~~~~~~~~~~~~~~~")

for key, val in expected_entries.items():
    print(key)
    print(az.hdi(np.array(val)))
    print(np.mean(np.array(val)))
    print("~~~~~~")
df = pd.DataFrame.from_records(records)
plot_ingroup(df, output_file)

