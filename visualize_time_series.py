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
from poster_plots import plot_covid


start_date = sys.argv[1]
end_date = sys.argv[2]
input_dir = sys.argv[3]
results_dir = sys.argv[4]
output_file = sys.argv[5]

def date_iterator(start_date, end_date):
    start_split = start_date.split("_")
    end_split = end_date.split("_")
    start_year = int(start_split[0])
    start_month = int(start_split[1])
    end_year = int(end_split[0])
    end_month = int(end_split[1])
    for cur_year in range(start_year, end_year + 1):
        if cur_year == end_year:
            cur_end_month = end_month
        else:
            cur_end_month = 12
        if cur_year == start_year:
            cur_start_month = start_month
        else:
            cur_start_month = 1
        for cur_month in range(cur_start_month, cur_end_month + 1):
            yield "{}_{}".format(cur_year, cur_month)

records = []
time_step = 0

perc_by_sub = {}
total_counts = {}
tgt_subs =  ["China_Flu", "Coronavirus"]
for k in tgt_subs:
    perc_by_sub[k] = {}
    perc_by_sub[k][k] = 0
    total_counts[k] = 0

for date in date_iterator(start_date, end_date):
    input_dir_path = "{}/{}".format(input_dir, date)
    text_network = TextNetwork.load(input_dir_path)
    parameters_path = "{}/{}/parameters.npz".format(results_dir, date)
    if not os.path.exists(parameters_path):
        continue
    with open(input_dir_path + "/idx2tgt_sub.json") as f:
        idx2tgt_sub = json.load(f)
    with open(input_dir_path + "/idx2src_sub.json") as f:
        idx2src_sub = json.load(f)
    for src_sub in idx2src_sub.values():
        for k in tgt_subs:
            if src_sub not in perc_by_sub[k]:
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

        gamma_mean = parameters["gamma_mean"] 
        gamma_var = parameters["gamma_var"] 
        gamma_mean_src = parameters["gamma_mean_src"] 
        gamma_z = parameters["z_gamma"] 

        gamma_mean = gamma_mean.reshape(-1, gamma_mean.shape[2])
        gamma_var = gamma_var.flatten()
        gamma_mean_src = gamma_mean_src.reshape(-1, gamma_mean_src.shape[2])
        gamma_z = gamma_z.reshape(-1, gamma_z.shape[2])
        for iteration in range(0, gamma_mean.shape[0]):
            gammas = gamma_mean[iteration, pair_owner]+ gamma_mean_src[iteration, pair_srcs] +\
                     gamma_var[iteration]*gamma_z[iteration]
            avg_innovation_prob = {}
            counts = {}
            for doc in range(len(text_network.tgt_blobs)):
                subreddit = text_network.subreddits[doc]
                sub_name = idx2tgt_sub[str(subreddit)]
                if subreddit not in avg_innovation_prob:
                    avg_innovation_prob[subreddit] = 0
                    counts[subreddit] = 0
                edges = text_network.edges[doc]
                if len(edges) == 0:
                    innovation_prob = 1
                else:
                    local = local_edge_map[subreddit, edges]
                    pair_pos = offsets[subreddit] + local
                    logit_gamma_src = gammas[pair_pos]
                    logit_gamma = np.concatenate([np.zeros(1), logit_gamma_src], axis=0)
                    probs = softmax(logit_gamma)
                    for edge, prob in zip(edges, probs[1:]):
                        src_sub = idx2src_sub[str(edge)]
                        perc_by_sub[sub_name][src_sub] += prob*len(text_network.tgt_blobs[doc])
                    innovation_prob = probs[0]
                perc_by_sub[sub_name][sub_name] += innovation_prob*len(text_network.tgt_blobs[doc])
                avg_innovation_prob[subreddit] += innovation_prob*len(text_network.tgt_blobs[doc])
                counts[subreddit] += len(text_network.tgt_blobs[doc])
                total_counts[sub_name] += len(text_network.tgt_blobs[doc])
            for subreddit in avg_innovation_prob:
                #to-do use idx2tgt_sub to remap to subreddit name
                records.append({"% In-Group Language": avg_innovation_prob[subreddit] / counts[subreddit],
                                "Subreddit": idx2tgt_sub[str(subreddit)], 
                                "Iteration": iteration,
                                "Time": date.replace("_", "-")})

for tgt_sub in perc_by_sub.keys():
    cur_percs = perc_by_sub[tgt_sub]
    self_cite = cur_percs[tgt_sub] / total_counts[tgt_sub]
    normalized = {}
    sum_cur_percs = 0
    sum_normalized = 0
    for sub in cur_percs.keys():
        cur_percs[sub] = cur_percs[sub] / total_counts[tgt_sub]
        sum_cur_percs += cur_percs[sub]
        if sub != tgt_sub:
            normalized[sub] = cur_percs[sub] / (1-self_cite)
            sum_normalized += normalized[sub]
    print(tgt_sub)
    print("sum_cur_percs", sum_cur_percs)
    print("sum_normalized", sum_normalized)
    print(sorted(list(normalized.items()), key=lambda x: x[1], reverse=True)[:20])
    print("~~~~~~~~~~~~~~~~")


df = pd.DataFrame.from_records(records)
df['Date'] = pd.to_datetime(df['Time'])
plot_covid(df, output_file)

