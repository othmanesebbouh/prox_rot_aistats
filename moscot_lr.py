from entropic_map import EntropicMap
import itertools
from get_data import get_data
import datetime
import json
import jax
from scores import bary_pred, bary_pred_lr
import jax.numpy as jnp

import sys
from evals import calc_domainAveraged_FOSCTTM

moscot = get_data("moscot")
X_full, Y_full = moscot['X'], moscot["Y"]

subset_sizes = [25, 50, 100, 250, 500, 1000]
ranks = [5, 6, 7, 8, 9, 10, 11, 12]
epsilons = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

n_exp = len(subset_sizes) * len(ranks) * len(epsilons) * len(seeds)

geom_batch_size = None
outer_iter = 80
outer_threshold = 1e-5
sinkhorn_iter = 10000
sinkhorn_threshold = 1e-4
sto_batch_size = 300
step_M = 1
step_g = None
maxiter_M = 1
maxiter_g = 5
monitor_compute_every_iter = 1
monitor_sinkhorn_iter = 20000
monitor_sinkhorn_threshold = 1e-4
compute_M_method = "nuclear_prox"
init_M_method = "low_rank_id"
gw_cost_fn = "IP"
monitor_compute_every_outer_iter = 1
local = True

param_dict = {
    "subset_sizes": subset_sizes,
    "ranks": ranks,
    "epsilons": epsilons,
    "seeds": seeds,
    "method": compute_M_method,
    "outer_iter": outer_iter,
    "sinkhorn_iter": sinkhorn_iter,
    "sinkhorn_threshold": sinkhorn_threshold
}

scores = {}
path_scores = 'lr_exp_runs/scores_lr' + str(datetime.datetime.now())[:19] + '.json'
with open(path_scores, 'w') as fp:
    json.dump(scores, fp)

path_dict = 'lr_exp_runs/param_dict_lr' + str(datetime.datetime.now())[:19] + '.json'
with open(path_dict, 'w') as fp:
    json.dump(param_dict, fp)

ct = 0
if __name__ == "__main__":
    for subset_size in subset_sizes:
        print("Doing subset size {}".format(subset_size))
        scores[subset_size] = {}
        for seed in seeds:
            scores[subset_size][seed] = {}
            print("Doing seed {} of subset size {}".format(seed, subset_size))
            key = jax.random.PRNGKey(seed)
            idx = jax.random.choice(key, a=X_full.shape[0], shape=(subset_size,), replace=False)
            X = X_full[idx]
            Y = Y_full[idx]
            for rank_M in ranks:
                scores[subset_size][seed][rank_M] = {}
                for eps in epsilons:
                    print("Doing (eps, rank_M)={} of seed {} of subset size {}".format((eps, rank_M),
                                                                                       seed, subset_size))
                    print("Solving Gromov...")
                    pred_map = EntropicMap()
                    pred_map.fit(X=X, Y=Y, eps=eps, geom_batch_size=geom_batch_size, gw_cost_fn=gw_cost_fn,
                                 outer_iter=outer_iter, outer_threshold=outer_threshold,
                                sinkhorn_iter=sinkhorn_iter, sinkhorn_threshold=sinkhorn_threshold, rank=None,
                                 compute_M_method=compute_M_method,
                                 init_M_method=init_M_method,
                                sto_batch_size=sto_batch_size, step_M=step_M, maxiter_M=maxiter_M,
                                 step_g=step_g, maxiter_g=maxiter_g,
                                seed=1, monitor_compute_every_outer_iter=monitor_compute_every_outer_iter,
                                 monitor_sinkhorn_iter=monitor_sinkhorn_iter,
                                 monitor_sinkhorn_threshold=monitor_sinkhorn_threshold,
                                data_source="moscot", rank_M=rank_M, local=local)

                    Y_pred = bary_pred_lr(X=X, Y=Y, M=pred_map.M, g=pred_map.g, beta=jnp.ones(len(Y))/len(Y),
                                                 eps=eps, norm_X=None, norm_Y=None, cost_fn="IP")
                    score_subset = calc_domainAveraged_FOSCTTM(Y_pred, Y).mean()
                    Y_pred_full = bary_pred_lr(X=X_full, Y=Y, M=pred_map.M,
                                                g=pred_map.g, beta=jnp.ones(len(Y))/len(Y),
                                                 eps=eps, norm_X=None, norm_Y=None, cost_fn="IP")
                    score_full = calc_domainAveraged_FOSCTTM(Y_pred_full, Y_full).mean()
                    scores[subset_size][seed][rank_M][eps] = (score_subset.item(), score_full.item())

                    with open(path_scores, 'w') as fp:
                        json.dump(scores, fp)
                    print(scores)
                    print("Done with exp number {} on {} ".format(ct + 1, n_exp))
                    ct += 1