from entropic_map import EntropicMap
import itertools
from get_data import get_data
import datetime
import json
import jax
import ott
import jax.numpy as jnp
from scores import bary_pred, bary_pred_lr
from ott.geometry.pointcloud import PointCloud
from ott.geometry.costs import CostFn
from ott.utils import default_progress_fn
from ott.problems.linear.potentials import EntropicPotentials
from ott.solvers.linear import sinkhorn
from tools import IP

import sys
from evals import calc_domainAveraged_FOSCTTM

moscot = get_data("moscot")
X_full, Y_full = moscot['X'], moscot["Y"]

subset_sizes = [25, 50, 100, 250, 500, 1000, 6224]
epsilons = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

n_exp = len(subset_sizes) * len(epsilons) * len(seeds)

outer_iter = 400

param_dict = {
    "subset_sizes": subset_sizes,
    "epsilons": epsilons,
    "seeds": seeds,
    "method":"sinkhorn",
    "outer_iter": outer_iter
}

scores_sinkhorn = {}
path_scores = 'sinkhorn_exp_runs/scores_sinkhorn' + str(datetime.datetime.now())[:19] + '.json'
with open(path_scores, 'w') as fp:
    json.dump(scores_sinkhorn, fp)

path_dict = 'sinkhorn_exp_runs/param_dict_sinkhorn' + str(datetime.datetime.now())[:19] + '.json'
with open(path_dict, 'w') as fp:
    json.dump(param_dict, fp)

ct = 0
if __name__ == "__main__":
    for subset_size in subset_sizes:
        print("Doing subset size {}".format(subset_size))
        scores_sinkhorn[subset_size] = {}
        for seed in seeds:
            scores_sinkhorn[subset_size][seed] = {}
            print("Doing seed {} of subset size {}".format(seed, subset_size))
            key = jax.random.PRNGKey(seed)
            idx = jax.random.choice(key, X_full.shape[0], (subset_size,), replace=False)
            X = X_full[idx]
            Y = Y_full[idx]
            geom_xx = PointCloud(x=X, cost_fn=IP())
            geom_yy = PointCloud(x=Y, cost_fn=IP())
            for eps_sink in epsilons:
                print("Doing eps {} of seed {} of subset size {}".format(eps_sink, seed, subset_size))
                print("Solving Gromov...")
                gw_sol = ott.solvers.quadratic.gromov_wasserstein.solve(geom_xx, geom_yy, epsilon=eps_sink,
                                                                        max_iterations=outer_iter)
                print("Finished solving Gromov")
                sol = gw_sol.linear_state
                plan = sol.matrix
                M = sol.matrix.dot(X).T.dot(Y).T
                geom_xy = PointCloud(x=X.dot(M.T), y=Y, cost_fn=IP(), epsilon=eps_sink)
                sol_sinkhorn = sinkhorn.solve(geom_xy)
                Y_pred_sink_bary = bary_pred(X=X, Y=Y, M=M, g=sol_sinkhorn.g, beta=jnp.ones(len(Y)) / len(Y),
                                             eps=eps_sink, norm_X=None, norm_Y=None, cost_fn="IP")
                score_subset = calc_domainAveraged_FOSCTTM(Y_pred_sink_bary, Y).mean()
                Y_pred_sink_bary_full = bary_pred(X=X_full, Y=Y, M=M, g=sol_sinkhorn.g, beta=jnp.ones(len(Y)) / len(Y),
                                                  eps=eps_sink, norm_X=None, norm_Y=None, cost_fn="IP")
                score_full = calc_domainAveraged_FOSCTTM(Y_pred_sink_bary_full, Y_full).mean()

                scores_sinkhorn[subset_size][seed][eps_sink] = (score_subset.item(), score_full.item())

                print(scores_sinkhorn)

                with open(path_scores, 'w') as fp:
                    json.dump(scores_sinkhorn, fp)
                print(scores_sinkhorn)
                print("Done with exp number {} on {} ".format(ct + 1, n_exp))
                ct += 1