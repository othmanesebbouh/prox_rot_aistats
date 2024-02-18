import time
import jax
from functools import partial
from ott.geometry.pointcloud import PointCloud
from jax import numpy as jnp
from ott.solvers.linear import sinkhorn
from ott.utils import default_progress_fn
from jax.experimental import host_callback
from ott.geometry.costs import CostFn
from jax.lax import fori_loop
import numpy as np
from scores import foscttm_scores, foscttm_scores_lr
from jaxopt.prox import prox_lasso, prox_elastic_net
from prox import L21columns, NuclearNorm, AbsValue

import os
import pickle
from tools import run_sinkhorn, stop_gromov, stop_gromov_lr


def compute_M(X, Y, alpha, beta, geom_batch_size, eps, method="exact", solver_state=None,
              **kwargs):
    if method == "exact":
        print("Using exact")
        PY = solver_state.apply(Y.T, axis=1)
        M = PY.dot(X)
    elif method == "l1_reg":
        PY = solver_state.apply(Y.T, axis=1)
        M = PY.dot(X)
        M = prox_lasso(M, l1reg=kwargs["l1_reg"])
    elif method == "l12_reg":
        PY = solver_state.apply(Y.T, axis=1)
        M = PY.dot(X)
        M = L21columns().prox(M, gamma=kwargs["l12_reg"])
    elif method == "nuclear_prox":
        r = kwargs["rank_M"]
        PY = solver_state.apply(Y.T, axis=1)
        M = PY.dot(X)
        u, s, vh = jnp.linalg.svd(M, full_matrices=True, hermitian=False)
        M1 = jnp.diag(s[:r]).dot(vh[:r, :]).T
        M2 = u[:, :r]
        M = M1, M2
    elif method == "nuclear":
        M1, M2 = kwargs["M_init"]
        Y_tilde = Y.dot(M2)
        M_tilde_X = solver_state.apply(Y_tilde.T, axis=1).dot(X)
        M1 = jnp.linalg.pinv(M2.T.dot(M2)).dot(M_tilde_X).T
        X_tilde = X.dot(M1)
        M_tilde_Y = solver_state.apply(X_tilde.T, axis=0).dot(Y)
        M2 = jnp.linalg.pinv(M1.T.dot(M1)).dot(M_tilde_Y).T
        M = M1, M2
    else:
        raise ValueError("Method {} not implemented".format(method))
    return M


def compute_ot(X, Y, alpha, beta, geom_batch_size, eps, method="sinkhorn", **kwargs):
    f, g = kwargs["g_init"]
    new_geom = update_geometry(X=X, Y=Y, M=kwargs["M_init"], geom_epsilon=eps, geom_batch_size=geom_batch_size)
    solver_state = run_sinkhorn(new_geom, a=alpha, b=beta,
                                init_dual_a=f, init_dual_b=g,
                                inner_iter=min(int(kwargs["sinkhorn_iter"] / 10), 2000),
                                threshold=kwargs["sinkhorn_threshold"], n_iter=kwargs["sinkhorn_iter"])
    host_callback.id_tap(print_cost, solver_state.reg_ot_cost)
    return solver_state.f, solver_state.g


def gromov_it_params(X, Y, alpha, beta, compute_M_method="exact",
                     sinkhorn_threshold=1e-3, sinkhorn_iter=2000, writer=None, **kwargs):
    @jax.jit
    def gromov_fn_it(val):
        i, geom, costs, metrics, M, init_dual_a, init_dual_b = val
        solver_state = run_sinkhorn(geom, a=alpha, b=beta,
                                    init_dual_a=init_dual_a, init_dual_b=init_dual_b,
                                    inner_iter=int(sinkhorn_iter / 10),
                                    threshold=sinkhorn_threshold, n_iter=sinkhorn_iter)
        costs = costs.at[i].set(solver_state.reg_ot_cost)
        init_dual_a, init_dual_b = solver_state.f, solver_state.g
        if compute_M_method == "nuclear" or compute_M_method == "nuclear_prox":
            M = compute_M(M_init=M, X=X, Y=Y, alpha=alpha, beta=beta,
                          geom_batch_size=geom.batch_size, eps=geom.epsilon,
                          method=compute_M_method, solver_state=solver_state, it=i, **kwargs)
        else:
            M = compute_M(M_init=M, X=X, Y=Y, alpha=alpha, beta=beta,
                          geom_batch_size=geom.batch_size, eps=geom.epsilon,
                          method=compute_M_method, solver_state=solver_state, **kwargs)
        if kwargs["rank_M"] is None:
            new_geom = update_geometry(X=X, Y=Y, M=M, geom_epsilon=geom.epsilon,
                                       geom_batch_size=geom.batch_size)
        else:
            new_geom = update_geometry_lr(X=X, Y=Y, M=M, geom_epsilon=geom.epsilon,
                                          geom_batch_size=geom.batch_size)
        solver_state = run_sinkhorn(new_geom, a=alpha, b=beta,
                                    init_dual_a=init_dual_a, init_dual_b=init_dual_b,
                                    inner_iter=int(sinkhorn_iter / 10),
                                    threshold=sinkhorn_threshold, n_iter=sinkhorn_iter)
        init_dual_a, init_dual_b = solver_state.f, solver_state.g

        return i + 1, new_geom, costs, metrics, M, init_dual_a, init_dual_b

    return gromov_fn_it


def gromov_jit(entropic_map, geom_init, M_init, X, Y,
                     alpha, beta, logdir,
                     compute_M_method="exact",
                     init_dual_a=None, init_dual_b=None, outer_threshold=1e-3, outer_iter=20,
                     sinkhorn_threshold=1e-3, sinkhorn_iter=2000, writer=None, metrics_to_track=None,
                     monitor_sinkhorn_iter=None, local=False, **kwargs):

    if init_dual_a is None:
        init_dual_a = jnp.zeros(X.shape[0])
    if init_dual_b is None:
        init_dual_b = jnp.zeros(Y.shape[0])
    costs_init = - jnp.ones(outer_iter // kwargs["monitor_compute_every_outer_iter"] + 1)
    metrics = None
    
    if compute_M_method == "exact" or compute_M_method == "l1_reg" or compute_M_method == "l12_reg" or \
            compute_M_method == "nuclear" or compute_M_method == "nuclear_prox":
        gromov_fn = gromov_it_params(X=X, Y=Y,
                                     alpha=alpha, beta=beta,
                                     compute_M_method=compute_M_method,
                                     sinkhorn_threshold=sinkhorn_threshold, sinkhorn_iter=sinkhorn_iter,
                                     **kwargs)
        init_val = (jnp.array(0, dtype="int32"), geom_init, costs_init, metrics, M_init, init_dual_a, init_dual_b)
        if compute_M_method == "nuclear" or compute_M_method == "nuclear_prox":
            cond_fun = stop_gromov_lr(outer_threshold, outer_iter)
        else:
            cond_fun = stop_gromov(outer_threshold, outer_iter)
    else:
        raise ValueError("Method {} not implemented".format(compute_M_method))

    val = init_val

    i = 0
    idx = 0
    costs = costs_init
    scores = {}

    while i < outer_iter:
        if monitor_sinkhorn_iter is not None:
            i, geom, _, metrics, M, f, g = val
            cond_val = idx, geom, costs, metrics, M, f, g

            idx = i // kwargs["monitor_compute_every_outer_iter"]
            if i % kwargs["monitor_compute_every_outer_iter"] == 0:
                if kwargs["data_source"] == "spatial":
                    score_names = ["pearsonr_val", "pearsonr_test", "f1_macro",
                                   "f1_micro", "f1_weighted"]
                    if len(scores) == 0:
                        scores = {score_name: [] for score_name in score_names}

                    score_fn = spatial_scores
                    scores_dict = score_fn(X_train=X, Y_train=Y,
                                         X_val=kwargs["X_val"],
                                         Y_val=kwargs["Y_val"],
                                         X_test=kwargs["X_test"], Y_test=kwargs["Y_test"],
                                         labels=kwargs["labels"],
                                         M=M,
                                         g=g,
                                         beta=beta, eps=entropic_map.eps,
                                         batch_size=kwargs["val_batch_size"])
                    print("iteration: ", i)
                    print("M", M)
                    entropic_map.M = M
                    entropic_map.g = g
                    for score_name in score_names:
                        print(score_name, scores_dict[score_name])
                        scores[score_name].append(scores_dict[score_name])
                        if not local:
                            writer.add_scalar(score_name, np.array(scores[score_name][idx]), int(idx))
                    costs = costs.at[idx].set(scores_dict["pearsonr_val"])
                elif kwargs["data_source"] == "scGM":
                    score_names = ["lta", "nz"]
                    if len(scores) == 0:
                        scores = {score_name: [] for score_name in score_names}
                    score_fn = foscttm_scores

                    scores_dict = score_fn(X=X, Y=Y, M=M, g=g, beta=beta, eps=entropic_map.eps,
                                           X_val=kwargs["X_val"], Y_val=kwargs["Y_val"])

                    entropic_map.M = M
                    entropic_map.g = g
                    for score_name in score_names:
                        print(score_name, scores_dict[score_name])
                        scores[score_name].append(scores_dict[score_name])
                        writer.add_scalar(score_name, np.array(scores[score_name][idx]), int(idx))

                    costs = costs.at[idx].set(scores_dict["foscttm"])

                elif kwargs["data_source"] == "moscot":
                    score_names = ["foscttm"]
                    if len(scores) == 0:
                        scores = {score_name: [] for score_name in score_names}
                    if compute_M_method == "nuclear" or compute_M_method == "nuclear_prox":
                        score_fn = foscttm_scores_lr
                    else:
                        score_fn = foscttm_scores

                    scores_dict = score_fn(X=X, Y=Y, M=M, g=g, beta=beta, eps=entropic_map.eps,
                                           X_val=kwargs["X_val"], Y_val=kwargs["Y_val"])

                    entropic_map.M = M
                    entropic_map.g = g
                    for score_name in score_names:
                        print(score_name, scores_dict[score_name])
                        scores[score_name].append(scores_dict[score_name])
                        if not local:
                            writer.add_scalar(score_name, np.array(scores[score_name][idx]), int(idx))
                    costs = costs.at[idx].set(scores_dict["foscttm"])
                    print("M", M[0])
                    if compute_M_method == "l1_reg" or compute_M_method == "l12_reg":
                        print("nz={}".format((M == 0).sum() / (M.shape[0] * M.shape[1]) * 100))

                path_params = logdir + "/params"
                if not os.path.exists(path_params):
                    os.mkdir(path_params)
                if not os.path.exists(path_params + "/M"):
                    os.mkdir(path_params + "/M")
                if not os.path.exists(path_params + "/g"):
                    os.mkdir(path_params + "/g")
                if not os.path.exists(path_params + "/scores_data"):
                    os.mkdir(path_params + "/scores_data")

                if kwargs["save_params"]:
                    with open(os.path.join(path_params, 'M', 'M_{}.pkl'.format(i)), 'wb') as f:
                        pickle.dump(M, f)
                    with open(os.path.join(path_params, 'g', 'g_{}.pkl'.format(i)), 'wb') as f:
                        pickle.dump(g, f)

                if not cond_fun(cond_val):
                    break

            val = gromov_fn(val)

    i, geom, _, metrics, M, f, g = val

    return i, geom, costs, metrics, M, g


@partial(jax.jit, static_argnums=(4,))
def update_geometry(X, Y, M, geom_epsilon, geom_batch_size,):
    z_x = X.dot(M.T)
    z_y = Y
    cost_fn = GWCost_IP()
    return PointCloud(x=z_x, y=z_y, epsilon=geom_epsilon, batch_size=geom_batch_size, cost_fn=cost_fn)


@partial(jax.jit, static_argnums=(4,))
def update_geometry_lr(X, Y, M, geom_epsilon, geom_batch_size):
    M1, M2 = M
    z_x = X.dot(M1)
    z_y = Y.dot(M2)
    cost_fn = GWCost_IP()
    return PointCloud(x=z_x, y=z_y, epsilon=geom_epsilon, batch_size=geom_batch_size, cost_fn=cost_fn)

@jax.tree_util.register_pytree_node_class
class GWCost_IP(CostFn):
    def __init__(self):
        super().__init__()

    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return - jnp.vdot(x, y)