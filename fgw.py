import time
import jax
from functools import partial
from ott.geometry.pointcloud import PointCloud
from jax import numpy as jnp
from jaxopt import GradientDescent, LBFGS, NonlinearCG
from ott.solvers.linear import sinkhorn
from ott.utils import default_progress_fn
from jax.experimental import host_callback
from ott.problems.linear import linear_problem
from ott.geometry.costs import CostFn
from sklearn.decomposition import TruncatedSVD
from initializers import init_M
from jaxopt import GradientDescent, LBFGS
from jax.lax import fori_loop
import numpy as np
from scores_fused import spatial_scores

import os
import pickle
from tools import stop_gromov, sample_batches


def sd_ot_cost_ls():
    # @partial(jax.jit, static_argnums=(10,11))
    def sd_ot_cost(M, X, Y, X_tilde, Y_tilde, alpha, beta, eps, f, g, geom_batch_size, eta_fused):
        new_geom = update_geometry(X=X, Y=Y, X_tilde=X_tilde, Y_tilde=Y_tilde, M=M, geom_epsilon=eps,
                                   geom_batch_size=geom_batch_size,
                                   eta_fused=eta_fused)

        cost_right, sgn = new_geom.apply_lse_kernel(f=f, g=g, eps=eps, vec=beta, axis=1)
        ent_cost = cost_right * sgn
        cost = g.dot(beta) - ent_cost.dot(alpha)  # eps implicit in apply_lse_kernel, [1] is sign of lse
        # https://ott-jax.readthedocs.io/en/latest/_autosummary/ott.geometry.pointcloud.PointCloud.apply_lse_kernel.html
        return cost + 1 / 8 * (M ** 2).sum()
    return sd_ot_cost


# @partial(jax.jit, static_argnums=(12,))
def stochastic_ls_update_M_fn(X, Y, X_tilde, Y_tilde, alpha, beta, eps, g,
                                geom_batch_size, sto_batch_size, step, batch_ids,
                                eta_fused, rank_M=None):
    def stochastic_grad_update_M(i, M):
        batch_idx = batch_ids[i]
        X_b = X[batch_idx]
        f_b = jnp.zeros(sto_batch_size)
        X_tilde_b = X_tilde[batch_idx]
        alpha_b = jnp.ones(sto_batch_size) / sto_batch_size
        opt_fun = sd_ot_cost_ls()
        solver = GradientDescent(fun=opt_fun, maxiter=1)
        M = solver.run(init_params=M, X=X_b, Y=Y, X_tilde=X_tilde_b,
                   Y_tilde=Y_tilde,
                   alpha=alpha_b, beta=beta, f=f_b, g=g, geom_batch_size=geom_batch_size,
                   eps=eps, eta_fused=eta_fused).params
        return M

    return stochastic_grad_update_M


def ot_update_fn(X, Y, X_tilde, Y_tilde, alpha, beta, eps, M, geom_batch_size, sto_batch_size,
                 stepsize, batch_ids, eta_fused, rank_M=None):
    def update(i, dual_vars):
        f, g = dual_vars
        batches_X, batches_Y = batch_ids
        batch_idx = batches_X[i]
        X_b = X[batch_idx]
        X_tilde_b = X_tilde[batch_idx]
        geom_X_b = update_geometry(X=X_b, Y=Y, X_tilde=X_tilde_b, Y_tilde=Y_tilde, M=M, geom_epsilon=eps,
                                   geom_batch_size=geom_batch_size, eta_fused=eta_fused)
        f_b = - geom_X_b.apply_lse_kernel(f=jnp.zeros(sto_batch_size), g=g, vec=beta, axis=1, eps=eps)[0]
        f = f.at[batch_idx].set(f_b)

        batch_idx = batches_Y[i]
        Y_b = Y[batch_idx]
        Y_tilde_b = Y_tilde[batch_idx]
        geom_Y_b = update_geometry(X=X, Y=Y_b, X_tilde=X_tilde, Y_tilde=Y_tilde_b, M=M, geom_epsilon=eps,
                                   geom_batch_size=geom_batch_size,
                                  eta_fused=eta_fused)
        g_b = - geom_Y_b.apply_lse_kernel(f=f, g=jnp.zeros(sto_batch_size), vec=alpha, axis=0, eps=eps)[0]
        g = g.at[batch_idx].set(g_b)
        dual_vars = f, g
        return dual_vars

    return update


def compute_M(X, Y, X_tilde, Y_tilde, alpha, beta, geom_batch_size, eps, eta_fused, method="exact",
              solver_state=None, **kwargs):
    if method == "stochastic":
        init_val = kwargs["M_init"]
        body_fn = stochastic_ls_update_M_fn(X=X, Y=Y, X_tilde=X_tilde, Y_tilde=Y_tilde,
                                              alpha=alpha, beta=beta, eps=eps,
                                              eta_fused=eta_fused, g=kwargs["g_init"],
                                              geom_batch_size=geom_batch_size, sto_batch_size=kwargs["sto_batch_size"],
                                              step=kwargs["stepsize"], batch_ids=kwargs["batch_ids"])
        M = fori_loop(0, kwargs["maxiter"], body_fn, init_val)
    else:
        raise ValueError("Method {} not implemented".format(method))
    return M


def compute_ot(X, Y, X_tilde, Y_tilde, alpha, beta, geom_batch_size, eps, eta_fused,
               method="sinkhorn", **kwargs):
    if method == "stochastic":
        dual_vars = kwargs["g_init"]
        body_fn = ot_update_fn(X=X, Y=Y, X_tilde=X_tilde, Y_tilde=Y_tilde, alpha=alpha, beta=beta, eps=eps,
                               eta_fused=eta_fused,
                               M=kwargs["M_init"], geom_batch_size=geom_batch_size,
                               sto_batch_size=kwargs["sto_batch_size"], stepsize=eps * kwargs["step_g_eps_factor"],
                               batch_ids=kwargs["batch_ids"])
        f, g = fori_loop(0, kwargs["maxiter"], body_fn, dual_vars)
        return f, g


def gromov_it_params_sto(X, Y, X_tilde, Y_tilde, alpha, beta, sto_batch_size, eta_fused, step_M, maxiter_M,
                         step_g_eps_factor, maxiter_g,
                         compute_M_method="semi_dual_stochastic", rank_M=None,
                         sinkhorn_iter=None, sinkhorn_threshold=None):
    @jax.jit
    def gromov_fn_it(val):
        i, geom, costs, metrics, M, g, sto_key = val
        sto_key, _ = jax.random.split(sto_key)

        if compute_M_method == "stochastic":
            f, g = g
            sto_key, sto_key1 = jax.random.split(sto_key)
            batches_X = sample_batches(sto_key, X.shape[0], sto_batch_size, n_iter=maxiter_g)
            batches_Y = sample_batches(sto_key1, Y.shape[0], sto_batch_size, n_iter=maxiter_g)
            batch_ids = batches_X, batches_Y
            f, g = compute_ot(X=X, Y=Y, X_tilde=X_tilde, Y_tilde=Y_tilde, alpha=alpha, beta=beta,
                              eta_fused=eta_fused, geom_batch_size=geom.batch_size, sto_batch_size=sto_batch_size,
                              eps=geom.epsilon, method="stochastic", M_init=M, g_init=(f, g),
                              step_g_eps_factor=step_g_eps_factor, maxiter=maxiter_g,
                              batch_ids=batch_ids)
            batch_ids = jnp.expand_dims(batches_X[-1], 0)
            M_new = compute_M(M_init=M, X=X, Y=Y, X_tilde=X_tilde, Y_tilde=Y_tilde, alpha=alpha, beta=beta,
                          geom_batch_size=geom.batch_size, sto_batch_size=sto_batch_size,
                          eta_fused=eta_fused, eps=geom.epsilon, method=compute_M_method, g_init=g,
                          stepsize=step_M, maxiter=maxiter_M, batch_ids=batch_ids,
                          it=i, rank=rank_M)
            M += 1 / jnp.sqrt(i + 1) * (M_new - M) # momentum
            sto_key, sto_key1 = jax.random.split(sto_key)
            batches_X = sample_batches(sto_key, X.shape[0], sto_batch_size, n_iter=maxiter_g)
            batches_Y = sample_batches(sto_key1, Y.shape[0], sto_batch_size, n_iter=maxiter_g)
            batch_ids = batches_X, batches_Y
            f, g = compute_ot(X=X, Y=Y, X_tilde=X_tilde, Y_tilde=Y_tilde, alpha=alpha, beta=beta,
                              eta_fused=eta_fused, geom_batch_size=geom.batch_size, sto_batch_size=sto_batch_size,
                              eps=geom.epsilon, method="stochastic", M_init=M, g_init=(f, g),
                              step_g_eps_factor=step_g_eps_factor, maxiter=maxiter_g,
                              batch_ids=batch_ids)
            g = f, g
        else:
            raise NameError("instance of compute_M_method not implemented")
        return i + 1, geom, costs, metrics, M, g, sto_key

    return gromov_fn_it

# @partial(jax.jit, static_argnums=(0, 9,10,11,18))
def gromov_jit_fused(entropic_map, geom_init, M_init, X, Y, X_tilde, Y_tilde,
                     alpha, beta, eta_fused, logdir,
                     compute_M_method="exact",
                     init_dual_a=None, init_dual_b=None, outer_threshold=1e-3, outer_iter=20,
                     sinkhorn_threshold=1e-3, sinkhorn_iter=2000, writer=None, metrics_to_track=None,
                     monitor_sinkhorn_iter=None, **kwargs):
    if init_dual_a is None:
        init_dual_a = jnp.zeros(X.shape[0])
    if init_dual_b is None:
        init_dual_b = jnp.zeros(Y.shape[0])

    costs_init = - jnp.ones(outer_iter // kwargs["monitor_compute_every_outer_iter"] + 1)
    metrics = None
    if compute_M_method == "stochastic":
        sto_key = jax.random.PRNGKey(kwargs["seed"])
        sto_key1, _ = jax.random.split(sto_key)
        g_init = jax.random.normal(key=sto_key, shape=(Y.shape[0],))
        f_init = jax.random.normal(key=sto_key1, shape=(X.shape[0],))
        g_init = (f_init, g_init)
        geom_init.x = X.dot(M_init.T)
        init_val = (jnp.array(0, dtype="int32"), geom_init, costs_init, metrics, M_init, g_init, sto_key1)
        gromov_fn = gromov_it_params_sto(X=X, Y=Y, X_tilde=X_tilde, Y_tilde=Y_tilde, alpha=alpha, beta=beta,
                                         eta_fused=eta_fused,
                                         sto_batch_size=kwargs["sto_batch_size"],
                                         step_M=kwargs["step_M"], maxiter_M=kwargs["maxiter_M"],
                                         step_g_eps_factor=kwargs["step_g_eps_factor"], maxiter_g=kwargs["maxiter_g"],
                                         compute_M_method=compute_M_method)
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
            i, geom, _, metrics, M, g, sto_key = val
            f, g = g
            cond_val = idx, geom, costs, metrics, M, g, sto_key

            idx = i // kwargs["monitor_compute_every_outer_iter"]
            if i % kwargs["monitor_compute_every_outer_iter"] == 0:
                if kwargs["data_source"] == "spatial":
                    score_names = ["pearsonr_val", "pearsonr_test", "f1_macro",
                                   "f1_micro", "f1_weighted"]
                    if len(scores) == 0:
                        scores = {score_name: [] for score_name in score_names}
                    scores_dict = spatial_scores(X_train=X, X_tilde=X_tilde, Y_train=Y, Y_tilde=Y_tilde,
                                                 X_val=kwargs["X_val"],
                                                 Y_val=kwargs["Y_val"],
                                                 X_test=kwargs["X_test"], Y_test=kwargs["Y_test"],
                                                 labels=kwargs["labels"],
                                                 M=M,
                                                 g=g, beta=beta, eps=entropic_map.eps,
                                                 eta=eta_fused,
                                                 batch_size=kwargs["val_batch_size"])
                    print("iteration: ", i)
                    print(M)
                    entropic_map.M = M
                    entropic_map.g = g
                    for score_name in score_names:
                        print(score_name, scores_dict[score_name])
                        scores[score_name].append(scores_dict[score_name])
                        writer.add_scalar(score_name, np.array(scores[score_name][idx]), int(idx))
                    costs = costs.at[idx].set(scores_dict["pearsonr_val"])

                path_params = logdir + "/params"
                if not os.path.exists(path_params):
                    os.mkdir(path_params)
                if not os.path.exists(path_params + "/M"):
                    os.mkdir(path_params + "/M")
                if not os.path.exists(path_params + "/g"):
                    os.mkdir(path_params + "/g")

                if kwargs["save_params"]:
                    with open(os.path.join(path_params, 'M', 'M_{}.pkl'.format(i)), 'wb') as f:
                        pickle.dump(M, f)
                    with open(os.path.join(path_params, 'g', 'g_{}.pkl'.format(i)), 'wb') as f:
                        pickle.dump(g, f)

                if not cond_fun(cond_val):
                    break

            val = gromov_fn(val)

    if compute_M_method == "exact" or compute_M_method == "GD" or compute_M_method == "LBFGS":
        i, geom, _, metrics, M, f, g = val
    else:
        i, geom, _, metrics, M, g, sto_key = val

    return i, geom, costs, metrics, M, g


def update_geometry(X, Y, X_tilde, Y_tilde, M, geom_epsilon, geom_batch_size, eta_fused):
    z_x = (X.dot(M.T), X_tilde)
    z_y = (Y, Y_tilde)
    cost_fn = FGWCost_IP()
    cost_fn.eta_fused = eta_fused
    return PointCloud(x=z_x, y=z_y, epsilon=geom_epsilon, batch_size=geom_batch_size, cost_fn=cost_fn)


@jax.tree_util.register_pytree_node_class
class FGWCost_IP(CostFn):
    def __init__(self):
        super().__init__()
        self.eta_fused = 0.

    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        Mx, x_tilde = x
        y, y_tilde = y
        return (1 - self.eta_fused) * (- jnp.vdot(Mx, y)) + \
            self.eta_fused * (- jnp.vdot(x_tilde, y_tilde))