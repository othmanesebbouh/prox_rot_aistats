import jax
from jax import numpy as jnp
from functools import partial
from tensorboardX import SummaryWriter

from ott.geometry.pointcloud import PointCloud
from ott.geometry.costs import CostFn
from initializers import init_M

from tools import run_sinkhorn

from gw import gromov_jit
from gw import update_geometry, update_geometry_lr

from fgw import update_geometry as update_geometry_fused
from fgw import gromov_jit_fused


class EntropicMap:

    def __init__(self):
        self.X = None
        self.Y = None
        self.X_tilde = None
        self.Y_tilde = None
        self.M = None
        self.alpha = None
        self.beta = None
        self.eps = None
        self.d_y = None
        self.geom_batch_size = None
        self.g_gw = None
        self.g_sink = None

    def fit(self, X, Y, eps, eta_fused=0, X_tilde=None, Y_tilde=None, X_val=None,
            Y_val=None, X_test=None, Y_test=None, use_validation=False,
            logdir=None, geom_batch_size=None, outer_iter=200, outer_threshold=1e-3,
            sinkhorn_iter=10000, sinkhorn_threshold=1e-3, compute_M_method="exact", init_M_method="random",
            init_M_seed=0, **kwargs):
        self.X = X
        self.Y = Y
        self.alpha = jnp.ones(X.shape[0]) / X.shape[0]
        self.beta = jnp.ones(Y.shape[0]) / Y.shape[0]
        self.eps = eps
        self.d_y = Y.shape[1]
        self.geom_batch_size = geom_batch_size
        init_dual_a = None
        init_dual_b = None

        if eta_fused > 0:

            self.M = init_M(init_M_method, X, Y, self.alpha, self.beta, "IP", seed=init_M_seed)
            geom_init = update_geometry_fused(X=X, Y=Y, X_tilde=X_tilde, Y_tilde=Y_tilde,
                                              M=self.M, geom_batch_size=self.geom_batch_size,
                                              geom_epsilon=self.eps,
                                              eta_fused=eta_fused)
        else:
            if compute_M_method == "nuclear" or compute_M_method == "nuclear_prox":
                self.M = init_M(init_M_method, X, Y, self.alpha, self.beta, "IP", seed=init_M_seed,
                                rank_M=kwargs["rank_M"])
                geom_init = update_geometry_lr(X=X, Y=Y, M=self.M, geom_batch_size=self.geom_batch_size,
                                               geom_epsilon=self.eps)
            else:
                self.M = init_M(init_M_method, X, Y, self.alpha, self.beta, "IP", seed=init_M_seed)
                geom_init = update_geometry(X=X, Y=Y,  M=self.M, geom_batch_size=self.geom_batch_size,
                                            geom_epsilon=self.eps)

        if logdir is None:
            writer = None
        else:
            writer = SummaryWriter(logdir)

        if eta_fused > 0:
            it, geom, costs, metrics, self.M, self.g_gw = gromov_jit_fused(self, geom_init=geom_init, M_init=self.M,
                                                                     X=X, Y=Y,
                                                                     X_tilde=X_tilde, Y_tilde=Y_tilde,
                                                                     use_validation=use_validation,
                                                                     X_val=X_val, Y_val=Y_val,
                                                                     X_test=X_test, Y_test=Y_test,
                                                                     alpha=self.alpha, beta=self.beta,
                                                                     eta_fused=eta_fused,
                                                                     compute_M_method=compute_M_method,
                                                                     init_dual_a=init_dual_a, init_dual_b=init_dual_b,
                                                                     outer_threshold=outer_threshold,
                                                                     outer_iter=outer_iter,
                                                                     sinkhorn_threshold=sinkhorn_threshold,
                                                                     sinkhorn_iter=sinkhorn_iter,
                                                                     writer=writer, logdir=logdir, **kwargs)
        else:
            it, geom, costs, metrics, self.M, self.g_gw = gromov_jit(self, geom_init=geom_init, M_init=self.M,
                                                                     X=X, Y=Y,
                                                                     use_validation=use_validation,
                                                                     X_val=X_val, Y_val=Y_val,
                                                                     X_test=X_test, Y_test=Y_test,
                                                                     alpha=self.alpha, beta=self.beta,
                                                                     compute_M_method=compute_M_method,
                                                                     init_dual_a=init_dual_a, init_dual_b=init_dual_b,
                                                                     outer_threshold=outer_threshold,
                                                                     outer_iter=outer_iter,
                                                                     sinkhorn_threshold=sinkhorn_threshold,
                                                                     sinkhorn_iter=sinkhorn_iter,
                                                                     writer=writer, logdir=logdir, **kwargs)
