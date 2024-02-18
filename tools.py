import jax
from functools import partial
from jax import numpy as jnp
from ott.solvers.linear import sinkhorn
from ott.utils import default_progress_fn
from jax.experimental import host_callback
from ott.problems.linear import linear_problem
from ott.geometry.costs import CostFn


def sample(key, num_items, batch_size):
    return jax.random.choice(key, num_items, (batch_size,), replace=False)


def sample_batches(key, num_items, batch_size, n_iter):
    key_array = jax.random.split(key, n_iter)
    return jax.vmap(sample, in_axes=(0, None, None))(key_array, num_items, batch_size)


@jax.tree_util.register_pytree_node_class
class IP(CostFn):

    def __init__(self):
        super().__init__()

    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return - jnp.vdot(x, y)  # - (x ** 2).sum(-1) * (y ** 2).sum(-1)



@jax.tree_util.register_pytree_node_class
class MySinkhorn(sinkhorn.Sinkhorn):
    def __call__(
            self,
            ot_prob,
            init=(None, None),
            rng=None,
            init_dual_a=None,
            init_dual_b=None
    ):
        if init_dual_a is None:
            rng = jax.random.PRNGKey(0)
            initializer = self.create_initializer()
            init_dual_a, _ = initializer(
                ot_prob, *init, lse_mode=self.lse_mode, rng=rng
            )
        if init_dual_b is None:
            rng = jax.random.PRNGKey(0)
            initializer = self.create_initializer()
            _, init_dual_b = initializer(
                ot_prob, *init, lse_mode=self.lse_mode, rng=rng
            )

        return sinkhorn.run(ot_prob, self, (init_dual_a, init_dual_b))


@partial(jax.jit, static_argnums=(6, 7))
def run_sinkhorn(geom, a, b, init_dual_a, init_dual_b, threshold, inner_iter, n_iter):
    solver = MySinkhorn(threshold=threshold, max_iterations=n_iter,
                        inner_iterations=inner_iter,
                        progress_fn=default_progress_fn)
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    return solver(prob, init_dual_a=init_dual_a, init_dual_b=init_dual_b)


def _converged(i, costs, threshold):
    return jnp.logical_and(
        jnp.logical_and(
        i >= 2, jnp.isclose(costs[i - 2], costs[i - 1], rtol=threshold)
    ), jnp.isclose(costs[i - 1], costs[i], rtol=threshold)
    )


def _diverged(i, costs, M):
    finite_costs = jnp.logical_and(jnp.isfinite(costs[i - 1]), jnp.isfinite(costs[i]))
    finite_M = jnp.isfinite((M ** 2).sum())
    return jnp.logical_or(jnp.logical_not(finite_costs),
                          jnp.logical_not(finite_M))


def _diverged_lr(i, costs, M):
    finite_costs = jnp.logical_and(jnp.isfinite(costs[i - 1]), jnp.isfinite(costs[i]))
    finite_M = jnp.logical_and(jnp.isfinite((M[0] ** 2).sum()), jnp.isfinite((M[1] ** 2).sum()))
    return jnp.logical_or(jnp.logical_not(finite_costs),
                          jnp.logical_not(finite_M))


def _continue(i, costs, threshold, max_iter, M):
    return jnp.logical_or(
        jnp.logical_and(i <= 2, i <= max_iter),
        jnp.logical_and(
            jnp.logical_not(_diverged(i, costs, M)),
            jnp.logical_not(_converged(i, costs, threshold))
        )
    )


def _continue_lr(i, costs, threshold, max_iter, M):
    return jnp.logical_or(
        jnp.logical_and(i <= 2, i <= max_iter),
        jnp.logical_and(
            jnp.logical_not(_diverged_lr(i, costs, M)),
            jnp.logical_not(_converged(i, costs, threshold))
        )
    )


def stop_gromov(threshold, max_iter):
    def cond_fn(val):
        i, geom, costs, metrics, M, init_dual_a, init_dual_b = val
        return _continue(i, costs, threshold, max_iter, M)

    return cond_fn


def stop_gromov_lr(threshold, max_iter):
    def cond_fn(val):
        i, geom, costs, metrics, M, init_dual_a, init_dual_b = val
        return _continue_lr(i, costs, threshold, max_iter, M)

    return cond_fn


def _continue_sto(i, costs, threshold, max_iter):
    return jnp.logical_or(
        jnp.logical_and(i <= 2, i <= max_iter),
        jnp.logical_and(
            jnp.logical_not(_diverged(i, costs)),
            jnp.logical_not(_converged(i, costs, threshold))
        )
    )


def stop_gromov_sto(threshold, max_iter):
    def cond_fn(val):
        i, _, costs, _, _, _ = val
        return _continue(i, costs, threshold, max_iter)

    return cond_fn


def print_cost(arg, transform):
    ent_gw_cost = arg
    print("ent_GW_cost={}".format(ent_gw_cost))
    print("--------")


def print_nz(arg, transform):
    nz = arg
    print("nz={}".format(nz))
    print("--------")