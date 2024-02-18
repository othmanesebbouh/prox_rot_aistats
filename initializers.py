import jax
import jax.numpy as jnp
import ott

from functools import partial

def compute_M_opt_lr(X, Y, a_init, b_init):
    return 4 * jnp.outer(Y.mean(axis=0), X.mean(axis=0))


def init_M(init_method, X, Y, alpha, beta, cost_type, seed=0, **kwargs):
    if init_method == "rank1_random":
        rng = jax.random.PRNGKey(seed)
        a_init = alpha + jax.random.uniform(rng, shape=(len(alpha),))
        a_init /= a_init.sum()
        b_init = beta - jax.random.uniform(rng, shape=(len(beta),))
        b_init /= b_init.sum()
        M_init = compute_M_opt_lr(X, Y, a_init, b_init)
    elif init_method == "low_rank_id":
        id = jnp.eye(kwargs["rank_M"]) / jnp.sqrt(kwargs["rank_M"])
        zeros = jnp.zeros(shape=(X.shape[1] - kwargs["rank_M"], kwargs["rank_M"]))
        M1 = jnp.concatenate((id, zeros), axis=0)
        zeros = jnp.zeros(shape=(Y.shape[1] - kwargs["rank_M"], kwargs["rank_M"]))
        M2 = jnp.concatenate((id, zeros), axis=0)
        M_init = (M1, M2)
    elif init_method == "ones":
        return jnp.ones((Y.shape[1], X.shape[1])) / jnp.maximum(Y.shape[1], X.shape[1])
    else:
        raise NameError("Initialization method {} not supported".format(init_method))
    return M_init
