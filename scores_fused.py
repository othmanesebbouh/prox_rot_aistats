from metrics_spatial import pearsonr, compute_celltype_f1_simple
import jax.numpy as jnp
from functools import partial
import pandas as pd

import jax


@jax.jit
def f_eps(x, x_tilde, Y, Y_tilde, g, beta, eps, eta):
    a = (g[None, :] - ((1 - eta) * (- x.dot(Y.T)) + eta * (- x_tilde.dot(Y_tilde.T)))) / eps
    return - eps * jax.nn.logsumexp(a=a, b=beta)


def beta_tilde_fn(Y, Y_tilde, g, beta, eps, eta):
    @jax.jit
    def beta_tilde_sample(x, x_tilde):
        return - jax.grad(f_eps, argnums=4)(x, x_tilde, Y, Y_tilde, g, beta, eps, eta)
    return beta_tilde_sample


@jax.jit
def beta_tilde_vec(x, x_tilde, Y, Y_tilde, g, beta, eps, eta):
    fun = beta_tilde_fn(Y, Y_tilde, g, beta, eps, eta)
    return jax.jit(jax.vmap(fun))(x, x_tilde)


def _continue(i, max_iter):
    return i < max_iter


def continue_loop(max_iter):
    def cond_fn(val):
        i, genes_val_pred, genes_test_pred = val
        return _continue(i, max_iter)
    return cond_fn


def continue_loop_labels(max_iter):
    def cond_fn(val):
        i, labels_pred = val
        return _continue(i, max_iter)
    return cond_fn


def spatial_score_inner_loop(X_train, X_tilde, Y_train, Y_tilde, X_val, X_test, M, g, beta, eps, eta):
    def body_fun(val):
        i, genes_val_pred, genes_test_pred = val
        betas = beta_tilde_vec(X_train[i], X_tilde[i], Y_train.dot(M), Y_tilde, g, beta, eps, eta)
        new_pred_val = betas.dot(X_val)
        new_pred_test = betas.dot(X_test)
        genes_val_pred = genes_val_pred.at[i].set(new_pred_val)
        genes_test_pred = genes_test_pred.at[i].set(new_pred_test)
        i += 1
        val = i, genes_val_pred, genes_test_pred
        return val
    return body_fun


def spatial_score_inner_loop_labels(X_train, X_tilde, Y_train, Y_tilde, X_val, M, g, beta, eps, eta):
    def body_fun(val):
        i, labels_pred = val
        new_pred = beta_tilde_vec(X_train[i], X_tilde[i], Y_train.dot(M), Y_tilde, g, beta, eps, eta).dot(X_val)
        labels_pred = labels_pred.at[i].set(new_pred)
        i += 1
        val = i, labels_pred
        return val
    return body_fun


@partial(jax.jit, static_argnums=(11,))
def generate_preds(X_train, X_tilde, Y_train, Y_tilde, X_val, X_test, M, g, beta, eps, eta, batch_size):
    X_train1 = X_train[:(X_train.shape[0] // batch_size) * batch_size, :]
    X_train1 = X_train1.reshape((X_train1.shape[0] // batch_size, batch_size, X_train1.shape[1]))
    X_tilde1 = X_tilde[:(X_tilde.shape[0] // batch_size) * batch_size, :]
    X_tilde1 = X_tilde1.reshape((X_tilde1.shape[0] // batch_size, batch_size, X_tilde1.shape[1]))
    X_train2 = X_train[(X_train.shape[0] // batch_size) * batch_size:]
    X_tilde2 = X_tilde[(X_tilde.shape[0] // batch_size) * batch_size:]
    genes_val_pred = jnp.zeros(shape=(X_train1.shape[0], batch_size, X_val.shape[1]))
    genes_test_pred = jnp.zeros(shape=(X_train1.shape[0], batch_size, X_test.shape[1]))
    cond_fun = continue_loop(max_iter=X_train1.shape[0])
    body_fun = spatial_score_inner_loop(X_train1, X_tilde1, Y_train, Y_tilde, X_val, X_test,
                                        M, g, beta, eps, eta)
    i, genes_val_pred, genes_test_pred = jax.lax.while_loop(cond_fun=cond_fun,
                                                            body_fun=body_fun,
                                                            init_val=(0, genes_val_pred, genes_test_pred))
    genes_val_pred = genes_val_pred.reshape(X_train1.shape[0] * batch_size, X_val.shape[1])
    genes_test_pred = genes_test_pred.reshape(X_train1.shape[0] * batch_size, X_test.shape[1])
    betas = beta_tilde_vec(X_train2, X_tilde2, Y_train.dot(M), Y_tilde, g, beta, eps, eta)
    last_pred_val = betas.dot(X_val)
    last_pred_test = betas.dot(X_test)
    genes_val_pred = jnp.concatenate((genes_val_pred, last_pred_val))
    genes_test_pred = jnp.concatenate((genes_test_pred, last_pred_test))
    return genes_val_pred, genes_test_pred


@partial(jax.jit, static_argnums=(10,))
def generate_preds_labels(X_train, X_tilde, Y_train, Y_tilde, X_val, M, g, beta, eps, eta, batch_size):
    X_train1 = X_train[:(X_train.shape[0] // batch_size) * batch_size, :]
    X_train1 = X_train1.reshape((X_train1.shape[0] // batch_size, batch_size, X_train1.shape[1]))
    X_train2 = X_train[(X_train.shape[0] // batch_size) * batch_size:]
    X_tilde1 = X_tilde[:(X_tilde.shape[0] // batch_size) * batch_size, :]
    X_tilde1 = X_tilde1.reshape((X_tilde1.shape[0] // batch_size, batch_size, X_tilde1.shape[1]))
    X_tilde2 = X_tilde[(X_tilde.shape[0] // batch_size) * batch_size:]
    labels_pred = jnp.zeros(shape=(X_train1.shape[0], batch_size, X_val.shape[1]))
    cond_fun = continue_loop_labels(max_iter=X_train1.shape[0])
    body_fun = spatial_score_inner_loop_labels(X_train1, X_tilde1, Y_train, Y_tilde, X_val, M, g, beta, eps, eta)
    i, labels_pred = jax.lax.while_loop(cond_fun=cond_fun,
                                        body_fun=body_fun,
                                        init_val=(0, labels_pred))
    labels_pred = labels_pred.reshape(X_train1.shape[0] * batch_size, X_val.shape[1])
    last_pred = beta_tilde_vec(X_train2, X_tilde2, Y_train.dot(M), Y_tilde, g, beta, eps, eta).dot(X_val)
    labels_pred = jnp.concatenate((labels_pred, last_pred))
    return labels_pred


def spatial_scores(X_train, X_tilde, Y_train, Y_tilde, X_val, Y_val, X_test, Y_test, labels,
                   M, g, beta, eps, eta, batch_size):
    genes_val_pred, genes_test_pred = generate_preds(X_train=X_train, X_tilde=X_tilde, Y_train=Y_train, Y_tilde=Y_tilde,
                                                     X_val=X_val, X_test=X_test,
                                                     M=M, g=g, beta=beta, eps=eps, eta=eta,
                                                     batch_size=batch_size)

    genes_val_pred, genes_test_pred = jnp.nan_to_num(genes_val_pred), jnp.nan_to_num(genes_test_pred)
    print("Done computing preds")
    scores_dict = {"pearsonr_val": pearsonr(genes_val_pred, Y_val).mean(),
                   "pearsonr_test": pearsonr(genes_test_pred, Y_test).mean()}
    celltypes_to_pull, gt_celltypes = labels
    one_hot = pd.get_dummies(celltypes_to_pull)
    one_hot = one_hot.values.astype(float)

    labels_pred = generate_preds_labels(X_train=X_train, X_tilde=X_tilde, Y_train=Y_train, Y_tilde=Y_tilde,
                                        X_val=one_hot,
                                        M=M, g=g, beta=beta, eps=eps, eta=eta, batch_size=batch_size)

    f1_scores = compute_celltype_f1_simple(celltypes=celltypes_to_pull, gt=gt_celltypes,
                                           pred=labels_pred)
    scores_dict["f1_macro"], scores_dict["f1_micro"], scores_dict["f1_weighted"] = \
        f1_scores["macro"], f1_scores["micro"], f1_scores["weighted"]

    return scores_dict