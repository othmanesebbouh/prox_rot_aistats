import jax
import jax.numpy as jnp
import numpy as  np
from scipy.stats import special_ortho_group
import pickle
import jax
from scores import spatial_scores


def get_data(data_source, X_dim=None, Y_dim=None, n_samples=None, seed=None, use_validation=False,
             center_X=False, center_X_tilde=False, **kwargs):
    data_dict = {}
    if data_source == "spatial":
        with open('data/spatial/data.pkl', 'rb') as f:
            data_src, data_tgt, celltypes_to_pull, gt_celltypes = pickle.load(f)
        if center_X:
            data_dict["X"] = jnp.array(data_tgt[1][0]) - jnp.array(data_tgt[1][0]).mean(axis=0)
            data_dict["Y"] = jnp.array(data_src[1][0]) - jnp.array(data_src[1][0]).mean(axis=0)
        else:
            data_dict["X"] = jnp.array(data_tgt[1][0])
            data_dict["Y"] = jnp.array(data_src[1][0])
        if center_X_tilde:
            data_dict["X_tilde"] = jnp.array(data_tgt[0]) - jnp.array(data_tgt[0]).mean(axis=0)
            data_dict["Y_tilde"] = jnp.array(data_src[0]) - jnp.array(data_src[0]).mean(axis=0)
        else:
            data_dict["X_tilde"] = jnp.array(data_tgt[0])
            data_dict["Y_tilde"] = jnp.array(data_src[0])

        data_dict["X_val"] = data_src[2]
        data_dict["Y_val"] = data_tgt[2]

        data_dict["X_test"] = data_src[3]
        data_dict["Y_test"] = data_tgt[3]

        data_dict["labels"] = celltypes_to_pull, gt_celltypes

        data_dict["val_fct"] = spatial_scores
        data_dict["scores_names"] = ("pearsonr_val", "pearsonr_test")

    elif data_source == "moscot":
        X = jnp.array(np.load("data/moscot/X_moscot.npy"))
        Y = jnp.array(np.load("data/moscot/Y_moscot.npy"))
        X /= ((X ** 2).sum(-1) ** 0.5)[:, None]
        Y /= ((Y ** 2).sum(-1) ** 0.5)[:, None]
        if center_X:
            X -= X.mean(axis=0)
            Y -= Y.mean(axis=0)

        data_dict["X"] = X
        data_dict["Y"] = Y
        data_dict["X_test"], data_dict["Y_test"], data_dict["labels"] = None, None, None

    elif data_source == "scGM":
        X = jnp.array(np.load("data/scGM/X_scGM.npy"))
        Y = jnp.array(np.load("data/scGM/Y_scGM.npy"))
        if center_X:
            data_dict["X"] = X - X.mean(axis=0)
            data_dict["Y"] = Y - Y.mean(axis=0)
        else:
            data_dict["X"] = X
            data_dict["Y"] = Y
        data_dict["X_val"] = np.loadtxt("data/scGM/scGM_typeExpression.txt")
        data_dict["Y_val"] = np.loadtxt("data/scGM/scGM_typeMethylation.txt")
        data_dict["X_test"], data_dict["Y_test"], data_dict["labels"] = None, None, None
    else:
        raise NameError("data source {} not implemented".format(data_source))

    return data_dict
