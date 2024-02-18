from typing import Dict, Optional

import pandas as pd
from sklearn.metrics import f1_score
import scipy
import scipy.stats as st
import numpy as np
import jax.numpy as jnp


def _bary(
    *,
    Q: np.ndarray,
    g: np.ndarray,
    R: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
  return np.dot(Q / g, R.T @ data)


# def pearsonr(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
def pearsonr(pred, gt):
  res = []
  for i in range(pred.shape[1]):
    corr, _ = st.pearsonr(pred[:, i], gt[:, i])
    res.append(corr)
  return np.asarray(res)


def corr(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
  norm_pred = scipy.linalg.norm(pred, ord=2, axis=0, keepdims=True)
  norm_gt = scipy.linalg.norm(gt, ord=2, axis=0, keepdims=True)
  return np.sum((pred / norm_pred) * (gt / norm_gt), axis=0)


def rmse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
  return np.sqrt(np.mean((pred - gt) ** 2, axis=0))


def compute_metrics(
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    norm_rmse: bool = True,
) -> Dict[str, np.ndarray]:
  pred = np.nan_to_num(pred)

  pearson = pearsonr(pred, gt)
  corr_ = corr(pred, gt)
  err = rmse(st.zscore(pred), st.zscore(gt)) if norm_rmse else rmse(pred, gt)

  return {
      "pred": pred,
      "gt": gt,
      "pearson": pearson,
      "rmse": err,
      "corr": corr_,
  }


def compute_celltype_f1(
    *,
    celltypes: pd.Series,
    gt: pd.Series,
    Q: Optional[np.ndarray] = None,
    g: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    pred: Optional[np.ndarray] = None,
) -> Dict[Optional[str], float]:
  if pred is None:
    one_hot = pd.get_dummies(celltypes)
    one_hot = one_hot.values.astype(float)
    pred = _bary(Q=Q, g=g, R=R, data=one_hot / b)
    pred = np.argmax(pred, axis=1)

  pred = celltypes.cat.categories[pred].values

  res = {
      k: f1_score(y_true=gt, y_pred=pred, average=k)
      for k in [None, "micro", "macro", "weighted"]
  }
  res['pred'] = pred
  res['gt'] = gt
  return res


def compute_celltype_f1_simple(
    celltypes,
    gt,
    pred,
):
    pred = jnp.argmax(pred, axis=1)
    pred = celltypes.cat.categories[pred].values
    res = {
      k: f1_score(y_true=gt, y_pred=pred, average=k)
      for k in [None, "micro", "macro", "weighted"]
    }
    return res
