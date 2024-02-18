import pathlib
import pickle
import warnings
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any, Literal, Union

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

__all__ = [
    "pca", "get_genes", "get_brain_data", "get_spatial_data"
]

_CONTRASTS = (
    'checkerboard', 'auditory&visual sentences', 'visual sentences',
    'horizontal vs vertical checkerboard',
    'auditory calculation vs auditory sentences',
    'visual processing vs checkerboard', 'vertical checkerboard',
    'auditory calculation', 'visual click vs visual sentences',
    'right auditory click', 'left visual click',
    'auditory&visual motor vs cognitive processing', 'visual processing',
    'auditory click vs auditory sentences',
    'visual processing vs auditory processing',
    'left auditory & visual click vs right auditory&visual click',
    'right auditory & visual click vs left auditory&visual click',
    'left auditory&visual click', 'right auditory & visual click',
    'cognitive processing vs motor', 'auditory&visual calculation',
    'visual sentences vs checkerboard',
    'auditory processing vs visual processing', 'visual calculation',
    'auditory processing', 'auditory sentences',
    'vertical vs horizontal checkerboard',
    'auditory&visual calculation vs sentences',
    'visual calculation vs sentences', 'left auditory click',
    'horizontal checkerboard', 'right visual click'
)

# trn_features, (geom_x, geom_y), val_features, tst_features
Data_t = Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray,
               np.ndarray]


def _bary(
    *,
    Q: np.ndarray,
    g: np.ndarray,
    R: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
  return np.dot(Q / g, R.T @ data)


def pca(
    adata: ad.AnnData, *, trn_genes: List[str], val_genes: List[str],
    tst_genes: List[str]
) -> Data_t:

  def densify(x: sp.spmatrix) -> np.ndarray:
    return x.A if sp.issparse(x) else x

  adata_trn = adata[:, trn_genes]
  adata_val = adata[:, val_genes]
  adata_tst = adata[:, tst_genes]

  sc.pp.pca(adata_trn, n_comps=30, random_state=0)

  X_train = np.asarray(adata_trn.obsm["X_pca"])
  # coordinates already standardized
  S_train = np.asarray(adata_trn.obsm["spatial"])
  X_val = densify(adata_val.X)
  X_test = densify(adata_tst.X)

  return X_train, (S_train, S_train), X_val, X_test


def get_genes(
    adata_sa1: ad.AnnData,
    adata_sa3: ad.AnnData,
    *,
    cluster: Optional[str],
    val_tst: Union[Tuple[int, int], Tuple[float, float]] = (5, 5),
    seed: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
  np.testing.assert_array_equal(adata_sa1.var_names, adata_sa3.var_names)

  def to_size(n: Union[int, float]) -> int:
    assert isinstance(n, float)
    return int(n * adata_sa1.n_vars)

  def split(genes: List[str]) -> Tuple[List[str], List[str]]:
    val, tst = [], []
    for i in range(len(genes)):
      gene = genes[i]
      which = i % 3
      if which == 0:
        tst.append(gene)
      elif which == 1:
        val.append(gene)
      elif which == 2:
        pass
      else:
        raise NotImplementedError(which)
    return val[:n_val], tst[:n_tst]

  if cluster is None:
    n_val, n_tst = map(to_size, val_tst)
    rng = np.random.RandomState(seed)
    genes = tuple(rng.permutation(adata_sa1.var_names))
    val = genes[:n_val]
    tst = genes[n_val:n_val + n_tst]
    trn = genes[n_val + n_tst:]
  else:
    n_val, n_tst = val_tst
    assert isinstance(n_val, int), n_val
    assert isinstance(n_tst, int), n_tst

    genes_1 = sc.get.rank_genes_groups_df(adata_sa1, group=cluster)["names"]
    genes_2 = sc.get.rank_genes_groups_df(adata_sa3, group=cluster)["names"]

    val1, tst1 = split(genes_1)
    val2, tst2 = split(genes_2)

    tst = list(set(tst1) | set(tst2))[:n_tst]
    val = list(((set(val1) | set(val2)) - set(tst)))[:n_val]
    trn = list(set(adata_sa1.var_names) - (set(val) | set(tst)))

  assert len(set(trn + val + tst)) == len(trn + val + tst)
  assert len(trn + val + tst) == adata_sa1.n_vars

  return trn, val, tst


def get_brain_data(
    mesh: Optional[Literal[3, 4, 5, 6, 7]],
    *,
    surf: str = "pial_left",
    embedding_path: Optional[Union[str, pathlib.Path]] = None,
    val_tst: Tuple[int, int] = (5, 5),
    n_landmarks: int = 300,
    k: int = 3,
    n_jobs: int = 4,
) -> Tuple[Data_t, Data_t, Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
  from nilearn import surface, datasets, image
  from fugw.scripts import lmds

  def load_imgs(paths) -> np.ndarray:
    images = [image.load_img(img) for img in paths]
    return np.stack([
        np.nan_to_num(surface.vol_to_surf(img, surf)) for img in images
    ],
                    axis=1)

  n_val, n_tst = val_tst
  val_contrasts = _CONTRASTS[:n_val]
  tst_contrasts = _CONTRASTS[n_val:n_val + n_tst]
  trn_contrasts = tuple(
      c for c in _CONTRASTS if c not in val_contrasts and c not in tst_contrasts
  )
  n_trn = len(trn_contrasts)
  all_contrasts = list(trn_contrasts + val_contrasts + tst_contrasts)
  n_contrasts = len(all_contrasts)

  assert n_contrasts == len(_CONTRASTS)
  assert len(set(trn_contrasts + val_contrasts + tst_contrasts)) == n_contrasts

  mesh = "fsaverage" if mesh is None else f"fsaverage{mesh}"
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    brain_data = datasets.fetch_localizer_contrasts(
        all_contrasts,
        data_dir="./data",
        n_subjects=2,  # first 2 subjects
        get_anats=True,
    )
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    surf = getattr(fsaverage, surf)

    src_features = load_imgs(brain_data["cmaps"][:n_contrasts])
    tgt_features = load_imgs(brain_data["cmaps"][n_contrasts:])

  # coordinates/triangles used for sparse solver
  coordinates, triangles = surface.load_surf_mesh(surf)
  if embedding_path is not None:
    assert mesh == "fsaverage", \
      "Precomputed distances are only available for the full mesh."
    assert "lmds" in embedding_path, embedding_path
    with open(embedding_path, "rb") as fin:
      emb_x = pickle.load(fin)
      if isinstance(emb_x, tuple):
        emb_x, emb_y = emb_x
      else:
        emb_y = np.array(emb_x, copy=True)
  else:
    emb_x = lmds.compute_lmds(
        coordinates,
        triangles,
        n_landmarks=n_landmarks,
        k=k,
        n_jobs=n_jobs,
        verbose=False,
    ).cpu().numpy()
    emb_y = np.array(emb_x, copy=True)

  geom_src = (emb_x, emb_y)
  geom_tgt = (np.array(emb_x, copy=True), np.array(emb_y, copy=True))

  trn_src = src_features[:, :n_trn]
  val_src = src_features[:, n_trn:n_trn + n_val]
  tst_src = src_features[:, n_trn + n_val:]

  trn_tgt = tgt_features[:, :n_trn]
  val_tgt = tgt_features[:, n_trn:n_trn + n_val]
  tst_tgt = tgt_features[:, n_trn + n_val:]

  return (trn_src, geom_src, val_src,
          tst_src), (trn_tgt, geom_tgt, val_tgt,
                     tst_tgt), (coordinates, triangles), {
                         "trn_contrasts": trn_contrasts,
                         "val_contrasts": val_contrasts,
                         "tst_contrasts": tst_contrasts
                     }


def get_spatial_data(
    path1: Union[str, pathlib.Path], path2: Union[str, pathlib.Path], *,
    trn_genes_path: Union[str, pathlib.Path],
    val_genes_path: Union[str,
                          pathlib.Path], tst_genes_path: Union[str,
                                                               pathlib.Path]
) -> Tuple[Data_t, Data_t]:
  adata_sa1 = ad.read(path1)
  adata_sa3 = ad.read(path2)
  np.testing.assert_array_equal(adata_sa1.var_names, adata_sa3.var_names)

  trn_genes = pd.read_csv(trn_genes_path, index_col=0)['0'].tolist()
  val_genes = pd.read_csv(val_genes_path, index_col=0)['0'].tolist()
  tst_genes = pd.read_csv(tst_genes_path, index_col=0)['0'].tolist()

  data_1 = pca(
      adata_sa1, trn_genes=trn_genes, val_genes=val_genes, tst_genes=tst_genes
  )
  data_3 = pca(
      adata_sa3, trn_genes=trn_genes, val_genes=val_genes, tst_genes=tst_genes
  )
  return data_1, data_3
