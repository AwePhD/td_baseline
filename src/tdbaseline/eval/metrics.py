import numpy as np
import torch
from torch import Tensor


def normalize(features_matrix: np.ndarray) -> np.ndarray:
    # [d] | [n, d] | other
    if features_matrix.ndim == 1:
        # [d] -> [n=1, d]
        features_matrix = features_matrix[None, :]
    # [n=1, d] | [n, d] | other
    if features_matrix.ndim != 2:
        raise ValueError(
            "needs 1-D or 2-D matrix for features_matrix.shape:",
            features_matrix.shape,
        )
    # [n=1, d] | [n, d]

    # [n]
    norms = np.sqrt(np.power(features_matrix, 2).sum(1))
    norms_stable = np.fmax(norms, np.finfo(norms.dtype).eps)

    return np.einsum("nd,n->nd", features_matrix, 1 / norms_stable)


def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """We compute the average precision over recall values i.e cut-off for each
    TPs.

    There is a penality if the co
    """
    # No TP -> AP = 0
    count_tp = labels.sum()
    if count_tp == 0:
        return count_tp

    indices_by_scores = scores.argsort()[::-1]
    labels_ranked = labels[indices_by_scores]

    tps = labels_ranked.cumsum()

    precisions = tps / np.arange(1, len(tps) + 1)
    # [n_tps]
    precisions_at_delta_recall = precisions[labels_ranked]

    return precisions_at_delta_recall.mean()


def compute_mean_average_precision(labels: Tensor, scores: Tensor) -> float:
    # (n_queries, n_galleries)
    indices_by_scores = scores.argsort(axis=1, descending=True)
    # (n_queries, n_galleries)
    labels_ranked = torch.gather(labels, dim=1, index=indices_by_scores)

    # (n_queries, n_galleries)
    tps = labels_ranked.cumsum(1)

    # (n_queries, n_galleries)
    precisions = tps / torch.arange(1, tps.size(1) + 1)
    # (n_queries)
    precisions_at_delta_recall = torch.einsum(
        "qg,qg->q", precisions, labels_ranked.float()
    )
    # (n_queries)
    average_precisions = torch.nan_to_num(  # nan because labels are 0 -> no TP detected for the query
        precisions_at_delta_recall / labels.sum(1), nan=0
    )

    return average_precisions.mean().item()
