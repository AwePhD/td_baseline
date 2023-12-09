import numpy as np


def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    We compute the average precision over recall values i.e cut-off for each TPs.
    There is a penality if the co
    """
    # No TP -> AP = 0
    count_tp = labels.sum()
    if count_tp == 0:
        return count_tp

    indices_by_scores = scores.argsort()[::-1]
    labels_ranked = labels[indices_by_scores]

    tps = labels_ranked.cumsum(0)

    precisions = tps / np.arange(1, len(tps) + 1)
    precisions_at_delta_recall = precisions[labels_ranked]
    mean_average_precision = precisions_at_delta_recall.sum() / count_tp

    return mean_average_precision


def compute_average_precision_with_recall_penality(
    labels: np.ndarray,
    scores: np.ndarray,
    count_gt: int,
):
    count_tp = labels.sum()
    recall_rate_penality = count_tp / count_gt

    return compute_average_precision(labels, scores) * recall_rate_penality
