from typing import Dict, List, NamedTuple

import numpy as np
import pandas as pd

from ..data_struct import Detections
from .ious import compute_ious
from .metrics import normalize


class Sample(NamedTuple):
    scores: np.ndarray
    labels: np.ndarray


class AnnotationsRow(NamedTuple):
    Index: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


def build_sample(
    representations_query: np.ndarray,
    annotations_sample: pd.DataFrame,
    threshold: float,
    frame_id_to_detections: Dict[int, Detections],
    frame_id_to_crops: Dict[int, np.ndarray],
) -> Sample:
    similarities_sample: List[np.ndarray] = []
    labels_sample: List[np.ndarray] = []
    annotations_gallery = annotations_sample.query("split == 'gallery'")[
        ["frame_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    ].astype("Int32")
    for annotations_frame in annotations_gallery.itertuples(index=False):
        frame_id, gt_x, gt_y, gt_w, gt_h = AnnotationsRow(*annotations_frame)
        scores = frame_id_to_detections[frame_id].scores
        bboxes = frame_id_to_detections[frame_id].bboxes
        features = frame_id_to_crops[frame_id]

        # [n_outputs=100]
        detection_is_positive = scores >= threshold
        n_positive = detection_is_positive.sum()

        if n_positive == 0:
            continue

        # [n_positive, 4]
        bboxes_positive = bboxes[detection_is_positive]
        # [n_positive, d_PSTR]
        features_positive = features[detection_is_positive]

        # [1, n_positive] -squeeze(0)-> [n_positive]
        similarities_frame: np.ndarray = np.einsum(
            "nd,md->nm",
            normalize(representations_query),
            normalize(features_positive),
        ).squeeze(0)
        labels_frame = np.zeros(n_positive, dtype=bool)

        # No search for ReID positive if no annotations
        if pd.isna(gt_x):
            similarities_sample.append(similarities_frame)
            labels_sample.append(labels_frame)
            continue

        threshold_iou = min(0.5, (gt_w * gt_h) / ((gt_w + 10) * (gt_h + 10)))
        bbox_gt_frame = np.array([gt_x, gt_y, gt_x + gt_w, gt_y + gt_h])
        # [n_positive]
        ious = compute_ious(bboxes_positive, bbox_gt_frame)
        if (ious < threshold_iou).all():
            similarities_sample.append(similarities_frame)
            labels_sample.append(labels_frame)
            continue

        # [n_positive]
        i_similarities_sorted = similarities_frame.argsort()[::-1]  # descend!
        for i_positive in i_similarities_sorted:
            if ious[i_positive] >= threshold_iou:
                labels_frame[i_positive] = True
                break

        similarities_sample.append(similarities_frame)
        labels_sample.append(labels_frame)

    return Sample(
        scores=np.concatenate(similarities_sample),
        labels=np.concatenate(labels_sample),
    )
