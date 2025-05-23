from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data_struct import Detections
from ..ious import compute_ious
from ..metrics import compute_average_precision, normalize
from ..pstr_output import import_detections_from_h5


class Sample(NamedTuple):
    scores: np.ndarray
    labels: np.ndarray


class AnnotationsRow(NamedTuple):
    Index: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


def _get_representations_query(
    annotations_query: pd.Series,
    detections_query: Detections,
) -> np.ndarray:
    # (4,)
    gt_x, gt_y, gt_w, gt_h = annotations_query[
        ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    ].astype("Int32")
    bbox_gt_query = np.array([gt_x, gt_y, gt_x + gt_w, gt_y + gt_h])
    ious_query = compute_ious(detections_query.bboxes, bbox_gt_query)
    i_best_output = ious_query.argmax()

    # (d_PSTR)
    query_features = detections_query.features_pstr[i_best_output]

    return query_features


def _build_sample(
    annotations_sample: pd.DataFrame,
    threshold: float,
    frame_id_to_detections: Dict[int, Detections],
) -> Sample:
    annotations_query = annotations_sample.query("split == 'query'").squeeze()
    representations_query = _get_representations_query(
        annotations_query,
        frame_id_to_detections[annotations_query.frame_id],
    )

    similarities_sample: List[np.ndarray] = []
    labels_sample: List[np.ndarray] = []
    annotations_gallery = annotations_sample.query("split == 'gallery'")[
        ["frame_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    ].astype("Int32")
    for annotations_frame in annotations_gallery.itertuples(index=False):
        frame_id, gt_x, gt_y, gt_w, gt_h = AnnotationsRow(*annotations_frame)
        detections_frame = frame_id_to_detections[frame_id]

        # [n_outputs=100]
        detection_is_positive = detections_frame.scores >= threshold
        n_positive = detection_is_positive.sum()

        if n_positive == 0:
            continue

        # [n_positive, 4]
        bboxes_positive = detections_frame.bboxes[detection_is_positive]
        # [n_positive, d_PSTR]
        features_detection_positive = detections_frame.features_pstr[
            detection_is_positive
        ]

        # [1, n_positive] -squeeze()-> [n_positive]
        similarities_frame: np.ndarray = np.einsum(
            "nd,md->nm",
            normalize(representations_query),
            normalize(features_detection_positive),
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


def evaluate_dreid_from_h5(
    annotations_file: Path,
    threshold: float,
    detections_file_h5: Path,
) -> None:
    assert annotations_file.exists()
    assert detections_file_h5.exists()

    annotations = pd.read_parquet(annotations_file).reset_index()
    annotations_samples = annotations.groupby("person_id")

    frame_id_to_detections = import_detections_from_h5(detections_file_h5)

    AP_sum = 0.0
    recall_sum = 0.0
    n_positive_detections_sum = 0
    for i, annotations_sample in tqdm(annotations_samples):
        if i in [484, 1226, 1489, 3091, 4667, 10354]:
            continue
        sample = _build_sample(
            annotations_sample,
            threshold,
            frame_id_to_detections,
        )

        n_gt = annotations_sample.query("split == 'gallery'").notna().all(axis=1).sum()  # type: ignore
        n_positive_reid = sample.labels.sum()
        recall_sample = n_positive_reid / n_gt

        n_positive_detections_sum += len(sample.scores)
        recall_sum += recall_sample
        AP_sum += recall_sample * compute_average_precision(
            sample.labels, sample.scores
        )
    mAP = AP_sum / len(annotations_samples)
    mean_recall = recall_sum / len(annotations_samples)
    n_positive_mean = int(n_positive_detections_sum / len(annotations_samples))
    print(
        f"mAP: {mAP:.2%}, recall: {mean_recall:.2%}, n_pos: {n_positive_mean:,d}"
    )
