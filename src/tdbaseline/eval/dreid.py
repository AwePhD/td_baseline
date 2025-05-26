from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from tdbaseline.crop_features.from_detections import (
    import_features_detection_from_hdf5,
)

from ..data_struct import Detections
from ..pstr_output import import_detections_from_h5
from .common import build_sample
from .ious import compute_ious
from .metrics import compute_average_precision


def _get_representation_query(
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
    representation_query = detections_query.features_pstr[i_best_output]

    return representation_query


def eval_dreid(
    annotations_samples, frame_id_to_detections, threshold: float
) -> Tuple[float, float, int]:
    AP_sum = 0.0
    recall_sum = 0.0
    n_positive_detections_sum = 0
    for i, annotations_sample in tqdm(annotations_samples):
        frame_id_to_features_detections = {
            frame_id: detections.features_pstr
            for frame_id, detections in frame_id_to_detections.items()
        }
        annotations_query = annotations_sample.query(
            "split == 'query'"
        ).squeeze()
        representation_query = _get_representation_query(
            annotations_query,
            frame_id_to_detections[annotations_query.frame_id],
        )
        sample = build_sample(
            representation_query,
            annotations_sample,
            threshold,
            frame_id_to_detections,
            frame_id_to_features_detections,
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
    return mAP, mean_recall, n_positive_mean


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

    mAP, mean_recall, n_positive_mean = eval_dreid(
        annotations_samples, frame_id_to_detections, threshold
    )

    print(
        f"mAP: {mAP:.2%}, recall: {mean_recall:.2%}, n_pos: {n_positive_mean:,d}"
    )


def evaluate_dreid_clip_from_h5(
    annotations_file: Path,
    threshold: float,
    detections_file_h5: Path,
    automatic_crops_file_h5: Path,
) -> None:
    assert annotations_file.exists()
    assert detections_file_h5.exists()
    assert automatic_crops_file_h5.exists()

    annotations = pd.read_parquet(annotations_file).reset_index()
    annotations_samples = annotations.groupby("person_id")

    frame_id_to_detections = import_detections_from_h5(detections_file_h5)
    frame_id_to_features_clip = import_features_detection_from_hdf5(
        automatic_crops_file_h5
    )

    assert set(frame_id_to_detections.keys()) == set(
        frame_id_to_features_clip.keys()
    )
    frame_ids = sorted(frame_id_to_detections.keys())
    frame_id_to_detections_automatic: Dict[int, Detections] = {}
    for frame_id in frame_ids:
        detections = frame_id_to_detections[frame_id]
        features_clip = frame_id_to_features_clip[frame_id]
        frame_id_to_detections_automatic[frame_id] = Detections(
            scores=detections.scores,
            bboxes=detections.bboxes,
            features_pstr=features_clip,
        )

    mAP, mean_recall, n_positive_mean = eval_dreid(
        annotations_samples, frame_id_to_detections_automatic, threshold
    )

    print(
        f"mAP: {mAP:.2%}, recall: {mean_recall:.2%}, n_pos: {n_positive_mean:,d}"
    )
