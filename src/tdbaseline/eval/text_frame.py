from pathlib import Path
from typing import Union, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from tdbaseline.eval.common import build_sample

from ..crop_features.from_detections import import_features_detection_from_hdf5
from ..data_struct import CropIndex, Detections
from ..pstr_output import import_detections_from_h5
from ..text_features import import_features_text_from_h5
from .ious import compute_ious
from .metrics import compute_average_precision, normalize


def _get_representation_queries(
    annotations_query: pd.Series,
    representation_text_query: np.ndarray,
    detections: Detections,
    crops_features: np.ndarray,
    threshold: float,
    alpha: float,
) -> Union[np.ndarray, None]:
    detection_is_positive = detections.scores >= threshold
    if not detection_is_positive.any():
        return None

    crops_features_positive = normalize(crops_features[detection_is_positive])
    # NOTE: representation_text_query is already normalized
    # [n_outputs=100, d_CLIP]
    crops_representations = (alpha * representation_text_query + (1 - alpha) * crops_features_positive)

    # [n_positive, 4]
    bboxes_positive = detections.bboxes[detection_is_positive]
    # (4,)
    gt_x, gt_y, gt_w, gt_h = annotations_query[
        ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    ].astype("Int32")
    threshold_iou = min(0.5, (gt_w * gt_h) / ((gt_w + 10) * (gt_h + 10)))
    bbox_gt_query = np.array([gt_x, gt_y, gt_x + gt_w, gt_y + gt_h])
    # [n_positive]
    ious = compute_ious(bboxes_positive, bbox_gt_query)
    detection_reid_is_positive = ious >= threshold_iou
    if not (detection_reid_is_positive).any():
        return None
    i_best_output = ious.argmax()
    
    query_representation = crops_representations[i_best_output]
    return query_representation


def evaluate_text_frame_from_h5(
    annotations_file: Path,
    threshold: float,
    alpha: float,
    crop_index_to_features_text_h5: Path,
    detections_file_h5: Path,
    automatic_crops_h5: Path,
) -> None:
    annotations = pd.read_parquet(annotations_file).reset_index()
    annotations_samples = annotations.groupby("person_id")

    crop_index_to_features_text = import_features_text_from_h5(
        crop_index_to_features_text_h5
    )
    frame_id_to_detections = import_detections_from_h5(detections_file_h5)
    frame_id_to_automatic_crops = import_features_detection_from_hdf5(
        automatic_crops_h5
    )

    AP_sum = 0.0
    n_positive_query = 0
    n_queries = 0
    recall_gallery_sum = 0.0
    n_positive_detections_sum = 0
    for person_id, annotations_sample in tqdm(annotations_samples):
        annotations_query = annotations_sample.query(
            "split == 'query'"
        ).squeeze()
        # [2, d_CLIP]
        representation_text_queries = normalize(
            crop_index_to_features_text[
                CropIndex(cast(int, person_id), annotations_query.frame_id)
            ]
        )
        n_queries += 2
        for representation_text_query in representation_text_queries:
            # (2, d_CLIP) IF not None
            representation_query = _get_representation_queries(
                annotations_query,
                representation_text_query,
                frame_id_to_detections[annotations_query.frame_id],
                frame_id_to_automatic_crops[annotations_query.frame_id],
                threshold,
                alpha,
            )

            if representation_query is None:
                # missing the query means:
                #   - 0 mAP (so do not change the mAP score)
                #   - Not a positive detection query
                #   - No impact on the gallery recall!
                continue

            sample = build_sample(
                representation_query,
                annotations_sample,
                threshold,
                frame_id_to_detections,
                frame_id_to_automatic_crops,
            )

            n_gt = annotations_sample.query("split == 'gallery'").notna().all(axis=1).sum()  # type: ignore
            n_positive_reid = sample.labels.sum()
            recall_sample = n_positive_reid / n_gt

            n_positive_detections_sum += len(sample.scores)
            n_positive_query += 1
            recall_gallery_sum += recall_sample
            AP_sum += recall_sample * compute_average_precision(
                sample.labels, sample.scores
            )
    mAP = AP_sum / n_queries
    mean_recall_query = n_positive_query / n_queries
    mean_recall_gallery = recall_gallery_sum / n_queries
    n_positive_mean = int(n_positive_detections_sum / n_queries)

    print(
        f"mAP: {mAP:.2%}, "
        f"recall (query): {mean_recall_query:.2%}, "
        f"recall (gallery): {mean_recall_gallery:.2%}, "
        f"n_pos: {n_positive_mean:,d}"
    )
