from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch

from tdbaseline.crop_features.from_dataset import (
    import_features_crop_from_hdf5,
)
from tdbaseline.eval.ious import compute_ious

from ..crop_features.from_detections import import_features_detection_from_hdf5
from ..pstr_output import import_detections_from_h5
from ..text_features import import_features_text_from_h5
from .metrics import compute_mean_average_precision, normalize


def _validate_crop(
    annotations_reid_gallery,
    crop,
    frame_id_to_detections,
    threshold: float,
) -> Union[int, None]:
    person_id, frame_id = crop
    annotations_frame = annotations_reid_gallery.query(
        f"person_id == {person_id} & frame_id == {frame_id}"
    ).squeeze()
    if annotations_frame.empty:
        # the crop belongs to a query only
        return None

    detections = frame_id_to_detections[frame_id]

    is_positive = detections.scores > threshold
    if not is_positive.any():
        return None

    gt_x, gt_y, gt_w, gt_h = annotations_frame[
        ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    ].astype("Int32")
    threshold_iou = min(0.5, (gt_w * gt_h) / ((gt_w + 10) * (gt_h + 10)))
    bbox_gt_query = np.array([gt_x, gt_y, gt_x + gt_w, gt_y + gt_h])
    bboxes_positive = detections.bboxes[is_positive]
    ious = compute_ious(bboxes_positive, bbox_gt_query)
    i_best_detection = ious.argmax()
    if ious[i_best_detection] < threshold_iou:
        return None

    i_best_detection_from_base_index = np.arange(len(detections.scores))[
        is_positive
    ][i_best_detection]
    return i_best_detection_from_base_index


def evaluate_text_only_tp_only_from_h5(
    annotations_file: Path,
    threshold: float,
    crop_index_to_features_text_h5: Path,
    detections_file_h5: Path,
    automatic_crops_h5: Path,
) -> None:
    annotations = pd.read_parquet(annotations_file).reset_index()

    crop_index_to_features_text = import_features_text_from_h5(
        crop_index_to_features_text_h5
    )
    frame_id_to_detections = import_detections_from_h5(detections_file_h5)
    frame_id_to_automatic_crops = import_features_detection_from_hdf5(
        automatic_crops_h5
    )

    # [n_crops]
    crops = list(crop_index_to_features_text.keys())

    # [2*n_crops, d_CLIP]
    representations_query = np.concatenate(
        [crop_index_to_features_text[crop] for crop in crops]
    )

    person_ids_gallery_all: List[int] = []
    representations_gallery_all: List[np.ndarray] = []
    annotations_reid_gallery = annotations.query("split == 'gallery'").dropna()
    for crop in crops:
        i_output_or_none = _validate_crop(
            annotations_reid_gallery,
            crop,
            frame_id_to_detections,
            threshold,
        )
        if i_output_or_none is None:
            continue

        i_output = i_output_or_none
        representations_gallery = frame_id_to_automatic_crops[crop.frame_id][
            i_output
        ]
        representations_gallery_all.append(representations_gallery)
        person_ids_gallery_all.append(crop.person_id)

    # [n_gallery_TP, d_CLIP]
    representations_gallery = np.array(representations_gallery_all)
    # [n_gallery_TP]
    person_ids_gallery = np.array(person_ids_gallery_all)
    # [2*n_crops, n_gallery_TP]
    similarities = np.einsum(
        "qd,gd->qg",
        normalize(representations_query),
        normalize(representations_gallery),
    )

    # [2*n_crops = n_captions]
    person_ids_query = np.array(
        [crop.person_id for crop in crops for _ in range(2)]
    )
    # [2*n_crops, n_gallery_TP]
    labels = person_ids_query[:, None] == person_ids_gallery[None, :]

    mAP = compute_mean_average_precision(
        torch.tensor(labels), torch.tensor(similarities)
    )
    n_tps = len(person_ids_gallery)
    recall = n_tps / len(crops)

    print(f"mAP: {mAP:.2%}, Gallery size: {n_tps:,d}, Recall: {recall:.2%}")


def evaluate_treid_tp_only_from_h5(
    annotations_file: Path,
    threshold: float,
    crop_index_to_features_text_h5: Path,
    detections_file_h5: Path,
    crops_features_file_h5: Path,
) -> None:
    annotations = pd.read_parquet(annotations_file).reset_index()

    crop_index_to_features_text = import_features_text_from_h5(
        crop_index_to_features_text_h5
    )
    frame_id_to_detections = import_detections_from_h5(detections_file_h5)
    crop_index_to_features_crops = import_features_crop_from_hdf5(
        crops_features_file_h5
    )

    # [n_crops]
    crops = list(crop_index_to_features_text.keys())

    # [2*n_crops, d_CLIP]
    representations_query = np.concatenate(
        [crop_index_to_features_text[crop] for crop in crops]
    )

    person_ids_gallery_all: List[int] = []
    representations_gallery_all: List[np.ndarray] = []
    annotations_reid_gallery = annotations.query("split == 'gallery'").dropna()
    for crop in crops:
        i_output_or_none = _validate_crop(
            annotations_reid_gallery,
            crop,
            frame_id_to_detections,
            threshold,
        )
        if i_output_or_none is None:
            continue
        person_ids_gallery_all.append(crop.person_id)
        representations_gallery_all.append(crop_index_to_features_crops[crop])

    # [n_gallery_TP, d_CLIP]
    representations_gallery = np.array(representations_gallery_all)
    # [n_gallery_TP]
    person_ids_gallery = np.array(person_ids_gallery_all)
    # [2*n_crops, n_gallery_TP]
    similarities = np.einsum(
        "qd,gd->qg",
        normalize(representations_query),
        normalize(representations_gallery),
    )

    # [2*n_crops = n_captions]
    person_ids_query = np.array(
        [crop.person_id for crop in crops for _ in range(2)]
    )
    # [2*n_crops, n_gallery_TP]
    labels = person_ids_query[:, None] == person_ids_gallery[None, :]

    mAP = compute_mean_average_precision(
        torch.tensor(labels), torch.tensor(similarities)
    )
    n_tps = len(person_ids_gallery)
    recall = n_tps / len(crops)

    print(f"mAP: {mAP:.2%}, Gallery size: {n_tps:,d}, Recall: {recall:.2%}")
