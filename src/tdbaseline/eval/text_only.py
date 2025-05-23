from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm

from ..crop_features.from_detections import import_features_detection_from_hdf5
from ..data_struct import CropIndex
from ..pstr_output import import_detections_from_h5
from ..text_features import import_features_text_from_h5
from .common import build_sample
from .metrics import compute_average_precision



def evaluate_text_only_from_h5(
    annotations_file: Path,
    threshold: float,
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
    recall_sum = 0.0
    n_positive_detections_sum = 0
    # 2*len(annotations_samples) is a bit hard to guess why, so we count the query manually
    n_queries = 0
    for person_id, annotations_sample in tqdm(annotations_samples):
        frame_id_query = (
            annotations_sample.query("split == 'query'").squeeze().frame_id
        )

        # for one crop_index there are two captions
        representation_queries = crop_index_to_features_text[
            CropIndex(cast(int, person_id), frame_id_query)
        ]
        for representation_query in representation_queries:
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
            recall_sum += recall_sample
            n_queries += 1
            AP_sum += recall_sample * compute_average_precision(
                sample.labels, sample.scores
            )
    mAP = AP_sum / n_queries
    mean_recall = recall_sum / n_queries
    n_positive_mean = int(n_positive_detections_sum / n_queries)
    print(
        f"mAP: {mAP:.2%}, recall: {mean_recall:.2%}, n_pos: {n_positive_mean:,d}"
    )
