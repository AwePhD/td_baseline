"""Evaluate the baseline performance in Text ReID framework.

- Evaluate model from ground_truth (no detection involved):
    - crops features from files (same as Text ReID on PEDES)
    - crops features from SYSU annotations
- Evaluate model from detections
"""
from typing import List
from pathlib import Path

import numpy as np

from ..metrics import compute_average_precision
from ..data_struct import CropIndex
from ..captions_features import import_captions_output_from_hdf5
from .crop_features_from_dataset import import_crop_features_from_hdf5


def _get_labels_from_crop_indexes(crop_indexes: List[CropIndex]) -> np.ndarray:
    # [1, n_gallery]
    person_ids = np.array(crop_indexes).reshape(1, -1)

    # [n_gallery, n_gallery]
    # [i, j] true if person_ids[i] == person_ids[j]
    labels = person_ids == person_ids.T

    # [n_query, n_gallery]
    return np.repeat(labels, repeats=2, axis=0)


def evaluate_from_ground_truth(
    h5_file_crop_features: Path,
    h5_file_captions_output: Path,
) -> float:
    crop_index_to_crop_features = import_crop_features_from_hdf5(
        h5_file_crop_features)
    crop_index_to_captions_output = import_captions_output_from_hdf5(
        h5_file_captions_output)

    # Extract indexes to be sures to use same crop_indexes for both dict
    crop_indexes = list(crop_index_to_captions_output.keys())

    # NOTE: same crop_index for two consecutive queries
    queries_features = np.concatenate([
        caption_features.reshape(1, -1)
        for crop_index in crop_indexes
        for caption_features in crop_index_to_captions_output[crop_index]
    ])
    gallery_features = np.concatenate([
        crop_index_to_crop_features[crop_index].reshape(1, -1)
        for crop_index in crop_indexes
    ])

    all_similarities = np.einsum(
        'nd,md->nm', queries_features, gallery_features)

    all_labels = _get_labels_from_crop_indexes(crop_indexes)

    # similarities size is not the same as labels 2900 vs 5800
    return np.mean([
        compute_average_precision(labels, similarities)
        for similarities, labels in zip(all_similarities, all_labels)
    ])
