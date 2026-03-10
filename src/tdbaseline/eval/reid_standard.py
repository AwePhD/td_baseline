from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tdbaseline.crop_features.from_dataset import (
    import_features_crop_from_hdf5,
)
from tdbaseline.data_struct import CropIndex
from tdbaseline.eval.metrics import compute_mean_average_precision, normalize


def evaluate_reid_from_h5(
    annotations_file: Path,
    features_crops_file_h5: Path,
) -> None:
    assert annotations_file.exists()
    assert features_crops_file_h5.exists()

    annotations = pd.read_parquet(annotations_file).dropna()
    crop_index_to_features = import_features_crop_from_hdf5(
        features_crops_file_h5
    )

    crops = [CropIndex(*index) for index in annotations.index]

    # (n_crops, d_CLIP)
    features_crops = np.array([crop_index_to_features[crop] for crop in crops])
    # (n_crops, d_CLIP)
    person_ids = np.array([crop.person_id for crop in crops])
    # (n_crops, n_crops)
    labels = person_ids[:, None] == person_ids[None, :]

    # (n_crops, n_crops)
    similarities = np.einsum(
        "nd,md->nm", normalize(features_crops), normalize(features_crops)
    )

    mAP = compute_mean_average_precision(
        torch.tensor(labels), torch.tensor(similarities)
    )

    length_gallery = labels.shape[1]
    print(f"mAP: {mAP:.2%}, Gallery size: {length_gallery:,d}")
