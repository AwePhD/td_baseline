from pathlib import Path

import numpy as np

from ..crop_features import import_features_crop_from_hdf5
from ..text_features import import_features_text_from_h5
from .metrics import compute_average_precision


def evaluate_treid_from_h5(
    annotations_file: Path,
    crop_index_to_features_text_h5: Path,
    crop_index_to_features_crops_h5: Path,
) -> None:
    assert annotations_file.exists()
    assert crop_index_to_features_text_h5.exists()
    assert crop_index_to_features_crops_h5.exists()

    crop_index_to_features_text = import_features_text_from_h5(
        crop_index_to_features_text_h5
    )
    crop_index_to_features_crops = import_features_crop_from_hdf5(
        crop_index_to_features_crops_h5
    )
    assert set(crop_index_to_features_text.keys()) == set(
        crop_index_to_features_crops.keys()
    )

    indexes = sorted(crop_index_to_features_crops.keys())
    person_ids = np.array([index.person_id for index in indexes])
    all_labels = (
        person_ids.reshape(-1, 1) == person_ids.reshape(1, -1)
    ).repeat(2, axis=0)

    # n_crops [(2, d_CLIP)] -> (n_crops * 2, d_CLIP)
    features_query = np.concatenate(
        [crop_index_to_features_text[index] for index in indexes]
    )

    # n_crops [(d_CLIP)] -> (n_crops, d_CLIP)
    features_gallery = np.array(
        [crop_index_to_features_crops[index] for index in indexes]
    )

    all_similarities = np.einsum("nd,md->nm", features_query, features_gallery)

    mean_average_precision = np.mean(
        [
            compute_average_precision(labels, similarities)
            for similarities, labels in zip(all_similarities, all_labels)
        ]
    )

    print(f"mAP: {mean_average_precision:.2%}")
