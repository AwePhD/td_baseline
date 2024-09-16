from pathlib import Path

import numpy as np

from ..metrics import compute_average_precision
from ..text_features import import_features_text_from_h5
from .io import import_features_image_from_h5


def _normalize(features_matrix: np.ndarray) -> np.ndarray:
    return features_matrix / np.linalg.norm(
        features_matrix, axis=1, keepdims=True
    )


def evaluate(
    h5_features_image: Path,
    h5_features_text: Path,
) -> float:
    crop_index_to_features_image = import_features_image_from_h5(
        h5_features_image
    )
    crop_index_to_features_text = import_features_text_from_h5(
        h5_features_text
    )

    crops_indexes = list(crop_index_to_features_text.keys())

    n_crops_indexes_images = len(list(crop_index_to_features_image.keys()))
    # If the origin is annotations or files, it should be 1, else it's <= 1.
    recall = n_crops_indexes_images / len(crops_indexes)

    # [n_text, 512]
    _features_text = [crop_index_to_features_text[i] for i in crops_indexes]
    features_text = _normalize(np.vstack(_features_text))  # type: ignore
    # [n_image, 512]
    _features_images = [crop_index_to_features_image[i] for i in crops_indexes]
    features_images = _normalize(np.vstack(_features_images))  # type: ignore

    person_ids = np.array([person_id for person_id, _ in crops_indexes])
    _all_labels = person_ids[:, None] == person_ids[None, :]
    # [n_text, n_person_ids]
    all_labels = np.repeat(_all_labels, repeats=2, axis=0)
    assert len(features_text) == len(all_labels)

    print(f"#Queries: {len(features_text):,d}")
    print(f"#Gallery: {len(features_images):,d}")

    all_similarities = np.einsum("nd,md->nm", features_text, features_images)

    mean_average_precision = np.mean(
        [
            compute_average_precision(labels, similarities)
            for similarities, labels in zip(all_similarities, all_labels)
        ]
    )

    print(f"Before recall: {mean_average_precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"After recall: {mean_average_precision * recall:.2%}")
    return mean_average_precision * recall
