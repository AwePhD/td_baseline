from pathlib import Path
# from typing import List

import numpy as np

# from tdbaseline.data_struct import CropIndex

from ..crop_features import import_features_crop_from_hdf5
from ..metrics import compute_average_precision
from ..text_features import import_features_text_from_h5


# def _normalize(features_matrix: np.ndarray) -> np.ndarray:
#     return features_matrix / np.linalg.norm(
#         features_matrix, axis=1, keepdims=True
#     )


# def evaluate(
#     h5_features_image: Path,
#     h5_features_text: Path,
# ) -> float:
#     crop_index_to_features_image = import_features_image_from_h5(
#         h5_features_image
#     )
#     crop_index_to_features_text = import_features_text_from_h5(
#         h5_features_text
#     )
#
#     crops_indexes = list(crop_index_to_features_text.keys())
#
#     n_crops_indexes_images = len(list(crop_index_to_features_image.keys()))
#     # If the origin is annotations or files, it should be 1, else it's <= 1.
#     recall = n_crops_indexes_images / len(crops_indexes)
#
#     # [n_text, 512]
#     features_text = _normalize(np.vstack([crop_index_to_features_text[i] for i in crops_indexes]))  # type: ignore
#     # [n_image, 512]
#     features_images = _normalize([crop_index_to_features_image[i] for i in crops_indexes])  # type: ignore
#
#     person_ids = np.array([person_id for person_id, _ in crops_indexes])
#     _all_labels = person_ids[:, None] == person_ids[None, :]
#     # [n_text, n_person_ids]
#     all_labels = np.repeat(_all_labels, repeats=2, axis=0)
#     assert len(features_text) == len(all_labels)
#
#     print(f"#Queries: {len(features_text):,d}")
#     print(f"#Gallery: {len(features_images):,d}")
#
#     all_similarities = np.einsum("nd,md->nm", features_text, features_images)
#
#     mean_average_precision = np.mean(
#         [
#             compute_average_precision(labels, similarities)
#             for similarities, labels in zip(all_similarities, all_labels)
#         ]
#     )
#
#     print(f"Before recall: {mean_average_precision:.2%}")
#     print(f"Recall: {recall:.2%}")
#     print(f"After recall: {mean_average_precision * recall:.2%}")
#     return mean_average_precision * recall


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
    features_gallery = np.array([crop_index_to_features_crops[index] for index in indexes])

    all_similarities = np.einsum("nd,md->nm", features_query, features_gallery)

    mean_average_precision = np.mean(
        [
            compute_average_precision(labels, similarities)
            for similarities, labels in zip(all_similarities, all_labels)
        ]
    )

    print(f"mAP: {mean_average_precision:.2%}")
