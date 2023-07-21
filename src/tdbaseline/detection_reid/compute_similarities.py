from typing import Callable, Tuple

import numpy as np
from numpy.linalg import norm

# eval script only works with functions with those parameters
# if you wish to build a smarter similarities computation, then make
# a builder method that outputs what you want. See build_baseline_similarities
ComputeSimilarities = Callable[[
    np.ndarray, np.ndarray, np.ndarray], np.ndarray]

PSTR_FEATURES_LENTGH = 3 * 256
CLIP_FEATURES_LENTGH = 512


def normalize(features: np.ndarray) -> np.ndarray:
    """
    Normalize the 2D vectors (n_features, features_dim)
    """
    return features / norm(features, axis=1).reshape(-1, 1)


def compute_similarities(
    query_features: np.ndarray,
    crops_features: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        'ik,jk->ij',
        normalize(query_features.reshape(1, -1)),
        normalize(crops_features),
    ).ravel()


def baseline_similarities(
    query_crop_features_pstr_clip: Tuple[np.ndarray, np.ndarray],
    query_text_features: np.ndarray,
    crops_features_pstr_clip: Tuple[np.ndarray, np.ndarray],
    weight_of_text_features: float,
) -> np.ndarray:
    _, query_crop_features = query_crop_features_pstr_clip
    _, crops_features = crops_features_pstr_clip

    assert query_crop_features.shape[0] == CLIP_FEATURES_LENTGH
    assert query_text_features.shape[0] == CLIP_FEATURES_LENTGH
    assert crops_features.shape[1] == CLIP_FEATURES_LENTGH

    if weight_of_text_features == 1:
        return compute_similarities(query_text_features, crops_features)
    if weight_of_text_features == 0:
        return compute_similarities(query_crop_features, crops_features)

    text_image_similarities = compute_similarities(
        query_text_features, crops_features)
    image_image_similarities = compute_similarities(
        query_crop_features, crops_features)

    return weight_of_text_features * text_image_similarities + (1 - weight_of_text_features) * image_image_similarities


def build_baseline_similarities(weight_of_text_features: float) -> ComputeSimilarities:
    def built_baseline_similarities(*args):
        return baseline_similarities(*args, weight_of_text_features)

    return built_baseline_similarities


def pstr_similarities(
    query_crop_features_pstr_clip: Tuple[np.ndarray, np.ndarray],
    query_text_features: np.ndarray,  # pylint: disable=unused-argument
    crops_features_pstr_clip: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    # PSTR only does image similarites
    query_crop_features, _ = query_crop_features_pstr_clip
    crops_features, _ = crops_features_pstr_clip

    assert query_crop_features.shape[0] == PSTR_FEATURES_LENTGH
    assert crops_features.shape[1] == PSTR_FEATURES_LENTGH

    return compute_similarities(
        query_crop_features,
        crops_features
    )
