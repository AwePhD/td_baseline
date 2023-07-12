from typing import Callable

import numpy as np
from numpy.linalg import norm
import torch

AVERAGE_WEIGHT = .5
ComputeSimilarities = Callable[..., np.ndarray]

PSTR_FEATURES_LENTGH = 3 * 256
CLIP_FEATURES_LENTGH = 512

def normalize(features: np.ndarray) -> np.ndarray:
    """
    Normalize the 2D vectors (n_features, features_dim)
    """
    return features / norm(features, axis=1).reshape(-1, 1)

def pstr_similarities(
    query_crop_features_pstr_clip: np.ndarray,
    query_text_features: np.ndarray,
    crops_features_pstr_clip: np.ndarray,
) -> np.ndarray:
    query_crop_features_pstr, _ = query_crop_features_pstr_clip
    assert len(query_crop_features_pstr) == PSTR_FEATURES_LENTGH

    crops_features_pstr, _ = crops_features_pstr_clip
    assert all(
            len(crop_features) == PSTR_FEATURES_LENTGH
            for crop_features in crops_features_pstr
    )

    return np.einsum(
        'ik,jk->ij',
        normalize(query_crop_features_pstr.reshape(1, -1)),
        normalize(crops_features_pstr),
    ).squeeze()


def image_similarity_only(
    query_image_features: np.ndarray,
    frame_features: np.ndarray,
) -> np.ndarray:
    return torch.einsum("d,nd->n", query_image_features, frame_features)

def text_similarity_only(
    query_text_features: np.ndarray,
    frame_features: np.ndarray,
) -> np.ndarray:
    return torch.einsum("d,nd->n", query_text_features, frame_features)

def average(
    query_image_features_pstr_clip: np.ndarray,
    query_text_features: np.ndarray,
    frame_features_pstr_clip: np.ndarray,
    weight: float = AVERAGE_WEIGHT
) -> np.ndarray:
    if weight == 1:
        return text_similarity_only(query_text_features, frame_features_pstr_clip)
    if weight == 0:
        return image_similarity_only( query_image_features_pstr_clip, frame_features_pstr_clip)

    text_image_similarities = text_similarity_only( query_text_features, frame_features_pstr_clip)
    image_image_similarities = image_similarity_only( query_image_features_pstr_clip, frame_features_pstr_clip)
    return weight * text_image_similarities + (1 - weight) * image_image_similarities