from typing import Callable, Tuple

from torch import Tensor
import torch

AVERAGE_WEIGHT = .5
ComputeSimilarities = Callable[..., Tensor]

def image_similarity_only(
    query_image_features: Tensor,
    frame_features: Tensor,
) -> Tensor:
    return torch.einsum("d,nd->n", query_image_features, frame_features)

def text_similarity_only(
    query_text_features: Tensor,
    frame_features: Tensor,
) -> Tensor:
    return torch.einsum("d,nd->n", query_text_features, frame_features)

def average(
    query_text_features: Tensor,
    query_image_features: Tensor,
    frame_features: Tensor,
    weight: float = AVERAGE_WEIGHT
) -> Tensor:
    text_image_similarities = text_similarity_only( query_text_features, frame_features)
    image_image_similarities = image_similarity_only( query_image_features, frame_features)
    return weight * text_image_similarities + (1 - weight) * image_image_similarities