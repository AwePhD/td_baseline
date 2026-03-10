from typing import List

import torch
import torchvision.transforms as T
from irra.model.clip_model import CLIP
from PIL.Image import Image as ImageType

from ..models.clip import IMAGE_SIZE

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

_transform_crop = T.Compose(
    [T.Resize(IMAGE_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)]
)


def _preprocess_crops(crops: List[ImageType]) -> torch.Tensor:
    return torch.stack([_transform_crop(crop) for crop in crops])


def _compute_features_from_crops(
    model: CLIP, crops_preprocessed: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        # Last layer + send to CPU
        crops_features: torch.Tensor = model.encode_image(
            crops_preprocessed.cuda()
        )[:, 0, :].cpu()

    return crops_features


def compute_clip_features_from_crops(
    model: CLIP, crops: List[ImageType]
) -> torch.Tensor:
    crops_preprocessed = _preprocess_crops(crops)
    features = _compute_features_from_crops(model, crops_preprocessed)
    return features
