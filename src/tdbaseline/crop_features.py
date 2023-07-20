from pathlib import Path
from typing import List
import numpy as np
import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from PIL import Image as Img
from PIL.Image import Image

from .models.clip import CLIP
from .models.clip import IMAGE_SIZE

BATCH_SIZE = 256
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

_preprocess_crop = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])


def compute_features_from_crops(
        model: CLIP, crops: List[Image]) -> np.ndarray:
    # crop method perfoms an integer approximation
    crops_preprocessed = torch.stack(
        [_preprocess_crop(crop) for crop in crops])

    crops_dataloader = DataLoader(
        crops_preprocessed, batch_size=BATCH_SIZE, shuffle=False)

    crops_features: List[torch.Tensor] = []
    for crops_batch in crops_dataloader:
        with torch.no_grad():
            # Last layer + send to CPU
            crops_features_batch = model.encode_image(
                crops_batch.cuda())[:, 0, :].cpu()
        crops_features.append(crops_features_batch)

    return torch.cat(crops_features).numpy()


def compute_features_from_one_frame(
    model: CLIP,
    frame_file: Path,
    bboxes: np.ndarray,
) -> np.ndarray:
    frame = Img.open(str(frame_file))
    # crop method perfoms an integer approximation
    crops = [frame.crop(bbox) for bbox in bboxes]

    return compute_features_from_crops(model, crops)
