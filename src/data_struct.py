from pathlib import Path
from typing import NamedTuple, Tuple, List

import torch

class DetectionOutput(NamedTuple):
    # (100,)
    scores: torch.Tensor
    # (100, 4)
    bboxes: torch.Tensor

class FrameOutput(NamedTuple):
    # (100, )
    scores: torch.Tensor
    # (100, 4)
    bboxes: torch.Tensor
    # (100, 512)
    features: torch.Tensor

class CaptionsOutput(NamedTuple):
    caption_1: torch.Tensor
    caption_2: torch.Tensor

class CropIndex(NamedTuple):
    person_id: int
    frame_id: int


class DoubleQuery(NamedTuple):
    captions: Tuple[str, str]
    CaptionsOutput

class FrameOutput(NamedTuple):
    scores: torch.Tensor
    bboxes: torch.Tensor
    image_features: torch.Tensor

class GalleryElement(NamedTuple):
    frame: Path
    output: FrameOutput

Gallery = List[GalleryElement]

class Sample(NamedTuple):
    query: DoubleQuery
    gallery: Gallery

