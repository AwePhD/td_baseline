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
    """
    The PEDES dataset has 2 captions for each crops.
    So each query in SYSU we can have two captions.
    Thus, it doubles the number of query. For `f` a single query frame in SYSU,
    we have 2 query in SYSU-PEDES `(f, caption_1), (f, caption_2)`
    """
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
    double_query: DoubleQuery
    gallery: Gallery

