from typing import NamedTuple, List, Optional

import torch

class DetectionOutput(NamedTuple):
    # (100,)
    scores: torch.Tensor
    # (100, 4)
    bboxes: torch.Tensor
    # (100, 512)
    features_pstr: torch.Tensor

class CaptionsOutput(NamedTuple):
    caption_1: torch.Tensor
    caption_2: torch.Tensor

class FrameOutput(NamedTuple):
    # (100,)
    scores: torch.Tensor
    # (100, 4)
    bboxes: torch.Tensor
    # (100, 512)
    features_pstr: torch.Tensor
    # (100, 512)
    features_clip: torch.Tensor

class CropIndex(NamedTuple):
    person_id: int
    frame_id: int

class Query(NamedTuple):
    frame_id: int
    frame_output: FrameOutput
    captions_output: CaptionsOutput
    # (4,)
    gt_bbox: torch.Tensor

class GalleryFrame(NamedTuple):
    frame_id: int
    # (100, 4)
    frame_output: FrameOutput
    gt_bbox: Optional[torch.Tensor]


Gallery = List[GalleryFrame]

class Sample(NamedTuple):
    person_id: int
    query: Query
    gallery: Gallery