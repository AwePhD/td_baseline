from typing_extensions import TypeAlias
from typing import NamedTuple, List, Optional

import numpy as np


class Detections(NamedTuple):
    # (100,)
    scores: np.ndarray
    # (100, 4)
    bboxes: np.ndarray
    # (100, 512)
    features_pstr: np.ndarray


#: (2, 512) No need to have two different name for this, too much clutter.
CaptionsOutput: TypeAlias = np.ndarray


class FrameOutput(NamedTuple):
    # (100,)
    scores: np.ndarray
    # (100, 4)
    bboxes: np.ndarray
    # (100, 512)
    features_pstr: np.ndarray
    # (100, 512)
    features_clip: np.ndarray


class CropIndex(NamedTuple):
    person_id: int
    frame_id: int


class Query(NamedTuple):
    frame_id: int
    frame_output: FrameOutput
    captions_output: CaptionsOutput
    # (4,)
    gt_bbox: np.ndarray


class GalleryFrame(NamedTuple):
    frame_id: int
    # (100, 4)
    frame_output: FrameOutput
    gt_bbox: Optional[np.ndarray]


Gallery = List[GalleryFrame]


class Sample(NamedTuple):
    person_id: int
    query: Query
    gallery: Gallery
