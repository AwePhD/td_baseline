"""
Compute CLIP features from

- Crops files
- GT bboxes in the annotations. TODO: show differences between the crops
and the GT
- Crops from PSTR detection -> already computed from clip_features
TODO: This computation comes before the clip_features scripts, it should
be isolated and exported in the crops_features file. Then, the clip_features
just have to open this file to make FrameOutput
"""

import re
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from irra.model.clip_model import CLIP

from ..crop_features import (
    compute_features_from_crops,
    build_dataloader_from_crops,
)
from ..models.clip import load_clip, WEIGHT_FILE
from ..cuhk_sysu_pedes import (
    import_test_annotations,
    DATA_FOLDER,
    FRAME_FOLDER
)
from ..utils import gt_bboxes_from_annotations
from ..data_struct import CropIndex


def _crop_index_from_filename(filename: str) -> CropIndex:
    extract_consecutive_numbers = re.compile(r'[\d]+')

    crop_index = tuple(
        int(number)
        for number in extract_consecutive_numbers.findall(filename)
    )
    return CropIndex(*crop_index)


def _compute_clip_features_from_crops(
    model: CLIP,
    crops: List[ImageType],
) -> np.ndarray:
    crops_dataloader = build_dataloader_from_crops(crops)

    all_features: List[torch.Tensor] = []
    for crops in crops_dataloader:
        all_features.append(compute_features_from_crops(model, crops))

    return torch.cat(all_features)


def from_crops_files(
    crops_folder: Path, model_weight: Path = WEIGHT_FILE
) -> Dict[CropIndex, torch.Tensor]:
    """Output a map between the crop index (Person ID, Frame ID) and the crops
    CLIP features.

    Args:
        crops_folder (Path): Crops folder with all .jpg files
        model_weight (Path, optional): Path of your model. Defaults to
        WEIGHT_FILE.

    Returns:
        Dict[CropIndex, torch.Tensor]: All crops feature from your folder,
        can be retrieved by crop index (person ID, frame ID)
    """
    model = load_clip(model_weight).eval().cuda()

    crops = [
        Image.open(crop_path)
        for crop_path in crops_folder.iterdir()
    ]

    all_features = _compute_clip_features_from_crops(model, crops)

    return {
        _crop_index_from_filename(crop_path.name): features
        for crop_path, features in zip(crops_folder.iterdir(), all_features)
    }


def from_annotations(
    data_folder: Path = DATA_FOLDER, model_weight: Path = WEIGHT_FILE
) -> Dict[CropIndex, torch.Tensor]:
    model = load_clip(model_weight).eval().cuda()

    annotations = import_test_annotations(data_folder)

    gt_bboxes = gt_bboxes_from_annotations(annotations)

    crops = [
        (
            Image
            .open(FRAME_FOLDER / f's{crop_index.frame_id}.jpg')
            .crop(gt_bboxes[crop_index])
        )
        for crop_index in gt_bboxes.keys()
    ]

    all_features = _compute_clip_features_from_crops(model, crops)

    return {
        CropIndex(*crop_index): features
        for crop_index, features in zip(gt_bboxes.keys(), all_features)
    }


def export_to_hdf5(
        crop_index_to_features: Dict[CropIndex, np.ndarray], h5_file: Path):
    with h5py.File(h5_file, 'w') as f:
        for crop_index, features in crop_index_to_features.items():
            group = f.create_group(
                f"p{crop_index.person_id}_s{crop_index.frame_id}")

            group.create_dataset('features', data=features)
