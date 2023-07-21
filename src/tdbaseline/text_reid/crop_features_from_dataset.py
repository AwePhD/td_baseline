"""
Compute CLIP features from different sources:

- Crops files
- GT bboxes in the annotations. TODO: show differences between the crops
and the GT

ATTENTION: do not confuse this module which is related to the Text ReID task
with the general crop_features.py module which is used for Text ReID and
Detection ReID.
"""

import re
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from irra.model.clip_model import CLIP

from ..crop_features import (
    compute_features_from_crops,
    build_dataloader_from_crops,
)
from ..models.clip import load_clip
from ..cuhk_sysu_pedes import import_test_annotations
from ..utils import (
    gt_bboxes_from_annotations,
    confirm_generation,
    crop_index_from_filename,
)
from ..data_struct import CropIndex


def _compute_clip_features_from_crops(
    model: CLIP,
    batch_size: int,
    num_workers: int,
    crops: List[ImageType],
) -> np.ndarray:
    crops_dataloader = build_dataloader_from_crops(
        crops, batch_size, num_workers)

    all_features: List[torch.Tensor] = []
    for crops in crops_dataloader:
        all_features.append(compute_features_from_crops(model, crops))

    return torch.cat(all_features)


def _from_files(
    crops_folder: Path,
    model: CLIP,
    batch_size: int,
    num_workers: int,
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
    crops = [
        Image.open(crop_path)
        for crop_path in crops_folder.iterdir()
    ]

    all_features = _compute_clip_features_from_crops(
        model, batch_size, num_workers, crops)

    return {
        crop_index_from_filename(crop_path.name): features
        for crop_path, features in zip(crops_folder.iterdir(), all_features)
    }


def _from_annotations(
    annotations: pd.DataFrame,
    frames_folder: Path,
    model: CLIP,
    batch_size: int,
    num_workers: int,
) -> Dict[CropIndex, torch.Tensor]:
    gt_bboxes = gt_bboxes_from_annotations(annotations)

    crops = [
        (
            Image
            .open(frames_folder / f's{crop_index.frame_id}.jpg')
            .crop(gt_bboxes[crop_index])
        )
        for crop_index in gt_bboxes.keys()
    ]

    all_features = _compute_clip_features_from_crops(
        model, batch_size, num_workers, crops)

    return {
        CropIndex(*crop_index): features
        for crop_index, features in zip(gt_bboxes.keys(), all_features)
    }


def _export_crops_features_to_hdf5(
        crop_index_to_features: Dict[CropIndex, np.ndarray], h5_file: Path):
    with h5py.File(h5_file, 'w') as f:
        for crop_index, features in crop_index_to_features.items():
            group = f.create_group(
                f"p{crop_index.person_id}_s{crop_index.frame_id}")

            group.create_dataset('features', data=features)


def generate_crop_features_from_files(
    model_weight: Path,
    crops_folder: Path,
    batch_size: int,
    num_workers: int,
    h5_file: Path
) -> None:
    if not confirm_generation(h5_file):
        return

    model = load_clip(model_weight).eval().cuda()

    crop_index_to_features = _from_files(
        crops_folder, model, batch_size, num_workers)
    _export_crops_features_to_hdf5(crop_index_to_features, h5_file)


def generate_crop_features_from_annotations(
    model_weight: Path,
    data_folder: Path,
    frames_folder: Path,
    batch_size: int,
    num_workers: int,
    h5_file: Path
) -> None:
    if not confirm_generation(h5_file):
        return

    model = load_clip(model_weight).eval().cuda()
    annotations = import_test_annotations(data_folder)

    crop_index_to_features = _from_annotations(
        annotations, frames_folder, model, batch_size, num_workers)
    _export_crops_features_to_hdf5(crop_index_to_features, h5_file)


def import_crop_features_from_hdf5(
    h5_file: Path,
) -> Dict[CropIndex, np.ndarray]:
    with h5py.File(h5_file, 'r') as h5_content:
        crop_index_to_crop_features = {
            crop_index_from_filename(filename): dataset['features'][...]
            for filename, dataset in h5_content.items()
        }
    return crop_index_to_crop_features
