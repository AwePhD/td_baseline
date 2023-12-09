"""
Compute CLIP features from different sources:

- Crops files
- GT bboxes in the annotations. TODO: show differences between the crops
and the GT

ATTENTION: do not confuse this module which is related to the Text ReID task
with the general crop_features.py module which is used for Text ReID and
Detection ReID.
"""

from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from PIL.Image import Image as ImageType
from irra.model.clip_model import CLIP

from tdbaseline.crop_features import compute_features_from_crops, preprocess_crops
from tdbaseline.models.clip import load_clip
from tdbaseline.cuhk_sysu_pedes import import_pedes_test_annotations
from tdbaseline.utils import (
    gt_bboxes_from_annotations,
    confirm_generation,
    crop_index_from_filename,
)
from tdbaseline.data_struct import CropIndex
from tdbaseline.text_reid.crops_features import export_crops_features_to_hdf5


def _collate_pil(batch: List[ImageType]) -> List[ImageType]:
    return batch


def _compute_clip_features_from_crops(
    model: CLIP,
    batch_size: int,
    num_workers: int,
    crops: List[ImageType],
) -> np.ndarray:
    crops_dataloader = DataLoader(
        crops,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
    )

    all_features: List[torch.Tensor] = []
    for crops_batch in crops_dataloader:
        crops_preprocessed = preprocess_crops(crops_batch)
        all_features.append(compute_features_from_crops(model, crops_preprocessed))

    return torch.cat(all_features).numpy()


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
        if crop_path.suffix == ".jpg"
    ]

    all_features = _compute_clip_features_from_crops(
        model, batch_size, num_workers, crops
    )

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
    crop_index_to_gt_bboxes = gt_bboxes_from_annotations(annotations)

    crops = [
        (
            Image.open(frames_folder / f"s{crop_index.frame_id}.jpg").crop(
                crop_index_to_gt_bboxes[crop_index]
            )
        )
        for crop_index in crop_index_to_gt_bboxes.keys()
    ]

    all_features = _compute_clip_features_from_crops(
        model, batch_size, num_workers, crops
    )

    return {
        CropIndex(*crop_index): features
        for crop_index, features in zip(crop_index_to_gt_bboxes.keys(), all_features)
    }


def generate_crop_features_from_files(
    model_weight: Path,
    crops_folder: Path,
    batch_size: int,
    num_workers: int,
    h5_file: Path,
) -> None:
    if not confirm_generation(h5_file):
        return

    model = load_clip(model_weight).eval().cuda()

    crop_index_to_features = _from_files(crops_folder, model, batch_size, num_workers)
    export_crops_features_to_hdf5(crop_index_to_features, h5_file)


def generate_crop_features_from_annotations(
    model_weight: Path,
    data_folder: Path,
    frames_folder: Path,
    batch_size: int,
    num_workers: int,
    h5_file: Path,
) -> None:
    if not confirm_generation(h5_file):
        return

    model = load_clip(model_weight).eval().cuda()
    annotations = import_pedes_test_annotations(data_folder).dropna()

    crop_index_to_features = _from_annotations(
        annotations, frames_folder, model, batch_size, num_workers
    )
    export_crops_features_to_hdf5(crop_index_to_features, h5_file)


def import_crop_features_from_hdf5(h5_file: Path) -> Dict[CropIndex, np.ndarray]:
    with h5py.File(h5_file, "r") as h5_content:
        crop_index_to_crop_features = {
            crop_index_from_filename(filename): dataset["features"][...]
            for filename, dataset in h5_content.items()
        }
    return crop_index_to_crop_features
