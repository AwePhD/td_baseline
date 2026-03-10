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
import torch.nn.functional as F
from irra.model.clip_model import CLIP
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_struct import CropIndex
from ..models.clip import load_clip
from ..utils import (
    confirm_generation,
    crop_index_from_filename,
    gt_bboxes_from_annotations,
)
from .common import compute_clip_features_from_crops


def _export_crop_index_to_features_to_hdf5(
    crop_index_to_features: Dict[CropIndex, np.ndarray], h5_file: Path
):
    with h5py.File(h5_file, "w") as f:
        for crop_index, features in crop_index_to_features.items():
            group = f.create_group(
                f"p{crop_index.person_id}_s{crop_index.frame_id}"
            )

            group.create_dataset("features", data=features)


def _from_files(
    files_test: List[Path],
    model: CLIP,
    batch_size: int,
    num_workers: int,
) -> Dict[CropIndex, np.ndarray]:
    """Output a map between the crop index (Person ID, Frame ID) and the crops
    CLIP features.

    Args:
        files_test (Path): Crops test .jpg files
        model_weight (Path, optional): Path of your model. Defaults to
        WEIGHT_FILE.

    Returns:
        Dict[CropIndex, torch.Tensor]: All crops feature from your folder,
        can be retrieved by crop index (person ID, frame ID)
    """
    crops: List[ImageType] = [
        Image.open(crop_path)
        for crop_path in files_test
        if crop_path.suffix == ".jpg"
    ]

    crops_dataloader: DataLoader[ImageType] = DataLoader(
        crops,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
    )
    all_features_list: List[torch.Tensor] = []
    for crops_batch in tqdm(crops_dataloader, leave=False):
        all_features_list.append(
            compute_clip_features_from_crops(model, crops_batch)
        )
    # (n_crops, d_CLIP)
    all_features = F.normalize(torch.cat(all_features_list)).numpy()

    return {
        crop_index_from_filename(crop_path.name): features
        for crop_path, features in zip(files_test, all_features)
    }


def _from_annotations(
    annotations: pd.DataFrame,
    frames_folder: Path,
    model: CLIP,
    batch_size: int,
    num_workers: int,
) -> Dict[CropIndex, np.ndarray]:
    crop_index_to_gt_bboxes = gt_bboxes_from_annotations(annotations)

    crops = [
        (
            Image.open(frames_folder / f"s{crop_index.frame_id}.jpg").crop(
                gt_bboxes
            )
        )
        for crop_index, gt_bboxes in crop_index_to_gt_bboxes.items()
    ]

    # (n_crops, d_CLIP)
    crops_dataloader: DataLoader[ImageType] = DataLoader(
        crops,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
    )
    all_features_list: List[torch.Tensor] = []
    for crops_batch in tqdm(crops_dataloader, leave=False):
        all_features_list.append(
            compute_clip_features_from_crops(model, crops_batch)
        )
    # (n_crops, d_CLIP)
    all_features = torch.cat(all_features_list).numpy()

    return {
        CropIndex(*crop_index): features
        for crop_index, features in zip(crop_index_to_gt_bboxes, all_features)
    }


def generate_features_crops_from_files(
    model_weight: Path,
    annotations_file: Path,
    crops_folder: Path,
    batch_size: int,
    num_workers: int,
    h5_file: Path,
) -> None:
    if not confirm_generation(h5_file):
        return

    annotations = pd.read_parquet(annotations_file).reset_index().dropna()
    model = load_clip(model_weight).eval().cuda()

    filenames_test = (
        "p"
        + annotations.person_id.astype(str)
        + "_s"
        + annotations.frame_id.astype(str)
        + ".jpg"
    ).unique()
    files_test = [
        crop_file
        for crop_file in crops_folder.iterdir()
        if crop_file.name in filenames_test
    ]

    crop_index_to_features = _from_files(
        files_test, model, batch_size, num_workers
    )
    _export_crop_index_to_features_to_hdf5(crop_index_to_features, h5_file)


def generate_features_crops_from_annotations(
    model_weight: Path,
    annotations_file: Path,
    frames_folder: Path,
    batch_size: int,
    num_workers: int,
    h5_file: Path,
) -> None:
    if not confirm_generation(h5_file):
        return

    model = load_clip(model_weight).eval().cuda()
    annotations = pd.read_parquet(annotations_file).dropna()

    crop_index_to_features = _from_annotations(
        annotations, frames_folder, model, batch_size, num_workers
    )
    _export_crop_index_to_features_to_hdf5(crop_index_to_features, h5_file)


def import_features_crop_from_hdf5(
    h5_file: Path,
) -> Dict[CropIndex, np.ndarray]:
    with h5py.File(h5_file, "r") as h5_content:
        crop_index_to_crop_features = {
            crop_index_from_filename(filename): dataset["features"][...]
            for filename, dataset in h5_content.items()
        }
    return crop_index_to_crop_features
