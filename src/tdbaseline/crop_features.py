from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torchvision.transforms as T

from irra.model.clip_model import CLIP
from torch.utils.data import DataLoader
from PIL import Image
from PIL.Image import Image as ImageType
from tqdm import tqdm
import h5py

from .models.clip import load_clip, IMAGE_SIZE
from .data_struct import DetectionOutput
from .utils import extract_int_from_str, confirm_generation
from .pstr_output import import_detection_output_from_hdf5


FRAME_BATCH_SIZE = 3
CROPS_BATCH_SIZE = 400
NUM_WORKERS = 4
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

_transform_crop = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])


def _preprocess_crops(crops: List[ImageType]) -> torch.Tensor:
    return torch.stack(
        [_transform_crop(crop) for crop in crops])


def build_dataloader_from_crops(
    crops: List[ImageType],
    batch_size: int = CROPS_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    # crop method perfoms an integer approximation
    crops_preprocessed = _preprocess_crops(crops)

    return DataLoader(
        crops_preprocessed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def compute_features_from_crops(
        model: CLIP, crops_preprocessed: torch.Tensor) -> torch.Tensor:
    crops_features: List[torch.Tensor] = []
    with torch.no_grad():
        # Last layer + send to CPU
        crops_features = model.encode_image(
            crops_preprocessed.cuda())[:, 0, :].cpu()

    return crops_features


def _collate_frames_dataloader(
    batch: List[Tuple[Path, DetectionOutput]]
) -> Tuple[List[Path], List[np.ndarray]]:
    batch_paths = [frame_sample[0] for frame_sample in batch]
    batch_bboxes = [frame_sample[1].bboxes for frame_sample in batch]

    return batch_paths, batch_bboxes


def _get_clip_features_from_one_batch_bboxes(
    model: CLIP,
    frame_files: List[Path],
    frames_bboxes: List[np.ndarray],
) -> Dict[Path, np.ndarray]:
    frames = [Image.open(str(frame_file)) for frame_file in frame_files]

    # Crop each bbox for each frame, gives a list of crops for each frames
    # Then flatten the list of list of crops
    # to prepare a list of crops for features computation
    frames_crops = [
        [
            frame.crop(bbox.astype(np.int32))
            for bbox in frame_bboxes
        ]
        for frame, frame_bboxes in zip(frames, frames_bboxes)
    ]
    crops = [
        crop
        for frame_crops in frames_crops
        for crop in frame_crops
    ]

    crops_preprocessed = _preprocess_crops(crops)

    features = compute_features_from_crops(model, crops_preprocessed)
    split_features = np.split(features, len(frame_files))

    assert len(frames) == len(split_features)

    return {
        frame_file: bboxes_features
        for frame_file, bboxes_features in zip(frame_files, split_features)
    }


def _get_bboxes_clip_features_from_detections(
    model: CLIP,
    frame_file_to_detection_output: Dict[Path, DetectionOutput],
    batch_size: int = FRAME_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Dict[Path, np.ndarray]:
    frames_dataloader = DataLoader(
        list(frame_file_to_detection_output.items()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_frames_dataloader
    )

    # Path -> [100, 512] array
    frame_file_to_bboxes_clip_features: Dict[Path, np.ndarray] = {}
    for frame_files, bboxes in tqdm(frames_dataloader):
        frame_file_to_bboxes_clip_features.update(
            _get_clip_features_from_one_batch_bboxes(
                model, frame_files, bboxes,)
        )

    return frame_file_to_bboxes_clip_features


def import_bboxes_clip_features_from_hdf5(
    h5_file: Path,
):
    with h5py.File(h5_file, 'r') as hd5_file:
        frame_id_to_bboxes_clip_features = {
            extract_int_from_str(file_name): dataset['features'][...]
            for file_name, dataset in hd5_file.items()
        }

    return frame_id_to_bboxes_clip_features


def _export_bboxes_clip_features_to_hdf5(
    frame_file_to_bboxes_clip_features: Dict[Path, np.ndarray],
    output_h5: Path,
) -> None:
    if output_h5.exists():
        output_h5.unlink()

    with h5py.File(output_h5, 'w') as f:
        for frame_file, features in frame_file_to_bboxes_clip_features.items():
            group = f.create_group(frame_file.name)

            group.create_dataset('features', data=features)


def generate_bboxes_clip_features_from_detections(
    weight_file: Path,
    frame_folder: Path,
    h5_file_detection_output: Path,
    batch_size: int,
    num_workers: int,
    output_h5_file: Path,
) -> None:
    if not confirm_generation(output_h5_file):
        return

    model = load_clip(weight_file)
    frame_file_to_detection_output = import_detection_output_from_hdf5(
        h5_file_detection_output, frame_folder)
    frame_id_to_bboxes_clip_features = _get_bboxes_clip_features_from_detections(
        model,
        frame_file_to_detection_output,
        batch_size,
        num_workers,
    )

    _export_bboxes_clip_features_to_hdf5(
        frame_id_to_bboxes_clip_features, output_h5_file)
