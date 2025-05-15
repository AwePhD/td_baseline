from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
from PIL import Image
from PIL.Image import Image as Image_T
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.clip import load_clip
from ..pstr_output import import_detections_from_h5
from ..utils import confirm_generation
from .common import compute_clip_features_from_crops

NAME_DATASET = "CLIP_features"


def append_detections_features_to_hdf5(
    frame_id_to_detections_features: Dict[int, np.ndarray], h5_file: Path
):
    with h5py.File(h5_file, "a") as f:
        for frame_id, features in frame_id_to_detections_features.items():
            group = f.create_group(str(frame_id))
            group.create_dataset(NAME_DATASET, data=features)


def generate_crop_features_from_detections(
    clip_weight: Path,
    h5_file_frame_id_to_detection_output: Path,
    frame_folder: Path,
    batch_size: int,
    num_workers: int,
    h5_file: Path,
) -> None:
    if not confirm_generation(h5_file):
        return

    frame_id_to_detection_output = import_detections_from_h5(
        h5_file_frame_id_to_detection_output
    )

    model = load_clip(clip_weight).eval().cuda()

    dataloader = DataLoader(
        list(frame_id_to_detection_output.keys()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    for frame_ids_batch in tqdm(dataloader):
        frame_ids_batch = frame_ids_batch.tolist()
        crops_batch: List[Image_T] = []
        for frame_id in frame_ids_batch:
            detection_output = frame_id_to_detection_output[frame_id]
            frame = Image.open(frame_folder / f"s{frame_id}.jpg")
            # 100 [PIL.Image]
            crops_batch += [
                frame.crop(bbox) for bbox in detection_output.bboxes
            ]

        # (bs * 100, d_CLIP)
        features_batch = compute_clip_features_from_crops(model, crops_batch)

        frame_id_to_features = {
            frame_id: features_batch[i_sample : i_sample + 100].numpy()
            for i_sample, frame_id in enumerate(frame_ids_batch)
        }

        append_detections_features_to_hdf5(frame_id_to_features, h5_file)
