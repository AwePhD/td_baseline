from typing import Dict
from pathlib import Path

import h5py
import pandas as pd
import numpy as np
from PIL import Image
from irra.model.clip_model import CLIP
from tqdm import tqdm

from ..utils import prompt_rm_to_user, extract_int_from_str
from ..crop_features import compute_features_from_one_frame
from ..data_struct import FrameOutput, DetectionOutput

TOKEN_BATCH_SIZE = 512
H5_FRAME_OUTPUT_FILENAME = "filename_to_frame_output.h5"
H5_FRAME_OUTPUT_FILE = Path.cwd() / "outputs" / H5_FRAME_OUTPUT_FILENAME


def _compute_frame_file_to_frame_output(
    model: CLIP,
    frame_file_to_detection: Dict[Path, DetectionOutput],
) -> Dict[Path, FrameOutput]:
    return {
        frame_file: FrameOutput(
            detection.scores,
            detection.bboxes,
            detection.features_pstr,
            compute_features_from_one_frame(
                model, frame_file, detection.bboxes),
        )
        for frame_file, detection in tqdm(frame_file_to_detection.items())
    }


def export_frame_output_to_hdf5(
    frame_file_to_frame_output: Dict[Path, FrameOutput],
    h5_file: Path
):
    if h5_file.exists():
        h5_file.unlink()

    with h5py.File(h5_file, 'w') as f:
        for frame_file, frame_output in frame_file_to_frame_output.items():

            group = f.create_group(frame_file.name)

            group.create_dataset('scores', data=frame_output.scores)
            group.create_dataset('bboxes', data=frame_output.bboxes)
            group.create_dataset(
                'features_pstr', data=frame_output.features_pstr)
            group.create_dataset(
                'features_clip', data=frame_output.features_clip)


def assert_detection_output_and_annotations_compatibility(
    annotations: pd.DataFrame,
    frame_file_to_detection: Dict[Path, DetectionOutput]
) -> None:
    annotations_frame_ids = set(
        f"s{frame_id}.jpg"
        for frame_id in annotations.index.get_level_values("frame_id").unique()
    )
    h5_frame_ids = set(
        frame_file.name
        for frame_file in frame_file_to_detection
    )
    assert annotations_frame_ids == h5_frame_ids


def generate_frame_output_to_hdf5(
    frame_file_to_detection: Dict[Path, DetectionOutput],
    model: CLIP,
    h5_file: Path = H5_FRAME_OUTPUT_FILE,
) -> None:
    if h5_file.exists():
        if not prompt_rm_to_user(h5_file):
            return
        else:
            h5_file.unlink()

    frame_file_to_frame_output = _compute_frame_file_to_frame_output(
        model, frame_file_to_detection)
    export_frame_output_to_hdf5(frame_file_to_frame_output, h5_file)


def import_frame_output_from_hdf5(h5_file: Path = H5_FRAME_OUTPUT_FILE) -> Dict[int, FrameOutput]:
    with h5py.File(h5_file, 'r') as hd5_file:
        frame_id_to_frame_output = {
            extract_int_from_str(frame_filename):
                FrameOutput(
                    frame_output[FrameOutput._fields[0]][...],
                    frame_output[FrameOutput._fields[1]][...],
                    frame_output[FrameOutput._fields[2]][...],
                    frame_output[FrameOutput._fields[3]][...],
            )
            for frame_filename, frame_output in hd5_file.items()
        }

    return frame_id_to_frame_output
