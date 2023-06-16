from pathlib import Path
from typing import Dict

import h5py
import torch

from pstr import PSTR
from data_struct import DetectionOutput

H5_FILENAME = "filename_to_detection.h5"

def export_to_hdf5(
    frame_file_to_detection: Dict[Path, DetectionOutput],
    file: Path
):
    with h5py.File(file, 'w') as f:
        for frame_file, detection_output in frame_file_to_detection.items():
            group = f.create_group(frame_file.name)

            group.create_dataset('scores', data=detection_output.scores)
            group.create_dataset('bboxes', data=detection_output.bboxes)

def import_from_hdf5(h5_file: Path, frame_folder: Path) -> Dict[Path, DetectionOutput]:
    frame_path_to_detections = {}

    with h5py.File(h5_file, 'r') as hd5_file:
        frame_path_to_detections = {
            frame_folder / frame_filename:
                DetectionOutput(
                    torch.tensor(detection_output[DetectionOutput._fields[0]]),
                    torch.tensor(detection_output[DetectionOutput._fields[1]]),
                )
            for frame_filename, detection_output in hd5_file.items()
        }
    return frame_path_to_detections


def _get_detector_outputs_by_path(model: PSTR) -> Dict[Path, DetectionOutput]:
    results_by_path = model.infer()

    return {
        path: DetectionOutput(
            scores=result[0][0][:,4],
            bboxes=result[0][0][:,:4],
        )
        for path, result in results_by_path.items()
    }

def main():
    model = PSTR()

    frame_file_to_detection = _get_detector_outputs_by_path(model)

    export_to_hdf5(frame_file_to_detection, Path.cwd() / "outputs" / H5_FILENAME)

if __name__ == "__main__":
    main()