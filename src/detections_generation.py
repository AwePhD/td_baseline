from pathlib import Path
from typing import Dict, NamedTuple

import h5py
import torch

from pstr import PSTR

class DetectionOutput(NamedTuple):
    # (100,)
    scores: torch.Tensor
    # (100, 4)
    bboxes: torch.Tensor

def export_to_hdf5(
    frame_path_to_detection: Dict[Path, DetectionOutput],
    filename: Path
):
    with h5py.File(filename, 'w') as f:
        for frame_path, detection_output in frame_path_to_detection.items():
            group = f.create_group(frame_path.name)

            group.create_dataset('scores', data=detection_output.scores)
            group.create_dataset('bboxes', data=detection_output.bboxes)

def import_from_hdf5(filename: Path, frame_folder: Path) -> Dict[Path, DetectionOutput]:
    frame_path_to_detections = {}

    with h5py.File(filename, 'r') as hd5_file:
        frame_path_to_detections = {
            frame_folder / frame_filename:
                DetectionOutput(
                    torch.tensor(detection_output[DetectionOutput._fields[0]]),
                    torch.tensor(detection_output[DetectionOutput._fields[1]]),
                )
            for frame_filename, detection_output in hd5_file.items()
        }
    return frame_path_to_detections


def _get_detector_outputs_by_path() -> Dict[Path, DetectionOutput]:
    results_by_path = PSTR().infer()

    return {
        path: DetectionOutput(
            scores=result[0][0][:,4],
            bboxes=result[0][0][:,:4],
        )
        for path, result in results_by_path.items()
    }

def main():
    model = PSTR()

    frame_path_to_detection = _get_detector_outputs_by_path()

    h5_filename = "filename_to_detection.h5"
    export_to_hdf5(frame_path_to_detection, Path.cwd() / "outputs" / h5_filename)

if __name__ == "__main__":
    main()