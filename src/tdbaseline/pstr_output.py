from pathlib import Path
from typing import Dict

import h5py

from .models.pstr import PSTR
from .data_struct import DetectionOutput
from .utils import confirm_generation

H5_FILENAME = "filename_to_detection.h5"
H5_FILE = Path.cwd() / "outputs" / H5_FILENAME


def _export_detection_output_to_hdf5(
    frame_file_to_detection: Dict[Path, DetectionOutput],
    h5_file: Path
) -> None:
    """Export PSTR outputs to a h5 file with a frame filename group and datasets
    scores, bboxes, features_pstr.

    Args:
        frame_file_to_detection (Dict[Path, DetectionOutput]): PSTR outputs mapped to frame file
        file (Path): path of h5 file to export
    """
    with h5py.File(h5_file, 'w') as f:
        for frame_file, detection_output in frame_file_to_detection.items():
            group = f.create_group(frame_file.name)

            group.create_dataset('scores', data=detection_output.scores)
            group.create_dataset('bboxes', data=detection_output.bboxes)
            group.create_dataset(
                'features_pstr', data=detection_output.features_pstr)


def import_detection_output_from_hdf5(
    h5_file: Path, frame_folder: Path
) -> Dict[Path, DetectionOutput]:
    """Generate a map between the frame file and their DetectionOutput from PSTR

    Args:
        h5_file (Path): h5 file containing the model's outputs
        frame_folder (Path): the folder where frames are located

    Returns:
        Dict[Path, DetectionOutput]: map between file object and model outputs.
    """
    with h5py.File(h5_file, 'r') as hd5_file:
        frame_path_to_detections = {
            frame_folder / frame_filename:
                DetectionOutput(
                    detection_output[DetectionOutput._fields[0]][...],
                    detection_output[DetectionOutput._fields[1]][...],
                    detection_output[DetectionOutput._fields[2]][...],
                )
            for frame_filename, detection_output in hd5_file.items()
        }
    return frame_path_to_detections


def generate_detection_output_to_hdf5(
    model: PSTR,
    h5_file: Path,
) -> None:
    if not confirm_generation(h5_file):
        return

    frame_file_to_detection_output = _get_detection_output(model)
    _export_detection_output_to_hdf5(frame_file_to_detection_output, h5_file)


def _get_detection_output(model: PSTR) -> Dict[Path, DetectionOutput]:
    """PSTR model computes the output of the list of frames loaded into it and outputs
    them in a dict that maps every file to their outputs.

    Args:
        model (PSTR): model configured.

    Returns:
        Dict[Path, DetectionOutput]: map between file object and model outputs.
    """
    results_by_path = model.infer()

    return {
        path: DetectionOutput(
            scores=result[:, 4],
            bboxes=result[:, :4],
            features_pstr=result[:, 5:],
        )
        for path, result in results_by_path.items()
    }
