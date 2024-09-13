from pathlib import Path
from typing import Dict

import h5py

from .data_struct import Detections
from .models.pstr import PSTR
from .utils import confirm_generation


def _export_detections_to_h5(
    frame_id_to_detections: Dict[int, Detections], h5_file: Path
) -> None:
    """Export PSTR outputs to a h5 file with a frame filename group and datasets
    scores, bboxes, features_pstr.

    Args:
        frame_file_to_detection (Dict[Path, DetectionOutput]): PSTR outputs mapped to frame file
        file (Path): path of h5 file to export
    """
    with h5py.File(h5_file, "w") as out_file:
        for frame_id, detections in frame_id_to_detections.items():
            group = out_file.create_group(str(frame_id))

            group.create_dataset("scores", data=detections.scores)
            group.create_dataset("bboxes", data=detections.bboxes)
            group.create_dataset("features_pstr", data=detections.features_pstr)


def import_detections_from_h5(in_file: Path) -> Dict[int, Detections]:
    """
    Generate a map between the frame file and their DetectionOutput from PSTR

    Args:
        h5_file (Path): h5 file containing the model's outputs
        frame_folder (Path): the folder where frames are located

    Returns:
        Dict[Path, DetectionOutput]: map between file object and model outputs.
    """
    with h5py.File(in_file, "r") as in_file:
        frame_path_to_detections = {
            frame_id: Detections(
                detections[Detections._fields[0]][...],
                detections[Detections._fields[1]][...],
                detections[Detections._fields[2]][...],
            )
            for frame_id, detections in in_file.items()
        }
    return frame_path_to_detections


def generate_detections_to_h5(
    config_file: Path,
    weight_file: Path,
    h5_file: Path,
) -> None:
    if not confirm_generation(h5_file):
        return

    model = PSTR(config_file, weight_file)

    frame_id_to_detections = _compute_detections(model)
    _export_detections_to_h5(frame_id_to_detections, h5_file)


def _compute_detections(model: PSTR) -> Dict[int, Detections]:
    """PSTR model computes the output of the list of frames loaded into it and
    outputs them in a dict that maps every file to their outputs.

    Args:
        model (PSTR): model configured.

    Returns:
        Dict[int, Detection]: map between file object and model outputs.
    """
    results_by_path = model.infer()

    return {
        frame_id: Detections(
            scores=result[:, 4],
            bboxes=result[:, :4],
            features_pstr=result[:, 5:],
        )
        for frame_id, result in results_by_path.items()
    }
