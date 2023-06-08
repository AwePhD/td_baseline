from pathlib import Path
from typing import NamedTuple, Dict, List

import torch

from models import PSTR

class ModelOutput(NamedTuple):
    #: (100,)
    scores: torch.Tensor
    # (100, 4)
    bboxes: torch.Tensor
    # (100, 256)
    features_image: torch.Tensor
    # (100, 256)
    features_text: torch.Tensor

class SampleOutput(NamedTuple):
    query: ModelOutput
    gallery: ModelOutput

class DetectorOutput(NamedTuple):
    # (100,)
    scores: torch.Tensor
    # (100, 4)
    bboxes: torch.Tensor

class ReIDOutput(NamedTuple):
    # (100, 256)
    features_image: torch.Tensor
    # (100, 4)
    features_text: torch.Tensor


def _get_detector_outputs_by_path() -> Dict[Path, DetectorOutput]:
    results_by_path = PSTR().infer()

    return {
        path: DetectorOutput(
            scores=result[0][0][:,4],
            bboxes=result[0][0][:,:4],
        )
        for path, result in results_by_path.items()
    }

def _get_features_text(paths: List[Path]) -> List[torch.Tensor]:
    ...

def _get_features_image(
    detector_outputs_by_path: Dict[Path, DetectorOutput]
) -> List[torch.Tensor]:
    ...


def _compute_reid_from_detections(
    detector_outputs_by_path: Dict[Path, DetectorOutput]
) -> Dict[Path, ReIDOutput]:
    # Get text features
    paths = list(detector_outputs_by_path.keys())

    # Get image features
    features_text = _get_features_text(paths)

    # Return ReID features
    features_image = _get_features_image(detector_outputs_by_path)

    return {
        path: ReIDOutput(
            features_image=f_img,
            features_text=f_txt,
        )
        for path, f_img, f_txt in zip(paths, features_image, features_text)
    }



def main():
    # Get bboxes for each frames
    detector_outputs_by_path = _get_detector_outputs_by_path()
    # Get features for each bboxes of each frames
    reid_outputs_by_path = _compute_reid_from_detections(detector_outputs_by_path)
    # Build Output
    model_outputs_by_path = {
        path: ModelOutput(
            scores=detector_outputs_by_path[path].scores,
            bboxes=detector_outputs_by_path[path].bboxes,
            bboxes=reid_outputs_by_path[path].features_image,
            bboxes=reid_outputs_by_path[path].features_text,
        )
        for path in detector_outputs_by_path.keys()
    }

    # Build samples output
    # e.g. organize into Query / Gallery




if __name__ == "__main__":
    main()