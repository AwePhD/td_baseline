from typing import NamedTuple, List, Tuple

import torch

from pstr import PSTR

class ModelOutput(NamedTuple):
    score: float
    # (4,)
    bbox: torch.Tensor
    # (?,)
    features_image: torch.Tensor
    # (?,)
    features_text: torch.Tensor

class SampleOutput(NamedTuple):
    query: ModelOutput
    gallery: ModelOutput

def _get_bboxes_and_scores() -> Tuple[List[torch.Tensor], List[float]]:
    results = PSTR().infer()

    bboxes = [
        result[0][0][:,:4]
        for result in results
    ]
    scores = [
        result[0][0][:,4]
        for result in results
    ]

    return bboxes, scores




def main():
    # Get bboxes for each samples
    # Get features for each bboxes of each sample
    # Build Output
    # Export output
    bboxes, scores = _get_bboxes_and_scores()

if __name__ == "__main__":
    main()