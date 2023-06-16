from typing import List, Tuple, Generator

import pandas as pd
import torch
import tqdm

from features_generation import _import_annotations
from data_struct import Sample

GALLERY_SIZE = 100

def _load_one_sample(annotation_sample: pd.DataFrame) -> Sample:
    ...

def _evaluate_one_sample(
    sample: Sample
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...

def main():
    # For each sample, evaluate the result of search
    # -> get the labels and scores array
    # NOTE: would be cool to have a generator.
    annotations = _import_annotations()
    samples: Generator[Sample] = (
        _load_one_sample(annotation_sample)
        for annotation_sample in annotations.groupby("frame_id")
    )

    labels_list: List[torch.Tensor] = []
    scores_list: List[torch.Tensor] = []

    for sample in tqdm(samples):
        labels_sample, scores_sample = _evaluate_one_sample(sample)

        labels_list.append(labels_sample)
        scores_list.append(scores_sample)

    labels = torch.stack(labels_list)
    scores = torch.stack(scores_list)

    # Evaluate the mAP on the whole sets of search performance

if __name__ == "__main__":
    main()