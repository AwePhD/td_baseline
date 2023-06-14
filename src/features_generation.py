from typing import Dict, Tuple, NamedTuple, List
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image
from irra.model.clip_model import CLIP, build_CLIP_from_openai_pretrained

from .tokenizer import SimpleTokenizer, tokenize
from .clip import load_clip
from .cuhk_sysu_pedes import read_annotations_csv
from .detections_generation import import_from_hdf5, H5_FILENAME, DetectionOutput

DATA_FOLDER = Path.home() / "data"
FRAME_FOLDER = DATA_FOLDER / "frames"
H5_FILE = Path.cwd() / "outputs" / H5_FILENAME
TOKEN_BATCH_SIZE = 512

class CropIndex(NamedTuple):
    person_id: int
    frame_id: int


def _import_annotations(data_folder: Path = DATA_FOLDER) -> pd.DataFrame:
    _, annotations = read_annotations_csv(
        data_folder / "annotations_train.csv",
        data_folder / "annotations_test.csv",
    )

    return annotations


def _collate_tokens_dataloader(batch: List[Tuple[CropIndex, Tuple[torch.Tensor, torch.Tensor]]]):
    crop_indexes = [
        sample[0]
        for sample in batch
        for _ in range(2)
    ]

    tokens = torch.stack(
        [
            tokens
            for sample in batch
            for tokens in sample[1]
        ]
    )

    return crop_indexes, tokens

def _get_text_features(
    annotations: pd.DataFrame,
    model: CLIP,
    token_batch_size: int = 512
) -> Dict[CropIndex, Tuple[torch.Tensor, torch.Tensor]]:
    annotations_query = annotations.query("type == 'query'")
    captions = pd.concat([annotations_query.caption_1, annotations_query.caption_2])

    # Tokenize captions
    tokenizer = SimpleTokenizer()
    tokens = {
        CropIndex(*crop_index): (
            tokenize(captions_pair.iloc[0], tokenizer),
            tokenize(captions_pair.iloc[1], tokenizer),
        )
        for crop_index, captions_pair in captions.groupby(by=['person_id', 'frame_id'])
    }

    # Set the tokens dataloader
    tokens_dataloader = DataLoader(
        [
            (crop_index, tokens_pair)
            for crop_index, tokens_pair in tokens.items()
        ],
        batch_size=token_batch_size,
        collate_fn=_collate_tokens_dataloader,
    )

    crop_indexes = []
    features_text = []
    for batch in tokens_dataloader:
        batch_crop_indexes, batch_tokens = batch

        crop_indexes.extend(batch_crop_indexes)

        with torch.no_grad():
            tokens_features = model.encode_text(batch_tokens.cuda())
        # Prends le token de <END_OF_SEQUENCE> => CLASS TOKEN
        features_text.extend(tokens_features[
             torch.arange(tokens_features.shape[0]), batch_tokens.argmax(dim=-1)
        ].float())
    assert len(crop_indexes) == len(features_text)

    n_samples = len(crop_indexes)
    return {
        crop_indexes[i]: (features_text[i], features_text[i+1])
        for i in range(0, n_samples, 2)
    }

def _compute_features_from_one_frame(model: CLIP, frame: Path, bboxes: torch.Tensor) -> torch.Tensor:
    # Get RoI from bboxes and frames
    frame = read_image(frame)

    # Preprocess RoI

    # Compute features all at once
    ...

def _get_image_features(
    model: CLIP,
    frame_path_to_detection: Dict[Path, DetectionOutput],
) -> Dict[Path, torch.Tensor]:
    # Take one frame at time and output the images features
    return {
        frame: _compute_features_from_one_frame(model, frame, detection.bboxes)
        for frame, detection in frame_path_to_detection.items()
    }

def main():
    # Import model
    model = load_clip()

    # Import annotations
    annotations = _import_annotations()

    # Compute text features
    crop_index_to_text_features = _get_text_features(annotations, model)

    # Compute image features
    frame_path_to_detection = import_from_hdf5(H5_FILE, FRAME_FOLDER)
    frame_to_image_features = _get_image_features(model, frame_path_to_detection)

    # Format features

    # Export them
    ...

if __name__ == "__main__":
    main()