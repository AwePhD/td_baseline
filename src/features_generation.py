from typing import Dict, Tuple, NamedTuple, List
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from irra.model.clip_model import CLIP, build_CLIP_from_openai_pretrained
from tokenizer import SimpleTokenizer, tokenize

from cuhk_sysu_pedes import read_annotations_csv

WEIGHT_FILE = Path().home() / "models" "clip_finetuned" / "clip_finetune.pth"
STRIDE_SIZE = 16
IMAGE_SIZE = (384, 128)
TOKEN_BATCH_SIZE = 512

class CropIndex(NamedTuple):
    person_id: int
    frame_id: int


def _load_clip(weight_file: Path = WEIGHT_FILE) -> CLIP:
    # Filter "base_mode." in front of params key.
    state_dict = {
        ".".join(key.split('.')[1:]): parameters
        for key, parameters in torch.load(weight_file)['model'].items()
    }

    model, _ = build_CLIP_from_openai_pretrained("ViT-B/16", IMAGE_SIZE, STRIDE_SIZE)
    model.load_state_dict(state_dict)

    return model

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

def _get_features_text(
    model: CLIP,
    token_batch_size: int = 512
) -> Dict[CropIndex, Tuple[torch.Tensor, torch.Tensor]]:
    # Import captions from CUHK-SYSU-PEDES annotations
    _, annotations = read_annotations_csv(
        Path.home() / "data" / "annotations_train.csv",
        Path.home() / "data" / "annotations_test.csv",
    )
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

def main():
    # Import model
    model = _load_clip()

    # Compute text features
    crop_index_to_text_features = _get_features_text(model)

    # Compute image features

    # Format features

    # Export them
    ...

if __name__ == "__main__":
    main()