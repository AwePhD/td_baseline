from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from irra.model.clip_model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_struct import CropIndex
from .models.clip import load_clip
from .models.tokenizer import SimpleTokenizer, tokenize
from .utils import confirm_generation, crop_index_from_filename


def _collate_tokens_dataloader(
    batch: List[Tuple[CropIndex, Tuple[torch.Tensor, torch.Tensor]]]
) -> Tuple[List[CropIndex], torch.Tensor]:
    """Format tuples (crop index, tensor of size 2, token_length) to two lists.
    One list for crop indexes and another for individual tokens. Crop indexes
    are doubled because for 1 crop index there are two tokens.

    Args:
        batch (List[Tuple[CropIndex, Tuple[torch.Tensor, torch.Tensor]]]):
            list of (crop_index, tokens [2, token_length])

    Returns:
        Tuple[List[CropIndex], torch.Tensor]: crop_indexes and tokens
    """
    crop_indexes = [sample[0] for sample in batch for _ in range(2)]

    tokens = torch.stack([tokens for sample in batch for tokens in sample[1]])

    return crop_indexes, tokens


def _compute_text_features(
    annotations: pd.DataFrame,
    model: CLIP,
    token_batch_size: int,
    vocab_file: Path,
) -> Dict[CropIndex, np.ndarray]:
    captions = annotations[["caption_1", "caption_2"]]

    # Tokenize captions
    tokenizer = SimpleTokenizer(vocab_file)
    tokens = [
        (
            CropIndex(*crop_index),
            (
                tokenize(caption_1, tokenizer),
                tokenize(caption_2, tokenizer),
            ),
        )
        for crop_index, caption_1, caption_2 in captions.itertuples()
    ]

    # Set the tokens dataloader
    tokens_dataloader: DataLoader = DataLoader(
        tokens,  # type: ignore
        batch_size=token_batch_size,
        collate_fn=_collate_tokens_dataloader,
        pin_memory=True,
    )

    crop_indexes: List[CropIndex] = []
    features_text_batch: List[torch.Tensor] = []
    for batch in tqdm(tokens_dataloader, leave=False):
        batch_crop_indexes, batch_tokens = batch

        crop_indexes.extend(batch_crop_indexes)

        with torch.no_grad():
            tokens_features: torch.Tensor
            # tokens_features = model.encode_text(batch_tokens.cuda()).cpu()
            tokens_features = model.encode_text(batch_tokens.cuda()).cpu()
        # Prends le token de <END_OF_SEQUENCE> => CLASS TOKEN
        features_text_batch.append(
            tokens_features[
                torch.arange(tokens_features.shape[0]),
                batch_tokens.argmax(dim=-1),
            ].float()
        )
    features_text = F.normalize(torch.cat(features_text_batch)).numpy()
    assert len(crop_indexes) == len(features_text)

    n_samples = len(crop_indexes)
    return {
        crop_indexes[i]: np.array([features_text[i], features_text[i + 1]])
        for i in range(0, n_samples, 2)
    }


def generate_text_features_to_h5(
    weight_file: Path,
    annotations_file: Path,
    token_batch_size: int,
    vocab_file: Path,
    h5_file: Path,
) -> None:
    if not confirm_generation(h5_file):
        return

    annotations = pd.read_parquet(annotations_file).dropna()
    model = load_clip(weight_file).eval().cuda()

    crop_index_to_text_features = _compute_text_features(
        annotations, model, token_batch_size, vocab_file
    )
    _export_text_features_to_h5(crop_index_to_text_features, h5_file)


def _export_text_features_to_h5(
    crop_index_to_captions_output: Dict[CropIndex, np.ndarray],
    h5_file: Path,
) -> None:
    with h5py.File(h5_file, "w") as out_file:
        for (
            crop_index,
            captions_output,
        ) in crop_index_to_captions_output.items():
            out_file.create_dataset(
                f"p{crop_index.person_id}_s{crop_index.frame_id}",
                data=captions_output,
            )


def import_features_text_from_h5(
    h5_file: Path,
) -> Dict[CropIndex, np.ndarray]:
    with h5py.File(h5_file, "r") as in_file:
        crop_index_to_features_text: Dict[CropIndex, np.ndarray] = {
            crop_index_from_filename(filename): text_features[...]
            for filename, text_features in in_file.items()
        }

    return crop_index_to_features_text
