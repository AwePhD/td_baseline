from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

import h5py
from irra.model.clip_model import CLIP
import torch
from torch.utils.data import DataLoader

from .data_struct import CaptionsOutput, CropIndex, FrameOutput
from .cuhk_sysu_pedes import import_test_annotations
from .models.clip import load_clip
from .models.tokenizer import SimpleTokenizer, tokenize
from .utils import crop_index_from_filename, confirm_generation


def _collate_tokens_dataloader(
    batch: List[Tuple[CropIndex, Tuple[torch.Tensor, torch.Tensor]]]
) -> Tuple[List[CropIndex], torch.Tensor]:
    """Format tuples (crop index, tensor of size 2, token_length) to
    two lists. One list for crop indexes and another for individual tokens.
    Crop indexes are doubled because for 1 crop index there are two tokens.

    Args:
        batch (List[Tuple[CropIndex, Tuple[torch.Tensor, torch.Tensor]]]):
            list of (crop_index, tokens [2, token_length])

    Returns:
        Tuple[List[CropIndex], torch.Tensor]: crop_indexes and tokens
    """
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
    token_batch_size: int,
    vocab_file: Path,
) -> Dict[CropIndex, CaptionsOutput]:
    annotations_query = annotations.query("type == 'query'")
    captions = pd.concat([annotations_query.caption_1,
                         annotations_query.caption_2])

    # Tokenize captions
    tokenizer = SimpleTokenizer(vocab_file)
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
            tokens_features: torch.Tensor
            tokens_features = model.encode_text(batch_tokens.cuda()).cpu()
        # Prends le token de <END_OF_SEQUENCE> => CLASS TOKEN
        features_text.extend(
            tokens_features[torch.arange(
                tokens_features.shape[0]), batch_tokens.argmax(dim=-1)]
            .float()
            .numpy()
        )
    assert len(crop_indexes) == len(features_text)

    n_samples = len(crop_indexes)
    return {
        crop_indexes[i]: CaptionsOutput(features_text[i], features_text[i+1])
        for i in range(0, n_samples, 2)
    }


def generate_captions_output_to_hdf5(
    weight_file: Path,
    data_folder: Path,
    token_batch_size: int,
    vocab_file: Path,
    h5_file: Path,
) -> None:
    if not confirm_generation(h5_file):
        return

    annotations = import_test_annotations(data_folder)
    model = load_clip(weight_file).eval().cuda()

    crop_index_to_captions_outputs = _get_text_features(
        annotations,
        model,
        token_batch_size,
        vocab_file
    )
    _export_caption_features_to_hdf5(crop_index_to_captions_outputs, h5_file)


def _export_caption_features_to_hdf5(
    crop_index_to_captions_output: Dict[CropIndex, CaptionsOutput],
    h5_file: Path
) -> None:

    with h5py.File(h5_file, 'w') as f:

        for crop_index, captions_output in crop_index_to_captions_output.items():
            group = f.create_group(
                f"p{crop_index.person_id}_s{crop_index.frame_id}")

            group.create_dataset('caption_1', data=captions_output.caption_1)
            group.create_dataset('caption_2', data=captions_output.caption_2)


def import_captions_output_from_hdf5(h5_file: Path) -> Dict[CropIndex, FrameOutput]:
    with h5py.File(h5_file, 'r') as hd5_file:
        crop_index_to_captions_output = {
            crop_index_from_filename(filename):
                CaptionsOutput(
                    dataset[CaptionsOutput._fields[0]][...],
                    dataset[CaptionsOutput._fields[1]][...],
            )
            for filename, dataset in hd5_file.items()
        }

    return crop_index_to_captions_output
