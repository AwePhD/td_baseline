from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

import h5py
from irra.model.clip_model import CLIP
import torch
from torch.utils.data import DataLoader

from .data_struct import FrameOutput
from .data_struct import CaptionsOutput, CropIndex
from .models.tokenizer import SimpleTokenizer, tokenize
from .utils import prompt_rm_to_user, extract_int_from_str

H5_CAPTIONS_OUTPUT_FILENAME = "filename_to_captions_output.h5"
H5_CAPTIONS_OUTPUT_FILE = Path.cwd() / "outputs" / H5_CAPTIONS_OUTPUT_FILENAME

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
    token_batch_size: int = 512
) -> Dict[CropIndex, CaptionsOutput]:
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
            tokens_features: torch.Tensor
            tokens_features = model.encode_text(batch_tokens.cuda()).cpu()
        # Prends le token de <END_OF_SEQUENCE> => CLASS TOKEN
        features_text.extend(
            tokens_features[torch.arange(tokens_features.shape[0]), batch_tokens.argmax(dim=-1)]
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
    annotations: pd.DataFrame,
    model: CLIP,
    h5_file: Path = H5_CAPTIONS_OUTPUT_FILE,
) -> None:
    if h5_file.exists():
        if not prompt_rm_to_user(h5_file):
            return
        else:
            h5_file.unlink()

    crop_index_to_captions_outputs = _get_text_features(annotations, model)
    export_caption_features_to_hdf5(crop_index_to_captions_outputs, h5_file)


def export_caption_features_to_hdf5(
    crop_index_to_captions_output: Dict[CropIndex, CaptionsOutput],
    h5_file: Path
) -> None:
    with h5py.File(h5_file, 'w') as f:
        for crop_index, captions_output in crop_index_to_captions_output.items():
            group = f.create_group(f"p{crop_index.person_id}_s{crop_index.frame_id}")

            group.create_dataset('caption_1', data=captions_output.caption_1)
            group.create_dataset('caption_2', data=captions_output.caption_2)


def import_captions_output_from_hdf5(
    h5_file: Path = H5_CAPTIONS_OUTPUT_FILE
) -> Dict[CropIndex, FrameOutput]:
    with h5py.File(h5_file, 'r') as hd5_file:
        crop_index_to_captions_output = {
            CropIndex(
                extract_int_from_str(crop_index_name.split("_")[0]),
                extract_int_from_str(crop_index_name.split("_")[1])
            ):
                CaptionsOutput(
                    captions_output[CaptionsOutput._fields[0]][...],
                    captions_output[CaptionsOutput._fields[1]][...],
                )
            for crop_index_name, captions_output in hd5_file.items()
        }

    return crop_index_to_captions_output