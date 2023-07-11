from typing import Dict, Tuple, List
from pathlib import Path

import h5py
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from irra.model.clip_model import CLIP
from tqdm import tqdm

from tokenizer import SimpleTokenizer, tokenize
from clip import load_clip, IMAGE_SIZE
from cuhk_sysu_pedes import read_annotations_csv
from detections_generation import import_from_hdf5, H5_FILENAME, DetectionOutput
from data_struct import (CropIndex, FrameOutput, CaptionsOutput)

DATA_FOLDER = Path.home() / "data"
FRAME_FOLDER = DATA_FOLDER / "frames"
H5_FILE = Path.cwd() / "outputs" / H5_FILENAME
TOKEN_BATCH_SIZE = 512
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
H5_FRAME_OUTPUT_FILENAME = "filename_to_frame_output.h5"
H5_CAPTIONS_OUTPUT_FILENAME = "filename_to_captions_output.h5"
H5_FRAME_OUTPUT_FILE = Path.cwd() / "outputs" / H5_FRAME_OUTPUT_FILENAME
H5_CAPTIONS_OUTPUT_FILE = Path.cwd() / "outputs" / H5_CAPTIONS_OUTPUT_FILENAME

def _import_annotations(data_folder: Path = DATA_FOLDER) -> pd.DataFrame:
    _, annotations = read_annotations_csv(
        data_folder / "annotations_train.csv",
        data_folder / "annotations_test.csv",
    )

    return annotations


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
            tokens_features[:, batch_tokens.argmax(dim=-1)]
            .float()
            .numpy()
        )
    assert len(crop_indexes) == len(features_text)

    n_samples = len(crop_indexes)
    return {
        crop_indexes[i]: CaptionsOutput(features_text[i], features_text[i+1])
        for i in range(0, n_samples, 2)
    }

preprocess_crop = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])

def _compute_features_from_one_frame(
    model: CLIP,
    frame_file: Path,
    bboxes: np.ndarray,
) -> np.ndarray:
    frame = Image.open(str(frame_file))
    # crop method perfoms an integer approximation
    crops = [ frame.crop(bbox) for bbox in bboxes ]

    crops_preprocessed = torch.stack([preprocess_crop(crop) for crop in crops])

    with torch.no_grad():
        crops_features: torch.Tensor
        crops_features = (
            model.encode_image(crops_preprocessed.cuda())
            # Last layer + send to CPU
            [:, 0, :].cpu()
        )

    return crops_features.numpy()

def _compute_and_frame_output(
    model: CLIP,
    frame_file_to_detection: Dict[Path, DetectionOutput],
) -> Dict[Path, FrameOutput]:
    return {
        frame_file: FrameOutput(
            detection.scores,
            detection.bboxes,
            detection.features_pstr,
            _compute_features_from_one_frame(model, frame_file, detection.bboxes),
        )
        for frame_file, detection in tqdm(frame_file_to_detection.items())
    }

def export_caption_features_to_hdf5(
    crop_index_to_captions_output: Dict[CropIndex, CaptionsOutput],
    h5_file: Path
) -> None:
    with h5py.File(h5_file, 'w') as f:
        for crop_index, captions_output in crop_index_to_captions_output.items():
            group = f.create_group(f"p{crop_index.person_id}_s{crop_index.frame_id}")

            group.create_dataset('caption_1', data=captions_output.caption_1)
            group.create_dataset('caption_2', data=captions_output.caption_2)


def export_frame_output_to_hdf5(
    frame_file_to_frame_output: Dict[Path, FrameOutput],
    h5_file: Path
):
    if h5_file.exists():
        h5_file.unlink()

    with h5py.File(h5_file, 'w') as f:
        for frame_file, frame_output in frame_file_to_frame_output.items():

            group = f.create_group(frame_file.name)

            group.create_dataset('scores', data=frame_output.scores)
            group.create_dataset('bboxes', data=frame_output.bboxes)
            group.create_dataset('features_pstr', data=frame_output.features_pstr)
            group.create_dataset('features_clip', data=frame_output.features_clip)

def _assert_detection_output_and_annotations_compatibility(
    annotations: pd.DataFrame,
    frame_file_to_detection: Dict[Path, DetectionOutput]
) -> None:
    annotations_frame_ids = set(
        f"s{frame_id}.jpg"
        for frame_id in annotations.index.get_level_values("frame_id").unique()
    )
    h5_frame_ids = set(
        frame_file.name
        for frame_file in frame_file_to_detection
    )
    assert annotations_frame_ids == h5_frame_ids

def _prompt_rm_to_user(h5_file: Path) -> bool:
    print(f"{h5_file.name} already exists.")
    user_input = input("Delete the file (y/N): ")
    return user_input.lower() in ['y', 'yes']


def _generate_captions_output_to_hdf5(
    annotations: pd.DataFrame,
    model: CLIP,
    h5_file: Path = H5_CAPTIONS_OUTPUT_FILE,
) -> None:
    if h5_file.exists():
        if not _prompt_rm_to_user(h5_file):
            return
        else:
            h5_file.unlink()

    crop_index_to_captions_outputs = _get_text_features(annotations, model)
    export_caption_features_to_hdf5(crop_index_to_captions_outputs, h5_file)


def _generate_frame_output_to_hdf5(
    frame_file_to_detection: Dict[Path, DetectionOutput],
    model: CLIP,
    h5_file: Path = H5_FRAME_OUTPUT_FILE,
) -> None:
    if h5_file.exists():
        if not _prompt_rm_to_user(h5_file):
            return
        else:
            h5_file.unlink()

    frame_file_to_frame_output = _compute_and_frame_output(model, frame_file_to_detection)
    export_frame_output_to_hdf5(frame_file_to_frame_output, h5_file)


def main():
    model = load_clip().eval().cuda()

    annotations = _import_annotations()

    _generate_captions_output_to_hdf5(annotations, model)

    frame_file_to_detection = import_from_hdf5(H5_FILE, FRAME_FOLDER)
    # Assure that annotations (used for evaluation) and the output
    # of the model (used for detection compute) have the same frame files
    _assert_detection_output_and_annotations_compatibility(annotations, frame_file_to_detection)

    _generate_frame_output_to_hdf5(frame_file_to_detection, model)

if __name__ == "__main__":
    main()

def _extract_int_from_str(s: str) -> int:
    return int("".join(c for c in s if c.isdigit()))

def import_frame_output_from_hdf5(h5_file: Path) -> Dict[int, FrameOutput]:
    with h5py.File(h5_file, 'r') as hd5_file:
        frame_id_to_frame_output = {
            _extract_int_from_str(frame_filename):
                FrameOutput(
                    frame_output[FrameOutput._fields[0]][...],
                    frame_output[FrameOutput._fields[1]][...],
                    frame_output[FrameOutput._fields[2]][...],
                    frame_output[FrameOutput._fields[3]][...],
                )
            for frame_filename, frame_output in hd5_file.items()
        }

    return frame_id_to_frame_output

def import_captions_output_from_hdf5(h5_file: Path) -> Dict[CropIndex, FrameOutput]:
    with h5py.File(h5_file, 'r') as hd5_file:
        crop_index_to_captions_output = {
            CropIndex(
                _extract_int_from_str(crop_index_name.split("_")[0]),
                _extract_int_from_str(crop_index_name.split("_")[1])
            ):
                CaptionsOutput(
                    captions_output[CaptionsOutput._fields[0]][...],
                    captions_output[CaptionsOutput._fields[1]][...],
                )
            for crop_index_name, captions_output in hd5_file.items()
        }

    return crop_index_to_captions_output