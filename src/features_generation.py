from typing import Dict, Tuple, NamedTuple, List
from pathlib import Path

import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from irra.model.clip_model import CLIP

from tokenizer import SimpleTokenizer, tokenize
from clip import load_clip, IMAGE_SIZE
from cuhk_sysu_pedes import read_annotations_csv
from detections_generation import import_from_hdf5, H5_FILENAME, DetectionOutput

DATA_FOLDER = Path.home() / "data"
FRAME_FOLDER = DATA_FOLDER / "frames"
H5_FILE = Path.cwd() / "outputs" / H5_FILENAME
TOKEN_BATCH_SIZE = 512
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
H5_FRAME_OUTPUT_FILENAME = "filename_to_frame_output"
H5_CAPTIONS_OUTPUT_FILENAME = "filename_to_captions_output"
H5_FRAME_OUTPUT_FILE = Path.cwd() / "outputs" / H5_FRAME_OUTPUT_FILENAME
H5_CAPTIONS_OUTPUT_FILE = Path.cwd() / "outputs" / H5_CAPTIONS_OUTPUT_FILENAME

class FrameOutput(NamedTuple):
    # (100,)
    scores: torch.Tensor
    # (100, 4)
    bboxes: torch.Tensor
    # (100, 512)
    features: torch.Tensor

class CaptionsOutput(NamedTuple):
    caption_1: torch.Tensor
    caption_2: torch.Tensor

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
            tokens_features = model.encode_text(batch_tokens.cuda()).cpu()
        # Prends le token de <END_OF_SEQUENCE> => CLASS TOKEN
        features_text.extend(tokens_features[
             torch.arange(tokens_features.shape[0]), batch_tokens.argmax(dim=-1)
        ].float())
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
    bboxes: torch.Tensor
) -> torch.Tensor:
    frame = Image.open(str(frame_file))
    crops = [
        frame.crop((
            int(bbox[0]),
            int( bbox[1]),
            int(bbox[2]-bbox[0]),
            int(bbox[3]-bbox[1])
        ))
        for bbox in bboxes
    ]

    crops_preprocessed = torch.stack([preprocess_crop(crop) for crop in crops])

    with torch.no_grad():
        crops_features = model.encode_image(crops_preprocessed.cuda()).cpu()

    return crops_features

def _get_image_features(
    model: CLIP,
    frame_file_to_detection: Dict[Path, DetectionOutput],
) -> Dict[Path, torch.Tensor]:
    # Take one frame at time and output the images features
    return {
        frame_file: _compute_features_from_one_frame(model, frame_file, detection.bboxes)
        for frame_file, detection in frame_file_to_detection.items()
    }

def export_frame_output_to_hdf5(
    frame_file_to_frame_output: Dict[Path, FrameOutput],
    h5_file: Path
) -> None:
    with h5py.File(h5_file, 'w') as f:
        for frame_file, detection_output in frame_file_to_frame_output.items():
            group = f.create_group(frame_file.name)

            group.create_dataset('scores', data=detection_output.scores)
            group.create_dataset('bboxes', data=detection_output.bboxes)
            group.create_dataset('features', data=detection_output.features)

def export_caption_features_to_hdf5(
    crop_index_to_captions_output: Dict[CropIndex, CaptionsOutput],
    h5_file: Path
) -> None:
    with h5py.File(h5_file, 'w') as f:
        for crop_index, captions_output in crop_index_to_captions_output.items():
            group = f.create_group(crop_index)

            group.create_dataset('caption_1', data=captions_output.caption_1)
            group.create_dataset('caption_2', data=captions_output.caption_2)



def main():
    model = load_clip()
    model.cuda()

    annotations = _import_annotations()

    crop_index_to_captions_outputs = _get_text_features(annotations, model)

    frame_file_to_detection= import_from_hdf5(H5_FILE, FRAME_FOLDER)
    frame_to_image_features = _get_image_features(model, frame_file_to_detection)
    frame_file_to_full_output = {
        frame_file: FrameOutput(
            frame_file_to_detection[frame_file].scores,
            frame_file_to_detection[frame_file].bboxes,
            frame_to_image_features[frame_file]
        )
        for frame_file in frame_file_to_detection
    }

    # Assure that annotations (used for evaluation) and the output
    # of the model (used for detection compute) are the same
    annotations_frame_ids = set(
        f"s{frame_id}.jpg"
        for frame_id in annotations.index.get_level_values("frame_id").unique()
    )
    h5_frame_ids = set(
        frame_file.name
        for frame_file in frame_file_to_detection
    )
    assert annotations_frame_ids == h5_frame_ids

    # Export them
    export_frame_output_to_hdf5(frame_file_to_full_output, H5_FRAME_OUTPUT_FILE)
    export_caption_features_to_hdf5(crop_index_to_captions_outputs, H5_CAPTIONS_OUTPUT_FILE)

if __name__ == "__main__":
    main()