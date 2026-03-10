import json
from pathlib import Path
from typing import List, TypedDict

import pandas as pd
from PIL import Image

from .utils import confirm_generation


class AnnotationsCategory(TypedDict):
    id: int
    name: str
    supercategory: str


class AnnotationsImage(TypedDict):
    file_name: str
    id: int
    width: int
    height: int


class AnnotationsDetection:
    area: int
    # 4 int
    bbox: List[int]
    category_id: int
    id: int
    image_id: int
    iscrowd: int
    person_id: int
    segmentation: List


class AnnotationsMM(TypedDict):
    categories: List[AnnotationsCategory]
    images: List[AnnotationsImage]
    annotations: List[AnnotationsDetection]


# Mandatory fields' values
CATEGORY_ID = 1
CATEGORY_NAME = "person"
SUPERCATEGORY = "object"


def _format_image(
    i: int, frame_id: int, frames_folder: Path
) -> AnnotationsImage:
    file_name = f"s{frame_id}.jpg"

    with Image.open(frames_folder / file_name) as image:
        width = image.width
        height = image.height

    return AnnotationsImage(
        id=i, file_name=file_name, width=width, height=height
    )


def _format_to_json(
    annotations: pd.DataFrame, frames_folder: Path
) -> AnnotationsMM:
    frame_ids = annotations.frame_id.unique()
    images = [
        _format_image(i, frame_id, frames_folder)
        for i, frame_id in enumerate(frame_ids)
    ]

    return AnnotationsMM(
        categories=[
            AnnotationsCategory(
                id=CATEGORY_ID, name=CATEGORY_NAME, supercategory=SUPERCATEGORY
            )
        ],
        images=images,
        annotations=[],
    )


def _export_json(
    annotations_mmlab: AnnotationsMM, annotations_json: Path
) -> None:
    with open(annotations_json, mode="w", encoding="utf-8") as file:
        json.dump(annotations_mmlab, file, indent=2)


def format_parquet_to_json(
    annotations_file: Path, annotations_json: Path, frames_folder: Path
) -> None:
    """Get MMLab's json annotations for test annotations from parquet CSU file.

    MMLab original code first generates the outputs and then evaluate.
    Thus, during the generation the input annotations are only a list of
    image to get as inputs.
    """
    assert annotations_file.exists() and annotations_file.is_file()
    assert frames_folder.exists() and frames_folder.is_dir()
    if not confirm_generation(annotations_json):
        return

    annotations = pd.read_parquet(annotations_file).reset_index()

    annotations_mm = _format_to_json(annotations, frames_folder)

    _export_json(annotations_mm, annotations_json)
