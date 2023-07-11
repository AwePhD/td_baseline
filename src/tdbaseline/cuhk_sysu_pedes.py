from pathlib import Path

import numpy as np
import pandas as pd

DATA_FOLDER = Path.home() / "data"
FRAME_FOLDER = DATA_FOLDER / "frames"
ANNOTATIONS_DATAFRAME = {
    # Name of the index for person IDs and frame IDs
    "INDEX": {
        "PERSON ID": "person_id",
        "FRAME ID": "frame_id",
    },
    # PEDES columns
    "PEDES COLUMNS": ["split", "caption_1", "caption_2"],
    # SYSU columns
    "SYSU COLUMNS": {
        "TRAIN": ["is_hard", "bbox_x", "bbox_y", "bbox_w", "bbox_h"],
        "TEST": ["type", "is_hard", "bbox_x", "bbox_y", "bbox_w", "bbox_h"],
    },
    "NAME COLUMNS": [""],
    # In Pedes, category of the sample in the test split.
    "TYPE VALUES": {"QUERY": "query", "GALLERY": "gallery"},
    "SPLIT TYPE VALUES": {"TRAIN": "train", "TEST": "test"},
}

COLUMNS_DTYPE = {
    "is_hard": bool,
    "type": "category",
    "bbox_x": np.uint16,
    "bbox_y": np.uint16,
    "bbox_w": np.uint16,
    "bbox_h": np.uint16,
    "caption_1": object,
    "caption_2": object,
}


def _read_annotations_csv(
    train_annotation_path: Path,
    test_annotation_path: Path,
):
    train_annotations = pd.read_csv(
        train_annotation_path,
        index_col=tuple(ANNOTATIONS_DATAFRAME["INDEX"].values()),
        dtype={
            column: dtype
            for column, dtype in COLUMNS_DTYPE.items()
            if column != "split"  # Exclude split column
        },
    )
    test_annotations = pd.read_csv(
        test_annotation_path,
        index_col=tuple(ANNOTATIONS_DATAFRAME["INDEX"].values()),
        dtype=COLUMNS_DTYPE,
    )

    return train_annotations, test_annotations


def import_test_annotations(data_folder: Path = DATA_FOLDER) -> pd.DataFrame:
    """Import test annotations of the CUHK-SYSU-PEDES to a DataFrame.

    Args:
        data_folder (Path, optional): Folder of the dataset. Defaults to DATA_FOLDER.

    Returns:
        pd.DataFrame: Test annotations of the dataset.
    """
    _, annotations = _read_annotations_csv(
        data_folder / "annotations_train.csv",
        data_folder / "annotations_test.csv",
    )

    return annotations

