from pathlib import Path

import numpy as np
import pandas as pd

ANNOTATIONS_TRAIN_FILENAME = "annotations_train.csv"
ANNOTATIONS_TEST_FILENAME = "annotations_test.csv"
INDEX_NAMES = ["person_id", "frame_id"]
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
        index_col=tuple(INDEX_NAMES),
        dtype={
            column: dtype
            for column, dtype in COLUMNS_DTYPE.items()
            if column != "split"  # Exclude split column
        },
    )
    test_annotations = pd.read_csv(
        test_annotation_path,
        index_col=tuple(INDEX_NAMES),
        dtype=COLUMNS_DTYPE,
    )

    return train_annotations, test_annotations


def import_test_annotations(data_folder: Path) -> pd.DataFrame:
    """Import test annotations of the CUHK-SYSU-PEDES to a DataFrame.

    Args:
        data_folder (Path, optional): Folder of the dataset. Defaults to DATA_FOLDER.

    Returns:
        pd.DataFrame: Test annotations of the dataset.
    """
    _, annotations = _read_annotations_csv(
        data_folder / ANNOTATIONS_TRAIN_FILENAME,
        data_folder / ANNOTATIONS_TEST_FILENAME,
    )

    return annotations
