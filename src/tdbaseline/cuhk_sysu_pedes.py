from pathlib import Path

import pandas as pd

ANNOTATIONS_FILENAME = "annotations.csv"

INDEXES = ["person_id", "frame_id"]

COLUMN_NAME_TO_DTYPE = {
    "split_sysu": "category",
    "is_hard": bool,
    "bbox_x": pd.UInt16Dtype(),
    "bbox_y": pd.UInt16Dtype(),
    "bbox_w": pd.UInt16Dtype(),
    "bbox_h": pd.UInt16Dtype(),
    "split_pedes": "category",
    "caption_1": object,
    "caption_2": object,
}


def read_annotations_csv(data_folder: Path) -> pd.DataFrame:
    return pd.read_csv(
        str(data_folder / ANNOTATIONS_FILENAME),
        index_col=tuple(INDEXES),
        dtype=COLUMN_NAME_TO_DTYPE,
    )


def import_pedes_test_annotations(data_folder: Path) -> pd.DataFrame:
    """Import PEDES test annotations of the CUHK-SYSU-PEDES to a DataFrame.

    Args:
        data_folder (Path, optional): Folder of the dataset. Defaults to DATA_FOLDER.

    Returns:
        pd.DataFrame: Test annotations of the dataset.
    """

    return read_annotations_csv(data_folder).query("split_pedes == 'test'")


def import_sysu_test_annotations(data_folder: Path) -> pd.DataFrame:
    """Import SYSU test annotations of the CUHK-SYSU-PEDES to a DataFrame.

    Args:
        data_folder (Path, optional): Folder of the dataset. Defaults to DATA_FOLDER.

    Returns:
        pd.DataFrame: Test annotations of the dataset.
    """
    return read_annotations_csv(data_folder).query("split_sysu != 'train'")
