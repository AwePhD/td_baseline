"""Various useful functions"""

from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd

from .data_struct import CropIndex

def gt_bboxes_from_annotations(annotations: pd.DataFrame) -> Dict[CropIndex, np.ndarray]:
    gt_bboxes = (
        annotations[["bbox_x", "bbox_y","bbox_w", "bbox_h" ]]
        [annotations.bbox_w != 0 ]
        .astype(np.int32)
        .copy()
    )
    gt_bboxes['bbox_x_end'] = gt_bboxes.bbox_x + gt_bboxes.pop("bbox_w")
    gt_bboxes['bbox_y_end'] = gt_bboxes.bbox_y + gt_bboxes.pop("bbox_h")

    return {
        CropIndex(person_id, frame_id): gt_bboxes.loc[person_id, frame_id].values
        for person_id, frame_id in gt_bboxes.index
    }

def extract_int_from_str(s: str) -> int:
    return int("".join(c for c in s if c.isdigit()))


def prompt_rm_to_user(h5_file: Path) -> bool:
    print(f"{h5_file.name} already exists.")
    user_input = input("Delete the file (y/N): ")
    return user_input.lower() in ['y', 'yes']
