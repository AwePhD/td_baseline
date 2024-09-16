from pathlib import Path
from typing import Dict

import numpy as np

from ..data_struct import CropIndex


def import_features_image_from_h5(
    h5_file: Path,
) -> Dict[CropIndex, np.ndarray]:
    raise NotImplementedError


def export_features_image_to_h5(
    crop_index_to_features_image: Dict[CropIndex, np.ndarray],
    h5_file: Path,
) -> None:
    raise NotImplementedError
