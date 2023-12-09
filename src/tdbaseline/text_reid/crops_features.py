from typing import Dict
from pathlib import Path

import numpy as np

from tdbaseline.data_struct import CropIndex


def export_crops_features_to_hdf5(
    crop_index_to_features: Dict[CropIndex, np.ndarray], h5_file: Path
):
    with h5py.File(h5_file, "w") as f:
        for crop_index, features in crop_index_to_features.items():
            group = f.create_group(f"p{crop_index.person_id}_s{crop_index.frame_id}")

            group.create_dataset("features", data=features)
