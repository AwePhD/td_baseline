from pathlib import Path
from tdbaseline.text_reid.images_features import (
    from_annotations,  export_to_hdf5)


def main():
    crop_index_to_features = from_annotations()

    h5_file = Path(
        '.', 'outputs', 'crop_index_to_clip_features_annotations')
    export_to_hdf5(crop_index_to_features, h5_file)


if __name__ == "__main__":
    main()
