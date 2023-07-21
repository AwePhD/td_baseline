from pathlib import Path
from tdbaseline.text_reid.images_features import from_crops_files,  export_to_hdf5
from tdbaseline.cuhk_sysu_pedes import DATA_FOLDER


def main():
    crops_folder = DATA_FOLDER / 'test_query'

    crop_index_to_features = from_crops_files(crops_folder)

    h5_file = Path('.', 'outputs', 'crop_index_to_clip_features_files')
    export_to_hdf5(crop_index_to_features, h5_file)


if __name__ == "__main__":
    main()