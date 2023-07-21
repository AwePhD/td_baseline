from pathlib import Path
from tdbaseline.text_reid.crop_features_from_dataset import generate_crop_features_from_files
from tdbaseline.cuhk_sysu_pedes import DATA_FOLDER


def main():
    crops_folder = DATA_FOLDER / 'test_query'
    model_weight = Path.home() / 'models' / 'clip_finetuned' / 'clip_finetune.pth'
    h5_file = Path('./outputs/crop_index_to_crop_features_from_files.h5')

    generate_crop_features_from_files(crops_folder, model_weight, h5_file)


if __name__ == "__main__":
    main()
