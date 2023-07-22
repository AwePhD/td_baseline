from pathlib import Path

from tdbaseline.config import get_config, build_path
from tdbaseline.text_reid.crop_features_from_dataset import generate_crop_features_from_files


def main():
    config = get_config(Path('./config.yaml'))

    generate_crop_features_from_files(
        build_path(config['models']['clip']['weight_path']),
        build_path(config['data']['crops_folder']),
        config['process']['crops_batch_size'],
        config['process']['num_workers'],
        build_path(config['h5_files']['crop_features_from_files'])
    )


if __name__ == "__main__":
    main()
