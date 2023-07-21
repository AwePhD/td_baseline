from pathlib import Path

from tdbaseline.config import build_path, get_config
from tdbaseline.captions_features import generate_captions_output_to_hdf5


def main():
    config = get_config(Path('./config.yaml'))

    generate_captions_output_to_hdf5(
        build_path(config['models']['clip']['weight_path']),
        build_path(config['data']['root_folder']),
        config['process']['tokens_batch_size'],
        build_path(config['models']['clip']['vocab_path']),
        build_path(config['h5_files']['captions_output'])
    )


if __name__ == "__main__":
    main()
