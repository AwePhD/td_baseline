# generate_captions_output.py edition
from pathlib import Path

from tdbaseline.config import build_path, get_config
from tdbaseline.text_features import generate_text_features_to_h5


def main():
    config = get_config(Path('./config.yaml'))

    generate_text_features_to_h5(
        build_path(config['models']['clip']['weight_path']),
        build_path(config['data']['annotations']),
        config['process']['tokens_batch_size'],
        build_path(config['models']['clip']['vocab_path']),
        build_path(config['h5_files']['features_text'])
    )


if __name__ == "__main__":
    main()
