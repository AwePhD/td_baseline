import argparse
from pathlib import Path

from tdbaseline.text_reid.eval import evaluate_from_ground_truth
from tdbaseline.config import get_config

def _get_crop_origin_from_args() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("crop_origin", type=str, choices=["files", "annotations"])

    return parser.parse_args().crop_origin


def main():
    config = get_config(Path('./config.yaml'))

    crop_origin = _get_crop_origin_from_args()
    crop_features_file = Path(
        config['h5_files'][f'crop_features_from_{crop_origin}']
    )

    mean_average_precision = evaluate_from_ground_truth(
        crop_features_file,
        Path(config['h5_files']['captions_output'])
    )
    print(f"mAP: {mean_average_precision:.2%}")


if __name__ == "__main__":
    main()

