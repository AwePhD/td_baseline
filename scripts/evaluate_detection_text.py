from pathlib import Path

from tdbaseline.text_reid.eval import evaluate_from_ground_truth
from tdbaseline.config import get_config


def main():
    config = get_config(Path('./config.yaml'))

    mean_average_precision = evaluate_from_ground_truth(
        config['h5_files']['crop_features_from_files'],
        config['h5_files']['captions_output']
    )
    print(f"mAP: {mean_average_precision:.2%}")


if __name__ == "__main__":
    main()
