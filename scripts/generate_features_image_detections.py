from pathlib import Path

from tdbaseline.config import build_path, get_config
from tdbaseline.crop_features.from_detections import (
    generate_crop_features_from_detections,
)


def main():
    config = get_config(Path("./config.yaml"))

    generate_crop_features_from_detections(
        build_path(config["models"]["clip"]["weight_path"]),
        build_path(config["h5_files"]["detections"]),
        build_path(config["data"]["frames_folder"]),
        config["process"]["frames_batch_size"],
        config["process"]["num_workers"],
        build_path(config["h5_files"]["crop_features_from_detections"]),
    )


if __name__ == "__main__":
    main()
