from pathlib import Path

from tdbaseline.crop_features import generate_bboxes_clip_features_from_detections
from tdbaseline.config import build_path, get_config


def main():
    config = get_config(Path("./config.yaml"))

    generate_bboxes_clip_features_from_detections(
        build_path(config["models"]["clip"]["weight_path"]),
        build_path(config["data"]["frames_folder"]),
        build_path(config["h5_files"]["detection_output"]),
        config["process"]["frames_batch_size"],
        config["process"]["num_workers"],
        build_path(config["h5_files"]["bboxes_clip_features"]),
    )


if __name__ == "__main__":
    main()
