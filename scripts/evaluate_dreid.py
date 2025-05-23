from pathlib import Path

from tdbaseline.config import build_path, get_config
from tdbaseline.detection_reid.eval import evaluate_dreid_from_h5


def main():
    config = get_config(Path("./config.yaml"))

    evaluate_dreid_from_h5(
        build_path(config["data.annotations"]),
        config["eval.detection_reid.threshold"],
        build_path(config["h5_files.detections"]),
    )


if __name__ == "__main__":
    main()
