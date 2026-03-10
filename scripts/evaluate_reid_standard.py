from pathlib import Path

from tdbaseline.config import build_path, get_config
from tdbaseline.eval.reid_standard import evaluate_reid_from_h5


def main():
    config = get_config(Path("./config.yaml"))

    evaluate_reid_from_h5(
        build_path(config["data.annotations"]),
        build_path(config["h5_files.crop_features_from_files"]),
    )


if __name__ == "__main__":
    main()
