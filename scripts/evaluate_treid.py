from pathlib import Path

from tdbaseline.config import get_config, build_path
from tdbaseline.text_reid.eval import evaluate_treid_from_h5


def main():
    config = get_config(Path("./config.yaml"))

    evaluate_treid_from_h5(
        build_path(config["data.annotations"]),
        build_path(config["h5_files.features_text"]),
        build_path(config["h5_files.crop_features_from_files"]),
    )


if __name__ == "__main__":
    main()
