from pathlib import Path

from tdbaseline.config import build_path, get_config
from tdbaseline.pstr_output import generate_detections_to_h5


def main():
    config = get_config(Path("./config.yaml"))

    generate_detections_to_h5(
        build_path(config["models"]["pstr"]["config_path"]),
        build_path(config["models"]["pstr"]["weight_path"]),
        build_path(config["data"]["annotations_json"]),
        build_path(config["data"]["root_folder"]),
        build_path(config["h5_files"]["detections"]),
    )


if __name__ == "__main__":
    main()
