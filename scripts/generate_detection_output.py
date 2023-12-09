from pathlib import Path
from tdbaseline.config import build_path, get_config

from tdbaseline.pstr_output import generate_detection_output_to_hdf5


def main():
    config = get_config(Path("./config.yaml"))

    generate_detection_output_to_hdf5(
        build_path(config["models"]["pstr"]["config_path"]),
        build_path(config["models"]["pstr"]["weight_path"]),
        build_path(config["h5_files"]["detection_output"]),
    )


if __name__ == "__main__":
    main()
