from pathlib import Path

from tdbaseline.config import build_path, get_config
from tdbaseline.format_parquet_to_json import format_parquet_to_json


def main():
    config = get_config(Path("./config.yaml"))

    format_parquet_to_json(
        build_path(config["data.annotations"]),
        build_path(config["data.annotations_json"]),
        build_path(config["data.frames_folder"]),
    )


if __name__ == "__main__":
    main()
