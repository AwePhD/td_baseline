from pathlib import Path
from tdbaseline.config import build_path, get_config
from tdbaseline.eval.text_frame import evaluate_text_frame_from_h5


def main():
    config = get_config(Path("./config.yaml"))

    evaluate_text_frame_from_h5(
        build_path(config["data.annotations"]),
        config["eval.threshold"],
        config["eval.alpha"],
        build_path(config["h5_files.features_text"]),
        build_path(config["h5_files.detections"]),
        build_path(config["h5_files.crop_features_from_detections"]),
    )


if __name__ == "__main__":
    main()

