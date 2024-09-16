import argparse
from pathlib import Path
from typing import Literal, Union

from typing_extensions import TypeAlias

from tdbaseline.config import get_config
from tdbaseline.text_reid.eval import evaluate

CropOrigin: TypeAlias = Union[
    Literal["files"], Literal["annotations"], Literal["detections"]
]


def _get_crop_origin_from_args() -> CropOrigin:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "crop_origin", type=str, choices=["files", "annotations", "detections"]
    )

    return parser.parse_args().crop_origin


def main():
    config = get_config(Path("./config.yaml"))

    crop_origin = _get_crop_origin_from_args()
    features_image = Path(config["h5_files"]["features_image"][crop_origin])
    features_text = Path(config["h5_files"]["features_text"])

    # NOTE: 64.78 is the target
    mean_average_precision = evaluate(features_image, features_text)

    print(f"(Origin {crop_origin}) mAP: {mean_average_precision:.2%}")


if __name__ == "__main__":
    main()
