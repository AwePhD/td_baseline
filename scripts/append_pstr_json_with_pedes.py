from typing import Dict, List
from pathlib import Path
import json

from PIL import Image

from tdbaseline.cuhk_sysu_pedes import import_pedes_test_annotations
from tdbaseline.config import get_config, build_path

ORIGINAL_PSTR_JSON_FILE = Path.home() / "data" / "sysu" / "annotation" / "test_new.json"
NEW_PSTR_JSON_FILE = (
    Path.home() / "data" / "sysu" / "annotation" / "test_new_pedes.json"
)


def main():
    config = get_config(Path("./config.yaml"))

    # Import JSON that is input in PSTR
    with open(ORIGINAL_PSTR_JSON_FILE, "r", encoding="utf-8") as json_file:
        json_annotations = json.load(json_file)
    json_annotation = json_annotations["images"][0]
    n_images = len(json_annotations["images"])

    # Import annotations from PEDES test split
    pedes_annotations = import_pedes_test_annotations(
        build_path(config["data"]["root_folder"])
    )
    pedes_frame_ids = pedes_annotations.index.get_level_values("frame_id")

    # Get metadata (width, height)
    frames_folder = build_path(config["data"]["frames_folder"])
    frame_id_to_meta_data = {}
    for frame_id in pedes_frame_ids:
        frame_file = frames_folder / f"s{frame_id}.jpg"
        frame = Image.open(frame_file)
        width, height = frame.width, frame.height
        frame_id_to_meta_data[frame_id] = {"width": width, "height": height}

    # Create json 'images' entries
    pedes_json_annotations: List[Dict] = []
    for i, frame_id in enumerate(pedes_frame_ids, n_images):
        # Get frame file
        file_name = f"s{frame_id}.jpg"
        frame = Image.open(frames_folder / file_name)

        pedes_json_annotations.append(
            {
                "file_name": file_name,
                "id": i,
                "width": frame.width,
                "height": frame.height,
            }
        )

    # Append the annotations
    json_annotations["images"].extend(pedes_json_annotations)

    # Write new json
    with open("./test_new_pedes.json", "w+t", encoding="utf-8") as new_json_file:
        json.dump(json_annotations, new_json_file, indent=2)


if __name__ == "__main__":
    main()
