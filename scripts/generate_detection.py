from pathlib import Path

from tdbaseline.models.pstr import PSTR
from tdbaseline.detections_generation import (
  get_detector_outputs_by_path,
  export_to_hdf5,
  H5_FILE
)

def main():
    model = PSTR()

    frame_file_to_detection = get_detector_outputs_by_path(model)

    export_to_hdf5(frame_file_to_detection, Path.cwd() / "outputs" / H5_FILE)

if __name__ == "__main__":
    main()