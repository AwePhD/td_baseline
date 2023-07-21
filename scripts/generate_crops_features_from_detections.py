from pathlib import Path

from tdbaseline.pstr_output import import_from_hdf5
from tdbaseline.crop_features import (
    compute_bboxes_clip_features_from_detections,
    export_bboxes_clip_features_to_hdf5,
)
from tdbaseline.models.clip import load_clip


def main():
    frame_file_to_detection_output = import_from_hdf5(
        Path('outputs/filename_to_detection.h5'),
        Path(Path.home() / 'data' / 'frames'),
    )

    model = load_clip().eval().cuda()

    frame_file_to_bboxes_clip_features = compute_bboxes_clip_features_from_detections(
        model,
        frame_file_to_detection_output
    )

    output_h5 = Path('outputs/frame_file_to_bboxes_clip_features')
    export_bboxes_clip_features_to_hdf5(
        frame_file_to_bboxes_clip_features, output_h5)


if __name__ == "__main__":
    main()
