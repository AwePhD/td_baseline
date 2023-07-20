from pathlib import Path

from tdbaseline.models.clip import load_clip
from tdbaseline.cuhk_sysu_pedes import import_test_annotations, FRAME_FOLDER
from tdbaseline.pstr_output import import_from_hdf5, H5_FILE
from tdbaseline.captions_features import generate_captions_output_to_hdf5
from tdbaseline.detection_reid.clip_features import (
    assert_detection_output_and_annotations_compatibility,
    generate_frame_output_to_hdf5,
)


def main():
    model = load_clip().eval().cuda()

    annotations = import_test_annotations()

    generate_captions_output_to_hdf5(
        annotations, model, Path.cwd() / 'outputs' / 'captions')

    frame_file_to_detection = import_from_hdf5(H5_FILE, FRAME_FOLDER)
    # Assure that annotations (used for evaluation) and the output
    # of the model (used for detection compute) have the same frame files
    assert_detection_output_and_annotations_compatibility(
        annotations, frame_file_to_detection)

    generate_frame_output_to_hdf5(
        frame_file_to_detection, model, Path.cwd() / 'outputs' / 'features')


if __name__ == "__main__":
    main()
