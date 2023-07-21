from tdbaseline.cuhk_sysu_pedes import import_test_annotations
from tdbaseline.models.clip import load_clip
from pathlib import Path
from tdbaseline.captions_features import generate_captions_output_to_hdf5


def main():
    model = load_clip().eval().cuda()
    annotations = import_test_annotations()

    h5_output_file = Path('outputs/crop_index_to_captions_output')
    generate_captions_output_to_hdf5(annotations, model, h5_output_file)


if __name__ == "__main__":
    main()
