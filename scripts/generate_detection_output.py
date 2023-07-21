from pathlib import Path

from tdbaseline.models.pstr import PSTR
from tdbaseline.pstr_output import generate_detection_output_to_hdf5


def main():
    config_file = Path('./configs/pstr/tdbaseline.py')
    weight_file = Path('~/models/pstr_resnet_cuhk/pstr_r50_cuhk.pth')
    h5_file = Path('./outputs/frame_file_to_detection_output.h5')

    generate_detection_output_to_hdf5(config_file, weight_file, h5_file)


if __name__ == "__main__":
    main()
