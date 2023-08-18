import argparse
from pathlib import Path

from tdbaseline.detection_reid.eval import (
    import_data,
    compute_mean_average_precision,
)
from tdbaseline.detection_reid.compute_similarities import (
    pstr_similarities, build_baseline_similarities)
from tdbaseline.config import get_config

def _get_evaluation_type_from_args() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["detection", "text_detection"], required=True)

    return parser.parse_args().type



def main():
    evaluation_type = _get_evaluation_type_from_args()
    config = get_config(Path('./config.yaml'))

    compute_similarities = (
        pstr_similarities
        if evaluation_type == "detection"
        else build_baseline_similarities(config['eval']['detection_reid']['weight'])
    )

    output_folder = Path('outputs')
    samples = import_data(
        Path(config['data']['root_folder']),
        Path(config['data']['frames_folder']),
        output_folder / 'crop_index_to_captions_output.h5',
        output_folder / 'frame_file_to_detection_output.h5',
        output_folder / 'frame_id_to_bboxes_clip_features.h5',
    )

    mean_average_precision = compute_mean_average_precision(
        samples,
        compute_similarities,
        config['eval']['detection_reid']['threshold'],
    )

    print(f"mAP: {mean_average_precision:.2%}")


if __name__ == "__main__":
    main()
