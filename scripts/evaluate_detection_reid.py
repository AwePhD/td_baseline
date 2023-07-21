from pathlib import Path

from tdbaseline.detection_reid.eval import (
    import_data,
    compute_mean_average_precision,
)
# pylint: disable=unused-import
from tdbaseline.detection_reid.compute_similarities import (
    pstr_similarities, build_baseline_similarities)
from tdbaseline.config import get_config


def main():
    config = get_config(Path('./config.yaml'))

    compute_similarities = pstr_similarities
    # weight_of_text_features = 0
    # compute_similarities = build_baseline_similarities(weight_of_text_features)

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
