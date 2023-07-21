from pathlib import Path

from tdbaseline.crop_features import generate_bboxes_clip_features_from_detections


def main():
    model_weight = Path.home() / 'models' / 'clip_finetuned' / 'clip_finetune.pth'
    frames_folder = Path(Path.home() / 'data' / 'frames')
    h5_file_detection_output = Path(
        'outputs/frame_file_to_detection_output.h5')
    batch_size = 3
    num_workers = 4
    h5_output_file = Path('outputs/frame_id_to_bboxes_clip_features.h5')

    generate_bboxes_clip_features_from_detections(
        model_weight,
        frames_folder,
        h5_file_detection_output,
        batch_size,
        num_workers,
        h5_output_file
    )


if __name__ == "__main__":
    main()
