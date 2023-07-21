import generate_captions_output
import generate_detection_output
import generate_crops_features_from_files
import generate_crops_features_from_annotations
import generate_bboxes_clip_features_from_detections


def main():
    generate_captions_output.main()
    generate_detection_output.main()
    generate_crops_features_from_files.main()
    generate_crops_features_from_annotations.main()
    generate_bboxes_clip_features_from_detections.main()


if __name__ == "__main__":
    main()
