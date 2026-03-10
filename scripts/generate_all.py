import generate_crop_index_to_features_text
import generate_features_image_annotations
import generate_features_image_crop
import generate_features_image_detections
import generate_frame_id_to_detections
import generate_json_mm


def main():
    # Data preparation
    print("Generate MMLab's JSON annotations input from CSU annotations.")
    generate_json_mm.main()

    # CLIP captions
    print("Generate map crop index -> features text")
    generate_crop_index_to_features_text.main()

    # PSTR detections
    print("Generate map frame ID -> PSTR outputs")
    generate_frame_id_to_detections.main()

    print("Generate CLIP image features from files crops")
    generate_features_image_crop.main()
    print("Generate CLIP image features from annotations crops")
    generate_features_image_annotations.main()
    print("Generate CLIP image features from automatic crops (detections)")
    generate_features_image_detections.main()


if __name__ == "__main__":
    main()
