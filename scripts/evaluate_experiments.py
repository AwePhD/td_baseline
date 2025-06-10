import evaluate_alpha
import evaluate_dreid_clip
import evaluate_reid_standard
import evaluate_textonly_tponly
import evaluate_treid_annotations
import evaluate_treid_tponly


def main():
    print("1.a TReID evaluation (from annotations = perfect detection) :: ")
    evaluate_treid_annotations.main()
    print("1.b text-only evaluation (only ReID TP) :: ")
    evaluate_textonly_tponly.main()
    print("1.c TReID evaluation (only ReID TP) :: ")
    evaluate_treid_tponly.main()
    print("1.d DReID with clip")
    evaluate_dreid_clip.main()

    print("2.a Evaluate ReID standard :: ")
    evaluate_reid_standard.main()
    print("2.b Evaluate alpha")
    evaluate_alpha.main()


if __name__ == "__main__":
    main()
