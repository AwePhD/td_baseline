import evaluate_treid_annotations


def main():
    print("1. TReID evaluation (from annotations) :: ", end="")
    evaluate_treid_annotations.main()
    print("1. text-only evaluation (only ReID TP) :: ", end="")
    evaluate_treid_annotations.main()


if __name__ == "__main__":
    main()
