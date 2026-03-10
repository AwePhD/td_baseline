import evaluate_dreid
import evaluate_textonly
import evaluate_treid
import evaluate_text_frame


def main():
    print("1. TReID evaluation :: ", end="")
    evaluate_treid.main()
    print("2. DReID evaluation :: ", end="")
    evaluate_dreid.main()
    print("3. text-only evaluation :: ", end="")
    evaluate_textonly.main()
    print("4. text+frame evaluation")
    evaluate_text_frame.main()


if __name__ == "__main__":
    main()
