import evaluate_tdreid
import evaluate_experiments

def main():
    print("Standard evaluations")
    evaluate_tdreid.main()
    print("Experiments' evaluations")
    evaluate_experiments.main()


if __name__ == "__main__":
    main()
