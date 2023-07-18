from tdbaseline.eval import (
  import_data,
  compute_mean_average_precision,
)
from tdbaseline.compute_similarities import pstr_similarities, build_baseline_similarities
def main():
    compute_similarities = pstr_similarities
    # weight_of_text_features = 1
    # compute_similarities = build_baseline_similarities(weight_of_text_features)

    samples = import_data()

    mean_average_precision = compute_mean_average_precision(samples, compute_similarities)

    print(f"mAP: {mean_average_precision:.2%}")


if __name__ == "__main__":
    main()
