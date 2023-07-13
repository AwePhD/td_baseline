"""
This module experiments on the value of the weights of the text features
for similarities computations:

- The similarities is computed by a mean of two similarities. The first similarities
is between the query text features and the gallery crops features. The second
similarities is between the query crops features and the gallery crops features.
weight is amount of text/image similarities we have, in percentage (from 0% to 100%)
"""
from pathlib import Path
import logging

import numpy as np

from eval import import_data, compute_mean_average_precision
from compute_similarities import build_baseline_similarities
from features_generation import H5_FRAME_OUTPUT_FILE, H5_CAPTIONS_OUTPUT_FILE

NUMPY_BIN_FILE = Path("outputs", "xp_weights")

def main():
  logging.basicConfig(level=logging.DEBUG)

  # Create meshgrid of elements
  min_weight, max_weight, n_weights = 0, 1, 21
  weights_of_text_features = np.linspace(min_weight, max_weight, n_weights)
  logging.info(f"Weight: {min_weight}-{max_weight} in {n_weights} steps")

  # Load data
  # !! +2GB of RAM using list
  samples = list(import_data(H5_CAPTIONS_OUTPUT_FILE, H5_FRAME_OUTPUT_FILE))

  # Experiments
  mean_average_precisions = np.zeros_like(weights_of_text_features)
  for i, weight in enumerate(weights_of_text_features):
    baseline_similarities = build_baseline_similarities(weight)

    mAP = compute_mean_average_precision(samples, baseline_similarities)
    mean_average_precisions[i] = mAP

    logging.info(f"{weight:3.2%} | {mAP:2.2%}")

  # Export result
  np.save(NUMPY_BIN_FILE ,mean_average_precisions)

if __name__ == "__main__":
  main()
