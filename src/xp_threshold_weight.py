"""
This module experiments on the value of the threshold and the weight
of the average for similarities computations:

- threshold is used to dismiss detections. A threshold at 100% we only keep
"perfect" detections, although, we will miss a lot of "great" detections.
- The similarities is computed by a mean of two similarities. The first similarities
is between the query text features and the gallery crops features. The second
similarities is between the query crops features and the gallery crops features.
weight is amount of text/image similarities we have, in percentage (from 0% to 100%)
"""
import logging
from typing import Tuple
from pathlib import Path
from contextlib import contextmanager

import numpy as np

from eval import import_data, compute_mean_average_precision
from features_generation import H5_FRAME_OUTPUT_FILE, H5_CAPTIONS_OUTPUT_FILE
from compute_similarities import average

NUMPY_BIN_FILE = Path("outputs", "xp_threshold_weight")

def cool_formatter(to_display: str) -> str:
    return f"{to_display:-^50s}"

def _make_grids(
    min_threshold: float, max_threshold: float, n_threshold: int,
    min_weight: float, max_weight: float, n_weight: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(
        np.linspace(min_threshold, max_threshold, n_threshold),
        np.linspace(min_weight, max_weight, n_weight)
    )

def main():
    logging.basicConfig(level=logging.DEBUG)

    logging.debug(cool_formatter(' Start grid '))
    # Create meshgrid of elements
    min_threshold, max_threshold, n_threshold = .1, .8, 4
    min_weight, max_weight, n_weight = 0, 1, 11
    threshold_grid, weight_grid = _make_grids(
        min_threshold, max_threshold, n_threshold,
        min_weight, max_weight, n_weight,
    )
    logging.info(f"Threshold: {min_threshold}-{max_threshold} in {n_threshold} steps")
    logging.info(f"Weight: {min_weight}-{max_weight} in {n_weight} steps")

    # Load data
    logging.debug(cool_formatter(' Start samples loading '))
    # !! ~2GB of RAM using list
    samples = list(import_data(H5_CAPTIONS_OUTPUT_FILE, H5_FRAME_OUTPUT_FILE))

    # Experiments
    mean_average_precisions = np.zeros_like(weight_grid)
    logging.debug(cool_formatter(' Start XP '))
    for i_weight in range(n_weight):
        for j_threshold in range(n_threshold):
            weight = weight_grid[i_weight, j_threshold]
            threshold = threshold_grid[i_weight, j_threshold]

            compute_similarities = (
                lambda q_txt, q_img, f_img :
                average(q_txt, q_img, f_img, weight)
            )

            mAP = compute_mean_average_precision(samples, compute_similarities, threshold)

            mean_average_precisions[i_weight, j_threshold] = mAP

            logging.info(f"{threshold:2.2%} | {weight:3.2%} | {mAP:2.2%}")

    # Export result
    logging.debug(cool_formatter(' Start exporting '))
    np.save(NUMPY_BIN_FILE ,mean_average_precisions)

if __name__ == "__main__":
    main()