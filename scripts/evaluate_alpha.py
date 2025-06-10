from pathlib import Path

import numpy as np
import pandas as pd

from tdbaseline.config import build_path, get_config
from tdbaseline.eval.text_frame import evaluate_text_frame_from_h5


def main():
    config = get_config(Path("./config.yaml"))
    alphas = np.linspace(0, 1, 25)
    metrics_alpha = pd.DataFrame(
        index=pd.RangeIndex(len(alphas)),
        columns=["mAP", "alpha"],
    )

    for i_row, alpha in enumerate(alphas):
        print(f"alpha={alpha}, ", end="")
        (mAP) = evaluate_text_frame_from_h5(
            build_path(config["data.annotations"]),
            config["eval.threshold"],
            alpha,
            build_path(config["h5_files.features_text"]),
            build_path(config["h5_files.detections"]),
            build_path(config["h5_files.crop_features_from_detections"]),
        )
        metrics_alpha.loc[i_row, "mAP"] = mAP
        metrics_alpha.loc[i_row, "alpha"] = alpha

    print(metrics_alpha)
    metrics_alpha.to_parquet("outputs/metrics_alpha.parquet")


if __name__ == "__main__":
    main()
