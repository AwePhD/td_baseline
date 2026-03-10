# Experiments

We conduct experiements to compare the TD-Baseline results to the T-Baseline and the D-Baseline results.

- Experiments in D-ReID framework to compare D-Baseline and TD-Baseline
    - [`weight_baseline`](weight_baseline.py): Try different weights for the text features of the TD-Baseline. `mAP(text_weight)`
    - [`threshold_weight`](threshold_weight.py): Try different weights AND evaluation IoU thresholds at the same time. `mAP(text_weight, iou_threshold)`