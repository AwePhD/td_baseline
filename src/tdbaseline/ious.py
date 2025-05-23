from typing import Optional

import numpy as np


def compute_ious(
    output_bboxes: np.ndarray,
    gt_bbox: np.ndarray,
) -> np.ndarray:
    """Compute IoUs between bboxes from model and its GT.

    output_bboxes (np.ndarray): (N_BBOXES, 4) tensor of bboxes from model
    gt_bbox (np.ndarray): (4,) Ground Truth bbox from annotation
    return: the index of the best matching (highest IoU) model bbox
    """
    # 1. calculate the inters coordinate
    ixmin = np.maximum(output_bboxes[:, 0], gt_bbox[0])
    ixmax = np.minimum(output_bboxes[:, 2], gt_bbox[2])
    iymin = np.maximum(output_bboxes[:, 1], gt_bbox[1])
    iymax = np.minimum(output_bboxes[:, 3], gt_bbox[3])

    iw = np.maximum(ixmax - ixmin, 0.0)
    ih = np.maximum(iymax - iymin, 0.0)

    # 2.calculate the area of inters
    inters = iw * ih

    # 3.calculate the area of union
    unions = (
        (output_bboxes[:, 2] - output_bboxes[:, 0])
        * (output_bboxes[:, 3] - output_bboxes[:, 1])
        + (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
        - inters
    )

    # 4.calculate the overlaps
    return inters / unions


def check_bboxes_match(
    output_bboxes: np.ndarray, gt_bbox: np.ndarray
) -> Optional[int]:
    """Check if one of the bbox outputs matche the GT.

    Return None if none of them match Based on
    https://github.com/daodaofr/AlignPS/blob/5d62dbcd39a4f5ed996e15057c1784df9a1161bb/tools/test_results.py#L155C21-L166C38
    """
    width, height = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    iou_threshold = min(0.5, (width * height) / ((width + 10) * (height + 10)))
    ious = compute_ious(output_bboxes, gt_bbox)

    ok_ious_indexes = [i for i, iou in enumerate(ious) if iou > iou_threshold]

    if not ok_ious_indexes:
        return None

    max_iou = 0
    i_max_iou = -1
    for i in ok_ious_indexes:
        if ious[i] > max_iou:
            max_iou = ious[i]
            i_max_iou = i

    return i_max_iou
    # width, height = gt_bbox[2], gt_bbox[3]
    # iou_threshold = min(0.5, (width * height) / ((width + 10) * (height + 10)))

    # gt_bbox_xyxy = np.array(
    #     [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]],
    # )
    # ious = compute_ious(output_bboxes.astype(float), gt_bbox_xyxy.astype(float))

    # ok_ious = [iou for iou in ious if iou > iou_threshold]

    # if not ok_ious:
    #     return None

    # i_best_bbox = int(np.argmax(ok_ious))

    # return i_best_bbox
