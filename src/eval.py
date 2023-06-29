from pathlib import Path
from typing import List, Generator, Dict, Optional, Tuple, Iterable, NewType

from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

from features_generation import (
    _import_annotations,
     import_captions_output_from_hdf5,
     import_frame_output_from_hdf5,
     H5_CAPTIONS_OUTPUT_FILE,
     H5_FRAME_OUTPUT_FILE,
)
from data_struct import Sample, CropIndex, Query, FrameOutput, GalleryFrame, CaptionsOutput, Gallery
from compute_similarities import average, ComputeSimilarities

GALLERY_SIZE = 100
SCORE_THRESHOLD = .25
#
T_FLOAT = NewType('T_FLOAT', torch.float)

def _get_frame_output_from_h5(h5_file: h5py.File, frame_id: int) -> FrameOutput:
    frame_id_key = f"s{frame_id}.jpg"
    return FrameOutput(
        torch.tensor(h5_file[frame_id_key][FrameOutput._fields[0]]),
        torch.tensor(h5_file[frame_id_key][FrameOutput._fields[1]]),
        torch.tensor(h5_file[frame_id_key][FrameOutput._fields[2]]),
    )

def _load_samples(
    annotations: pd.DataFrame,
    frame_id_to_frame_output: Dict[int, FrameOutput],
    crop_index_to_captions_output: Dict[CropIndex, torch.Tensor]
) -> Generator[Sample, None, None]:
    for _, annotation_sample in annotations.groupby("person_id"):
        all_gt_bboxes = (
            annotation_sample[["bbox_x", "bbox_y","bbox_w", "bbox_h" ]]
            [annotation_sample.bbox_w != 0 ]
            # int32 because tensor cannot convert from uint16 (dtype of bbox coords).
            .astype('Int32')
            .copy()
        )
        all_gt_bboxes['bbox_x_end'] = all_gt_bboxes.bbox_x + all_gt_bboxes.pop("bbox_w")
        all_gt_bboxes['bbox_y_end'] = all_gt_bboxes.bbox_y + all_gt_bboxes.pop("bbox_h")

        query_index = CropIndex(*annotation_sample.query("type == 'query'").index.tolist()[0])
        query = Query(
            query_index.frame_id,
            frame_id_to_frame_output[query_index.frame_id],
            crop_index_to_captions_output[query_index],
            torch.tensor(all_gt_bboxes.loc[query_index].values, dtype=torch.int32)
        )

        gallery_indexes =  [
            CropIndex(person_id, frame_id)
            for person_id, frame_id in annotation_sample.query("type == 'gallery'").index.tolist()
        ]
        gallery = [
            GalleryFrame(
                frame_index.frame_id,
                frame_id_to_frame_output[frame_index.frame_id],
                torch.tensor(all_gt_bboxes.loc[frame_index].values, dtype=torch.int32)
                if frame_index in all_gt_bboxes.index
                else None

            )
            for frame_index in gallery_indexes
        ]

        yield Sample(query_index.person_id, query, gallery)


def _load_samples_io(
    annotations: pd.DataFrame,
    frame_output_h5_file: Path,
    crop_index_to_captions_output: Dict[CropIndex, CaptionsOutput]
) -> Generator[Sample, None, None]:
    frame_output_h5 = h5py.File(frame_output_h5_file, 'r')
    for _, annotation_sample in annotations.groupby("person_id"):
        gt_bboxes = (
            annotation_sample[["bbox_x", "bbox_y","bbox_w", "bbox_h" ]]
            [annotation_sample.bbox_x != 0 ]
            # int32 because tensor cannot convert from uint16 (dtype of bbox coords).
            .astype('Int32')
            .copy()
        )
        gt_bboxes['bbox_x_end'] = gt_bboxes.bbox_x + gt_bboxes.pop("bbox_w")
        gt_bboxes['bbox_y_end'] = gt_bboxes.bbox_y + gt_bboxes.pop("bbox_h")

        query_index = CropIndex(*annotation_sample.query("type == 'query'").index.tolist()[0])
        query = Query(
            query_index.frame_id,
            _get_frame_output_from_h5(frame_output_h5, query_index.frame_id),
            crop_index_to_captions_output[query_index],
            torch.tensor(gt_bboxes.loc[query_index].values, dtype=torch.int32),

        )

        gallery_indexes =  [
            CropIndex(person_id, frame_id)
            for person_id, frame_id in annotation_sample.query("type == 'gallery'").index.tolist()
        ]
        gallery: List[GalleryFrame] = [
            GalleryFrame(
                frame_index.frame_id,
                _get_frame_output_from_h5(frame_output_h5, frame_index.frame_id),
                torch.tensor(gt_bboxes.loc[frame_index].values, dtype=torch.int32)
                if frame_index in gt_bboxes.index
                else None

            )
            for frame_index in gallery_indexes
        ]

        yield Sample(query_index.person_id, query, gallery)
    frame_output_h5.close()

def _compute_ious(
    output_bboxes: torch.Tensor,
    gt_bbox: torch.Tensor,
) -> torch.Tensor:
    """
    Compute IoUs between bboxes from model and its GT.

    output_bboxes (torch.Tensor): (N_BBOXES, 4) tensor of bboxes from model
    gt_bbox (torch.Tensor): (4,) Ground Truth bbox from annotation
    return: the index of the best matching (highest IoU) model bbox
    """
    # 1. calculate the inters coordinate
    ixmin = np.maximum(output_bboxes[:, 0], gt_bbox[0])
    ixmax = np.minimum(output_bboxes[:, 2], gt_bbox[2])
    iymin = np.maximum(output_bboxes[:, 1], gt_bbox[1])
    iymax = np.minimum(output_bboxes[:, 3], gt_bbox[3])

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
    inters = iw * ih

    # 3.calculate the area of union
    uni = (
        (output_bboxes[:, 2] - output_bboxes[:, 0] + 1.)
        * (output_bboxes[:, 3] - output_bboxes[:, 1] + 1.)
        + (gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.)
        - inters
    )

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
    return inters / uni


def _get_features_of_best_output_bbox(
    query: Query
) -> torch.Tensor:
    """
     Filter query bbox output (does not depend upon captions)
     1 Select bbox with best IoU compared to gt
     2 Normalize its image features
    """
    i_best_match = _compute_ious(query.frame_output.bboxes, query.gt_bbox).argmax()

    return normalize(
        query.frame_output.features[i_best_match].unsqueeze(0))[0]

def _check_bboxes_match(
    output_bboxes: torch.Tensor,
    gt_bbox: torch.Tensor
) -> Optional[int]:
    width, height = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    iou_threshold =  min(0.5, (width * height) / ((width + 10) * (height + 10)))
    ious = _compute_ious(output_bboxes, gt_bbox)

    for i, iou in enumerate(ious):
        if iou > iou_threshold:
            return i

    return None



def _compute_labels_scores_for_one_gallery_frame(
    frame: GalleryFrame,
    query_text_features: torch.Tensor,
    query_image_features: torch.Tensor,
    compute_similarities: ComputeSimilarities,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1 Keep detections above a certain thresholds
    pass to the next element
        => If there is no detections remaining, return None.
    2 Normalize image feautres
    3 Build Similarity matrix
      NOTE: This is here that we can introduce different method
    4 Does not have the query person in frame
        => Return labels with only False and similarities
    5 Determine by IoU if a bbox can be considered as TP
       5.1 Compute IoU threshold based on height and width of GT bbox
       5.2 Compute IoU between GT and outputs,
       consider the best IoU score superior to the threshold as TP.
       If no IoU scores is above the threshold => all FPs.
       5.3 If there is a TP set its label to True
        => Return labels and similarities
    """
    kept_index = frame.frame_output.scores >= threshold

    n_result = kept_index.sum()
    if n_result == 0:
        return None

    # [n_result, 512]
    frame_features = normalize(
        frame.frame_output.features[kept_index])

    # [n_result]
    similarities = compute_similarities(
        query_image_features,
        query_text_features,
        frame_features,
    )

    labels = torch.zeros(n_result, dtype=bool)

    # No query person inside the gallery frame
    if frame.gt_bbox is None:
        return labels, similarities

    # [n_result]
    indices_by_similarities = similarities.argsort(descending=True)
    i_bbox = _check_bboxes_match(
        frame.frame_output.bboxes[kept_index][indices_by_similarities], frame.gt_bbox)
    if i_bbox is not None:
        labels[i_bbox] = True

    return labels, similarities

def _compute_average_precision(
    labels: torch.Tensor,
    scores: torch.Tensor,
    count_gt: int,
) -> T_FLOAT:
    """
    Namely, we compute the average precision over recall values i.e cut-off for each TPs.
    There is a penality if the co
    """
    indices_by_scores = scores.cuda().argsort(descending=True).cpu()
    labels_ranked = labels[indices_by_scores]

    count_tp = labels_ranked.sum()
    if count_tp == 0:
        return count_tp

    recall_rate_penality = count_tp / count_gt

    tps = labels_ranked.cumsum(0)

    precisions = (tps / torch.arange(1, len(tps) + 1))
    precisions_at_delta_recall = precisions[labels_ranked]
    mean_average_precision = precisions_at_delta_recall.sum() / count_tp

    return  mean_average_precision * recall_rate_penality

def _evaluate_one_query_for_one_sample(
    gallery: Gallery,
    query_text_features: torch.Tensor,
    query_image_features: torch.Tensor,
    compute_similarities: ComputeSimilarities,
    threshold: float
) -> T_FLOAT:
    """
    1. Init labels and scores variables for the sample. They
    respectively denote the positions of the GTs among the detections results
    and the score of the results.
    2. For each frame in gallery, get scores and labels of the search
    3. Gather results from every searches in gallery.
    """
    labels_temp: List[torch.Tensor] = []
    scores_temp: List[torch.Tensor] = []

    for gallery_frame in gallery:
        # (labels, scores) for the gallery frame OR None
        result = _compute_labels_scores_for_one_gallery_frame(
            gallery_frame,
            query_text_features,
            query_image_features,
            compute_similarities,
            threshold
        )
        if result is None:
            continue

        labels_temp.append(result[0])
        scores_temp.append(result[1])

    labels = torch.cat(labels_temp)
    scores = torch.cat(scores_temp)

    count_gt = sum(frame.gt_bbox is not None for frame in gallery)

    return _compute_average_precision(labels, scores, count_gt)

def _evaluate_one_sample(
    sample: Sample,
    compute_similarities: ComputeSimilarities,
    threshold: float = SCORE_THRESHOLD,
) -> Tuple[T_FLOAT, T_FLOAT]:
    """
     1. Get query image features
     2. Evaluate two searches - 2 mAP values. One search by caption
     3. Return both mAPs
    """
    # [512]
    query_image_features = _get_features_of_best_output_bbox(sample.query)

    return tuple(
        _evaluate_one_query_for_one_sample(
            sample.gallery,
            normalize(caption_features.unsqueeze(0))[0],
            query_image_features,
            compute_similarities,
            threshold
        )
        for caption_features in sample.query.captions_output
    )

def import_data(
    h5_captions_output_file: Path,
    h5_frame_output_file: Path,
) -> Generator[Sample, None, None]:
    annotations = _import_annotations()
    crop_index_to_captions_output = import_captions_output_from_hdf5(h5_captions_output_file)
    frame_id_to_frame_output = import_frame_output_from_hdf5(h5_frame_output_file)

    return _load_samples(
        annotations, frame_id_to_frame_output, crop_index_to_captions_output)

def compute_mean_average_precision(
    samples: Iterable[Sample],
    compute_similarities: ComputeSimilarities,
    threshold: float,
) -> float:
    average_precisions: List[T_FLOAT] = []

    for sample  in tqdm(samples):
        average_precisions.extend(
            _evaluate_one_sample(sample, compute_similarities, threshold))

    return np.mean(average_precisions)

def main():
    samples = import_data(H5_CAPTIONS_OUTPUT_FILE, H5_FRAME_OUTPUT_FILE)
    mean_average_precision = compute_mean_average_precision(samples, average, SCORE_THRESHOLD)
    print(f"mAP: {mean_average_precision:.2%}")


if __name__ == "__main__":
    main()