from pathlib import Path
from typing import List, Generator, Dict, Optional, Tuple, Iterable

from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd

from features_generation import (
    _import_annotations,
     import_captions_output_from_hdf5,
     import_frame_output_from_hdf5,
     H5_CAPTIONS_OUTPUT_FILE,
     H5_FRAME_OUTPUT_FILE,
)
from data_struct import Sample, CropIndex, Query, FrameOutput, GalleryFrame, CaptionsOutput, Gallery
from compute_similarities import pstr_similarities, ComputeSimilarities

GALLERY_SIZE = 100
SCORE_THRESHOLD = .50
#
def _get_frame_output_from_h5(h5_file: h5py.File, frame_id: int) -> FrameOutput:
    frame_id_key = f"s{frame_id}.jpg"
    return FrameOutput(
        h5_file[frame_id_key][FrameOutput._fields[0]][...],
        h5_file[frame_id_key][FrameOutput._fields[1]][...],
        h5_file[frame_id_key][FrameOutput._fields[2]][...],
        h5_file[frame_id_key][FrameOutput._fields[3]][...],
    )

def _load_samples(
    annotations: pd.DataFrame,
    frame_id_to_frame_output: Dict[int, FrameOutput],
    crop_index_to_captions_output: Dict[CropIndex, np.ndarray]
) -> Generator[Sample, None, None]:
    for _, annotation_sample in annotations.groupby("person_id"):
        all_gt_bboxes = (
            annotation_sample[["bbox_x", "bbox_y","bbox_w", "bbox_h" ]]
            [annotation_sample.bbox_w != 0 ]
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
            all_gt_bboxes.loc[query_index].values.astype('int32')
        )

        gallery_indexes =  [
            CropIndex(person_id, frame_id)
            for person_id, frame_id in annotation_sample.query("type == 'gallery'").index.tolist()
        ]
        gallery = [
            GalleryFrame(
                frame_index.frame_id,
                frame_id_to_frame_output[frame_index.frame_id],
                all_gt_bboxes.loc[frame_index].values.astype(np.int32)
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
            gt_bboxes.loc[query_index].values.astype(np.int32),

        )

        gallery_indexes =  [
            CropIndex(person_id, frame_id)
            for person_id, frame_id in annotation_sample.query("type == 'gallery'").index.tolist()
        ]
        gallery: List[GalleryFrame] = [
            GalleryFrame(
                frame_index.frame_id,
                _get_frame_output_from_h5(frame_output_h5, frame_index.frame_id),
                gt_bboxes.loc[frame_index].values.astype(np.int32)
                if frame_index in gt_bboxes.index
                else None

            )
            for frame_index in gallery_indexes
        ]

        yield Sample(query_index.person_id, query, gallery)
    frame_output_h5.close()

def _compute_ious(
    output_bboxes: np.ndarray,
    gt_bbox: np.ndarray,
) -> np.ndarray:
    """
    Compute IoUs between bboxes from model and its GT.

    output_bboxes (np.ndarray): (N_BBOXES, 4) tensor of bboxes from model
    gt_bbox (np.ndarray): (4,) Ground Truth bbox from annotation
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
    unions = (
        (output_bboxes[:, 2] - output_bboxes[:, 0] + 1.)
        * (output_bboxes[:, 3] - output_bboxes[:, 1] + 1.)
        + (gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.)
        - inters
    )

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
    return inters / unions


def _get_features_of_best_output_bbox(
    query: Query
) -> np.ndarray:
    """
     Filter query bbox output (does not depend upon captions)
     1 Select bbox with best IoU compared to gt
     2 Normalize its image features
    """
    i_best_match = _compute_ious(query.frame_output.bboxes, query.gt_bbox).argmax()

    return (
        query.frame_output.features_pstr[i_best_match],
        query.frame_output.features_clip[i_best_match],
    )

def _check_bboxes_match(
    output_bboxes: np.ndarray,
    gt_bbox: np.ndarray
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
    query_text_features: np.ndarray,
    query_crop_features_pstr_clip: np.ndarray,
    compute_similarities: ComputeSimilarities,
    threshold: float,
) -> Tuple[float, float]:
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
    crops_features_pstr_clip = (
        frame.frame_output.features_pstr[kept_index],
        frame.frame_output.features_clip[kept_index],
    )

    # [n_result]
    similarities = compute_similarities(
        query_crop_features_pstr_clip,
        query_text_features,
        crops_features_pstr_clip,
    )

    labels = np.zeros(n_result, dtype=bool)

    # No query person inside the gallery frame
    if frame.gt_bbox is None:
        return labels, similarities

    # [n_result]
    indices_by_similarities = similarities.argsort()[::-1]
    i_bbox = _check_bboxes_match(
        frame.frame_output.bboxes[kept_index][indices_by_similarities],
        frame.gt_bbox)
    if i_bbox is not None:
        labels[i_bbox] = True

    return labels, similarities

def _compute_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    count_gt: int,
) -> float:
    """
    Namely, we compute the average precision over recall values i.e cut-off for each TPs.
    There is a penality if the co
    """
    # No TP -> AP = 0
    count_tp = labels.sum()
    if count_tp == 0:
        return count_tp

    indices_by_scores = scores.argsort()[::-1]
    labels_ranked = labels[indices_by_scores]

    tps = labels_ranked.cumsum(0)

    precisions = (tps / np.arange(1, len(tps) + 1))
    precisions_at_delta_recall = precisions[labels_ranked]
    mean_average_precision = precisions_at_delta_recall.sum() / count_tp

    recall_rate_penality = count_tp / count_gt
    return  mean_average_precision * recall_rate_penality

def _evaluate_one_query_for_one_sample(
    gallery: Gallery,
    query_text_features: np.ndarray,
    query_crop_features_pstr_clip: np.ndarray,
    compute_similarities: ComputeSimilarities,
    threshold: float
) -> float:
    """
    1. Init labels and scores variables for the sample. They
    respectively denote the positions of the GTs among the detections results
    and the score of the results.
    2. For each frame in gallery, get scores and labels of the search
    3. Gather results from every searches in gallery.
    """
    labels_temp: List[np.ndarray] = []
    scores_temp: List[np.ndarray] = []

    for gallery_frame in gallery:
        # (labels, scores) for the gallery frame OR None
        result = _compute_labels_scores_for_one_gallery_frame(
            gallery_frame,
            query_text_features,
            query_crop_features_pstr_clip,
            compute_similarities,
            threshold
        )
        if result is None:
            continue

        # Add one dim to be able to concatenate them
        labels_temp.append(result[0].reshape(1, -1))
        scores_temp.append(result[1].reshape(1, -1))

    labels = np.concatenate(labels_temp, axis=1).ravel()
    scores = np.concatenate(scores_temp, axis=1).ravel()

    count_gt = sum(frame.gt_bbox is not None for frame in gallery)

    return _compute_average_precision(labels, scores, count_gt)

def _evaluate_one_sample(
    sample: Sample,
    compute_similarities: ComputeSimilarities,
    threshold: float = SCORE_THRESHOLD,
) -> Tuple[float, float]:
    """
     1. Get query image features
     2. Evaluate two searches - 2 mAP values. One search by caption
     3. Return both mAPs
    """
    # [512]
    query_crop_features_pstr_clip = _get_features_of_best_output_bbox(sample.query)

    return tuple(
        _evaluate_one_query_for_one_sample(
            sample.gallery,
            caption_features,
            query_crop_features_pstr_clip,
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
    average_precisions: List[float] = []

    for sample  in tqdm(samples):
        average_precisions.extend(
            _evaluate_one_sample(sample, compute_similarities, threshold))

    return np.mean(average_precisions)

def main():
    samples = import_data(H5_CAPTIONS_OUTPUT_FILE, H5_FRAME_OUTPUT_FILE)
    mean_average_precision = compute_mean_average_precision(samples, pstr_similarities, SCORE_THRESHOLD)
    print(f"mAP: {mean_average_precision:.2%}")


if __name__ == "__main__":
    main()