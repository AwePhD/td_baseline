from pathlib import Path
from typing import List, Generator, Dict, Optional, Tuple, Iterable

from tqdm import tqdm
import numpy as np
import pandas as pd

from ..metrics import compute_average_precision_with_recall_penality
from ..cuhk_sysu_pedes import import_test_annotations
from ..captions_features import import_captions_output_from_hdf5
from ..data_struct import Sample, CropIndex, Query, FrameOutput, GalleryFrame, DetectionOutput, Gallery
from ..crop_features import import_bboxes_clip_features_from_hdf5
from ..utils import gt_bboxes_from_annotations, extract_int_from_str
from ..pstr_output import import_detection_output_from_hdf5
from .compute_similarities import ComputeSimilarities


GALLERY_SIZE = 100


def _load_samples(
    annotations: pd.DataFrame,
    frame_file_to_detection_output: Dict[Path, DetectionOutput],
    frame_id_to_bboxes_clip_features: Dict[int, np.ndarray],
    crop_index_to_captions_output: Dict[CropIndex, np.ndarray]
) -> Generator[Sample, None, None]:
    frame_id_to_detection_output = {
        extract_int_from_str(frame_file.name): detection_output
        for frame_file, detection_output in frame_file_to_detection_output.items()
    }

    for _, annotation_sample in annotations.groupby("person_id"):
        sample_gt_bboxes = gt_bboxes_from_annotations(annotation_sample)

        query_index = CropIndex(
            *annotation_sample.query("type == 'query'").index[0])
        query = Query(
            query_index.frame_id,
            FrameOutput(
                frame_id_to_detection_output[query_index.frame_id].scores,
                frame_id_to_detection_output[query_index.frame_id].bboxes,
                frame_id_to_detection_output[query_index.frame_id].features_pstr,
                frame_id_to_bboxes_clip_features[query_index.frame_id],
            ),
            crop_index_to_captions_output[query_index],
            sample_gt_bboxes[query_index]
        )

        gallery_annotations = annotation_sample.query("type == 'gallery'")
        gallery = [
            GalleryFrame(
                frame_id,
                FrameOutput(
                    frame_id_to_detection_output[frame_id].scores,
                    frame_id_to_detection_output[frame_id].bboxes,
                    frame_id_to_detection_output[frame_id].features_pstr,
                    frame_id_to_bboxes_clip_features[frame_id],
                ),
                sample_gt_bboxes[(person_id, frame_id)]
                if (person_id, frame_id) in sample_gt_bboxes.keys()
                else None
            )
            for person_id, frame_id in gallery_annotations.index
        ]

        yield Sample(query_index.person_id, query, gallery)


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
     Filter query bbox output (does not depend upon captions)
     1 Select bbox with best IoU compared to gt
     2 Normalize its image features
    """
    i_best_match = np.argmax(_compute_ious(
        query.frame_output.bboxes, query.gt_bbox))

    return (
        query.frame_output.features_pstr[i_best_match],
        query.frame_output.features_clip[i_best_match],
    )


def _check_bboxes_match(
    output_bboxes: np.ndarray,
    gt_bbox: np.ndarray
) -> Optional[int]:
    """
    Check if one of the bbox outputs matche the GT.
    Return None if none of them match
    """
    width, height = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    iou_threshold = min(0.5, (width * height) / ((width + 10) * (height + 10)))
    ious = _compute_ious(output_bboxes, gt_bbox)

    for i, iou in enumerate(ious):
        if iou > iou_threshold:
            return i

    return None


def _compute_labels_scores_for_one_gallery_frame(
    frame: GalleryFrame,
    query_text_features: np.ndarray,
    query_crop_features_pstr_clip: Tuple[np.ndarray, np.ndarray],
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
    # NOTE: Tricky part
    # We prioritize the matching by starting with the best IoU
    # Also we have to reorder the similarites because i_bbox
    # is an index based on the similarities decreasing order!
    # If we don't, the similarities does not match their labels anymore
    indices_by_similarities = similarities.argsort()[::-1]
    ranked_similarities = similarities[indices_by_similarities]
    ranked_bboxes = frame.frame_output.bboxes[kept_index][indices_by_similarities]
    i_bbox = _check_bboxes_match(
        ranked_bboxes,
        frame.gt_bbox)
    if i_bbox is not None:
        labels[i_bbox] = True

    return labels, ranked_similarities


def _evaluate_one_query_for_one_sample(
    gallery: Gallery,
    query_text_features: np.ndarray,
    query_crop_features_pstr_clip: Tuple[np.ndarray, np.ndarray],
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

    return compute_average_precision_with_recall_penality(labels, scores, count_gt)


def _evaluate_one_sample(
    sample: Sample,
    compute_similarities: ComputeSimilarities,
    threshold: float,
) -> Tuple[float, float]:
    """
     1. Get query image features
     2. Evaluate two searches - 2 mAP values. One search by caption
     3. Return both mAPs
    """
    query_crop_features_pstr_clip = _get_features_of_best_output_bbox(
        sample.query)

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
    data_folder: Path,
    frames_folder: Path,
    h5_captions_output_file: Path = None,
    h5_detection_output: Path = None,
    h5_bboxes_clip_features: Path = None,
) -> Generator[Sample, None, None]:
    annotations = import_test_annotations(data_folder)
    crop_index_to_captions_output = (
        import_captions_output_from_hdf5(h5_captions_output_file))
    frame_file_to_detection_output = (
        import_detection_output_from_hdf5(h5_detection_output, frames_folder))
    frame_id_to_bboxes_clip_features = (
        import_bboxes_clip_features_from_hdf5(h5_bboxes_clip_features))

    return _load_samples(
        annotations,
        frame_file_to_detection_output,
        frame_id_to_bboxes_clip_features,
        crop_index_to_captions_output
    )


def compute_mean_average_precision(
    samples: Iterable[Sample],
    compute_similarities: ComputeSimilarities,
    threshold: float,
) -> float:
    average_precisions: List[float] = []

    for sample in tqdm(samples):
        average_precisions.extend(
            _evaluate_one_sample(sample, compute_similarities, threshold))

    return np.mean(average_precisions)
