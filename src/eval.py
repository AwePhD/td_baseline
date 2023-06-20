from typing import List, Tuple, Generator, Dict

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
from data_struct import Sample, CropIndex, Query, FrameOutput, GalleryElement

GALLERY_SIZE = 100
SCORE_THRESHOLD = .25

def _load_samples(
    annotations: pd.DataFrame,
    frame_id_to_frame_output: Dict[int, FrameOutput],
    crop_index_to_captions_output: Dict[CropIndex, torch.Tensor]
) -> Generator[Sample, None, None]:
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
            frame_id_to_frame_output[query_index.frame_id],
            crop_index_to_captions_output[query_index],
            torch.tensor(gt_bboxes.loc[query_index].values, dtype=torch.int32)
        )

        gallery_indexes =  [
            CropIndex(person_id, frame_id)
            for person_id, frame_id in annotation_sample.query("type == 'gallery'").index.tolist()
        ]
        gallery: List[GalleryElement] = [
            GalleryElement(
                frame_index.frame_id,
                frame_id_to_frame_output[frame_index.frame_id],
                torch.tensor(gt_bboxes.loc[frame_index].values, dtype=torch.int32)
                if frame_index in gt_bboxes.index
                else None

            )
            for frame_index in gallery_indexes
        ]

        yield Sample(query_index.person_id, query, gallery)

def _get_argmax_iou(
    output_bboxes: torch.Tensor,
    gt_bbox: torch.Tensor
) -> int:
    """
    Compute IoUs between bboxes from model and theirs GT.

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
    iou = inters / uni
    nmax = np.argmax(iou)
    return nmax

def _get_features_of_best_output_bbox(
    query_frame_output: FrameOutput,
    query_annotation: pd.DataFrame,
) -> torch.Tensor:
    gt_bbox = (
        query_annotation.bbox_x,
        query_annotation.bbox_y,
        query_annotation.bbox_x + query_annotation.bbox_w,
        query_annotation.bbox_y + query_annotation.bbox_h,
    )
    i_best_match = _get_argmax_iou(query_frame_output.bboxes, gt_bbox)

    return normalize(query_frame_output.features[i_best_match])


def _compute_labels_scores_for_one_gallery_frame(
    frame: FrameOutput,
    compute_method,
    frame_index,
    annotation_sample,
    query_text_features,
    query_image_features,
    threshold,
    ):
    """
     1 Keep detections above a certain thresholds
     pass to the next element if there is no detections remaining.
     2 Normalize Image feautres
     3 Build Similarity matrice
       NOTE: This is here that we can introduce different method
     4 NOT(Has query person)
       2.3.4.False -> All labels are False, append scores and pass to the
       next sample.
     5 Determine by IoU if a bbox can be considered as TP
        5.1 Compute IoU threshold based on height and width of GT bbox
        5.2 Compute IoU between GT and outputs,
        consider the best IoU score superior to the threshold as TP. Else FP.
        5.3 Append labels with a True to the TP, if it exists
     6 Append scores with similarity
    """
    kept_index = frame.frame_output.scores >= threshold

    n_result = kept_index.sum()
    if n_result == 0:
        # No correct detections in this frame
        return (None, None)
    else:
        labels = torch.zeros(n_result, dtype=bool)

    frame_features = normalize(
        frame.frame_output.features[kept_index])

    similarities = _build_similarity(
        compute_method,
        frame_features,
        query_image_features,
        query_text_features,
    )

    # No query person, fill labels and scores
    if annotation_sample.iloc[frame_index].bbox_w == 0:
        return labels, similarities

    i_bbox = _check_bboxes_match(
        frame.bboxes[kept_index], annotation_sample.iloc[frame_index])
    if not(i_bbox is None):
        labels[i_bbox] = True

    return labels, similarities

def _evaluate_one_sample(
    sample: Sample,
    compute_method,
    threshold: float = SCORE_THRESHOLD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
     0. Init labels and scores variables for the samples. They
     respectively denote the positions of the GTs among the detections results
     and the score of the results.
     1 Filter query bbox output (does not depend upon captions)
     1.1 Select bbox with best IoU compared to gt
     1.2 Normalize its image features
         |-> query_image_features
     2 Perform two searches, one for each caption
         2.1 Get Captions features
         2.2 Normalize them
         2.3 For each element in gallery, get scores and labels
    3. Turn list to tensor and return
    """
    labels_sample_list: List[torch.Tensor] = []
    scores_sample_list: List[torch.Tensor] = []


    query_index = sample.double_query.index
    query_image_features = _get_features_of_best_output_bbox(
        sample.double_query.frame_output,
        annotation_sample.query("type == 'query'"))

    for caption_output in zip(sample.double_query.captions_output):
        query_text_features = _get_text_features()

        for gallery_element in sample.gallery:
            frame = gallery_element.frame_output
            frame_index = CropIndex(query_index, gallery_element.frame_id)

            labels, scores = _compute_labels_scores_for_one_gallery_frame(
                    frame,
                    frame_index,
                    compute_method,
                    query_text_features,
                    query_image_features,
                    threshold,
            )
            if labels is None:
                continue

            labels_sample_list.append(labels)
            scores_sample_list.append(similarities)


def main():
    # Import annotations and model outputs
    annotations = _import_annotations()
    crop_index_to_captions_output = import_captions_output_from_hdf5(H5_CAPTIONS_OUTPUT_FILE)
    frame_id_to_frame_output = import_frame_output_from_hdf5(H5_FRAME_OUTPUT_FILE)

    samples: Generator[Sample] = _load_samples(
        annotations, frame_id_to_frame_output, crop_index_to_captions_output)

    labels_list: List[torch.Tensor] = []
    scores_list: List[torch.Tensor] = []

    for sample  in samples:
        labels_sample, scores_sample = _evaluate_one_sample(sample, compute_method)

        labels_list.append(labels_sample)
        scores_list.append(scores_sample)

    labels = torch.stack(labels_list)
    scores = torch.stack(scores_list)

    # Evaluate the mAP on the whole sets of search performance

if __name__ == "__main__":
    main()