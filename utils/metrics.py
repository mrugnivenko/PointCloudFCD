import math

import numpy as np
import pandas as pd

from utils.crop_new_2 import *
from utils.data_preprocessing import load_nii_to_array


def calculate_contrast(points_labels_flats, pred_soft_flats):
    """
    Function calculates contrast metric

    points_labels_flats: list, list with points labels - 1 for FCD point and 0 for no-FCD point
    pred_soft_flats: list, list with probabilities of each point to be FCD point

    contrast: float, contrast metric
    """

    probability_inside_fcd = np.dot(points_labels_flats, pred_soft_flats) / np.sum(
        points_labels_flats
    )
    probability_outside_fcd = np.dot(
        np.ones(len(points_labels_flats)) - np.array(points_labels_flats),
        pred_soft_flats,
    ) / (len(points_labels_flats) - np.sum(points_labels_flats))

    return (probability_inside_fcd - probability_outside_fcd) / (
        probability_inside_fcd + probability_outside_fcd
    )


def top10_f(pred, label, coords, crop_size=64):
    """
    Function calculates top 10 score on point cloud and minimal number of crops to detect FCD.

    :params pred: np.array, array with probabilities of each point to be FCD point
    :params label: np.array, array with points labels - 1 for FCD point and 0 for no-FCD point
    :params coords: np.array, array with points coordinates
    :params crop_size: int, size of crop to calculate top 10 crop metric

    :outputs top10: float, top 10 score
    :outputs min_number_of_crops: int, minimal number of crops to detect FCD
    """

    df = pd.DataFrame(
        {
            "pred": pred,
            "label": label,
            "coord1": coords[:, 0],
            "coord2": coords[:, 1],
            "coord3": coords[:, 2],
        }
    )

    for i in range(1, 4):
        df[f"coord{i}"] = df[f"coord{i}"] // crop_size

    df = df.groupby([f"coord{i}" for i in range(1, 4)]).mean().reset_index()
    df.label = 1 - df.label

    df = df.sort_values(["pred", "label"], ascending=False).reset_index(drop=True)
    if df[df.label != 1].shape[0] == 0:
        min_number_of_crops = None
    else:
        min_number_of_crops = df[df.label != 1].head(1).index.item() + 1

    df = df.head(10)
    df = df[df.label != 1]

    if df.shape[0] == 0:
        top10 = 0.0
    else:
        top10 = 1 - float(df.head(1).index.item()) / 10

    return top10, min_number_of_crops


def compute_metrics_on_voxels(y_pred, y_true):
    """
    Function computes DICE, IoU, contrast, and recall between y_true and y_pred

    :params y_pred: np.array, array-like object with predictions
    :params y_true: np.array, array-like object with ground-truth labels 0 and 1

    :outputs dice, iou, contrast and recall metrics
    """

    y_pred = np.array(y_pred).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    assert (
        y_pred.shape == y_true.shape
    ), "Predictions and ground-truth have different sizes"

    y_true_sum = y_true.sum()
    predicted_fcd_thr = y_pred > 0.5

    tp = np.logical_and(predicted_fcd_thr, y_true).sum()
    fp = np.logical_and(1 - y_true, predicted_fcd_thr).sum()
    fn = np.logical_and(y_true, 1 - predicted_fcd_thr).sum()

    dice = 2 * tp / (predicted_fcd_thr.sum() + y_true_sum)
    iou = tp / np.logical_or(predicted_fcd_thr, y_true).sum()

    d_in = y_pred @ y_true / y_true_sum
    d_out = y_pred @ (1 - y_true) / (y_true.shape[0] - y_true_sum)
    contrast = (d_in - d_out) / (d_in + d_out)

    recall = tp / (tp + fn)

    return dice, iou, contrast, recall


def get_top_10_metric_and_mask(config, subject, prediction, label, crop_size=64):
    """
    Function calculates top 10 score on voxel based data and minimal number of crops to detect FCD. Also it makes top 10 masks.

    :params prediction: np.array, array with interpolated predictions - probabilities of each point to be FCD point
    :params label: np.array, array with labels - 1 for FCD point and 0 for no-FCD point
    :params crop_size: int, size of crop to calculate top 10 crop metric

    :outputs top10: float, top 10 score
    :outputs min_number_of_crops: int, minimal number of crops to detect FCD
    """

    df = pd.DataFrame(
        columns=[
            "mean_prediction",
            "mean_label",
            "sagittal_iter",
            "coronal_iter",
            "axial_iter",
        ]
    )

    sagittal_shape, coronal_shape, axial_shape = prediction.shape

    i = 0
    for sagittal_iter in range(math.ceil(sagittal_shape / crop_size)):
        for coronal_iter in range(math.ceil(coronal_shape / crop_size)):
            for axial_iter in range(math.ceil(axial_shape / crop_size)):

                prediction_crop = prediction[
                    sagittal_iter
                    * crop_size : min((sagittal_iter + 1) * crop_size, sagittal_shape),
                    coronal_iter
                    * crop_size : min((coronal_iter + 1) * crop_size, coronal_shape),
                    axial_iter
                    * crop_size : min((axial_iter + 1) * crop_size, axial_shape),
                ]

                label_crop = label[
                    sagittal_iter * crop_size : (sagittal_iter + 1) * crop_size,
                    coronal_iter * crop_size : (coronal_iter + 1) * crop_size,
                    axial_iter * crop_size : (axial_iter + 1) * crop_size,
                ]

                df.loc[i] = [
                    prediction_crop.mean(),
                    label_crop.mean(),
                    sagittal_iter,
                    coronal_iter,
                    axial_iter,
                ]

                i += 1

    df.mean_label = 1 - df.mean_label
    df = df.sort_values(["mean_prediction", "mean_label"], ascending=False).reset_index(
        drop=True
    )

    if df[df.mean_label != 1].shape[0] == 0:
        min_number_of_crops = None
    else:
        min_number_of_crops = df[df.mean_label != 1].head(1).index.item() + 1

    df = df.head(10)
    top_10_df = df.copy()
    df = df[df.mean_label != 1]

    if df.shape[0] == 0:
        top10 = 0.0
    else:
        top10 = 1 - float(df.head(1).index.item()) / 10

    top_10_mask = np.zeros(prediction.shape)

    for i in top_10_df.index:
        sagittal_iter, coronal_iter, axial_iter = top_10_df.loc[
            i, ["sagittal_iter", "coronal_iter", "axial_iter"]
        ].astype(int)

        top_10_mask[
            sagittal_iter * crop_size : (sagittal_iter + 1) * crop_size,
            coronal_iter * crop_size : (coronal_iter + 1) * crop_size,
            axial_iter * crop_size : (axial_iter + 1) * crop_size,
        ] = (
            1 - i / 10
        )

    original_brain = nib.load(
        f"{config.path_to_data}/{config.BRAIN_MODALITY}_brains/{subject}.nii.gz"
    )
    mask = nib.Nifti1Image(top_10_mask, original_brain.affine)
    nib.save(mask, f"experiments/{config.EXP_NAME}/top_10_masks/{subject}.nii.gz")

    return top10, min_number_of_crops


def get_overview_df_for_metric(
    prediction_path: str,
    label_path: str,
    crop_size: int,
    number_of_top_crops: int,
    folder_to_save: str,
    name_to_save: str,
):
    """
    Creates dataframe with information about top 10 crops of a given brain for metric calculation.

    Parameters
        prediction_path: path to prediction
        label_path: str: path to label
        crop_size: crop size
        number_of_top_crops: number of top crops to keep
        folder_to_save: name of folder to write resulting df
        name_to_save:  name of .csv file with results

    Returns
        Dataframe with information about top 10 crops for metric calculation.
    """
    crops_df = pd.DataFrame(
        columns=["x", "y", "z", "average_prediction", "label_size", "intersection_size"]
    )

    prediction = load_nii_to_array(prediction_path)
    label = (load_nii_to_array(label_path) > 0).astype("uint8")

    for x in range(0, prediction.shape[0] - crop_size // 2, crop_size // 2):
        for y in range(0, prediction.shape[1] - crop_size // 2, crop_size // 2):
            for z in range(0, prediction.shape[2] - crop_size // 2, crop_size // 2):

                crop_pred = prediction[
                    x : min(x + crop_size, prediction.shape[0]),
                    y : min(y + crop_size, prediction.shape[1]),
                    z : min(z + crop_size, prediction.shape[2]),
                ]
                crop_pred = (crop_pred > 0.5).astype("uint8")
                crop_label = label[
                    x : min(x + crop_size, prediction.shape[0]),
                    y : min(y + crop_size, prediction.shape[1]),
                    z : min(z + crop_size, prediction.shape[2]),
                ]

                crop_df = pd.DataFrame.from_dict(
                    {
                        "x": [x],
                        "y": [y],
                        "z": [z],
                        "average_prediction": [np.mean(crop_pred)],
                        "label_size": [label.sum()],
                        "intersection_size": [crop_label.sum()],
                    }
                )
                crops_df = crops_df.append(crop_df)

    crops_df = crops_df.sort_values(by="average_prediction", ascending=False)[
        :number_of_top_crops
    ]
    crops_df.to_csv(f"{folder_to_save}/{name_to_save}.csv")
