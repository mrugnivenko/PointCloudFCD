"""This module contains functions for brain related data visualization."""
import typing as tp

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import torch

# Typing for images - could be either PyTorch tensor or Nifti Image.
InputType = tp.Union[torch.Tensor, nibabel.nifti1.Nifti1Image]


def plot_central_cuts(img: InputType, label: bool = False) -> None:
    """
    Plot central slices of MRI.

    Parameters:
        img: MR image;
        label: Name of object to plot.
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if len(img.shape) > 3:
            img = img[0, :, :, :]
    elif isinstance(img, nibabel.nifti1.Nifti1Image):
        img = img.get_fdata()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    if label:
        fig.suptitle(f"Central cuts of {label}", fontsize=16)
    else:
        fig.suptitle("Central cuts", fontsize=16)

    axes[0].imshow(img[img.shape[0] // 2, :, :], cmap="gray")
    axes[0].set_title(f"coordinate sagital = {img.shape[0] // 2}")
    axes[1].imshow(img[:, img.shape[1] // 2, :], cmap="gray")
    axes[1].set_title(f"coordinate coronal = {img.shape[1] // 2}")
    axes[2].imshow(img[:, :, img.shape[2] // 2], cmap="gray")
    axes[2].set_title(f"coordinate axial = {img.shape[2] // 2}")
    plt.show()


def plot_certain_cuts(img: InputType, coordinates: tuple, label: bool = False) -> None:
    """
    Plots certain slices of MRI.

    Parameters:
        img: MR image;
        coordinates: Coordinates of slices to plot;
        label: Name of object to plot.
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if len(img.shape) > 3:
            img = img[0, :, :, :]
    elif isinstance(img, nibabel.nifti1.Nifti1Image):
        img = img.get_fdata()

    coordinate_sagital, coordinate_coronal, coordinate_axial = coordinates

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    fig.suptitle(f"Certain cuts", fontsize=16)
    if label:
        fig.suptitle(f"Certain cuts of {label}", fontsize=16)

    axes[0].imshow(img[coordinate_sagital, :, :], cmap="gray")
    axes[0].set_title(f"coordinate sagital = {coordinate_sagital}")
    axes[1].imshow(img[:, coordinate_coronal, :], cmap="gray")
    axes[1].set_title(f"coordinate coronal = {coordinate_coronal}")
    axes[2].imshow(img[:, :, coordinate_axial], cmap="gray")
    axes[2].set_title(f"coordinate axial = {coordinate_axial}")

    plt.show()


def get_center_coord_of_mask(img: InputType) -> list:
    """
    Find central coordinates of the mask.

    Parameters:
        img: Mask of FCD.

    Returns:
        Corresponding central coordinates of mask.
    """
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if len(img.shape) > 3:
            img = img[0, :, :, :]
    elif isinstance(img, nibabel.nifti1.Nifti1Image):
        img = img.get_fdata()

    axial_coord = []
    for i in range(img.shape[2]):
        if (img[:, :, i] != 0).any():
            axial_coord.append(i)
    axial_center_coord = (min(axial_coord) + max(axial_coord)) // 2

    coronal_coord = []
    for i in range(img.shape[1]):
        if (img[:, i, :] != 0).any():
            coronal_coord.append(i)
    coronal_center_coord = (min(coronal_coord) + max(coronal_coord)) // 2

    sagital_coord = []
    for i in range(img.shape[0]):
        if (img[i, :, :] != 0).any():
            sagital_coord.append(i)
    sagital_center_coord = (min(sagital_coord) + max(sagital_coord)) // 2

    return [sagital_center_coord, coronal_center_coord, axial_center_coord]
