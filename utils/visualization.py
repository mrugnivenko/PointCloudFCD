"""
Description of the following fucntions:
    * plot_central_cuts
    * plot_certain_cuts
    * get_center_coord_of_bb
"""

import torch
import nibabel
import numpy as np
import matplotlib.pyplot as plt

def plot_central_cuts(img, label=False):
    """
    Function plots central slices of MRI
    
    :params img: torch.tensor or nibabel.nifti1.Nifti1Image, MR image
    :params label: str, name of object for which we plot slices
    """

    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if (len(img.shape) > 3):
            img = img[0, :, :, :]
    elif isinstance(img, nibabel.nifti1.Nifti1Image):
        img = img.get_fdata()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    fig.suptitle('Central cuts', fontsize=16)
    if label:
        fig.suptitle(f'Central cuts of {label}', fontsize=16)

    axes[0].imshow(img[img.shape[0] // 2, :, :], cmap='gray')
    axes[0].set_title(f'coordinate sagital = {img.shape[0] // 2}')
    axes[1].imshow(img[:, img.shape[1] // 2, :], cmap='gray')
    axes[1].set_title(f'coordinate coronal = {img.shape[1] // 2}')
    axes[2].imshow(img[:, :, img.shape[2] // 2], cmap='gray')
    axes[2].set_title(f'coordinate axial = {img.shape[2] // 2}')

    plt.show()

def plot_certain_cuts(img, coordinates, label=False):
    """
    Function plots certain slices of MRI

    :params img: torch.tensor or nibabel.nifti1.Nifti1Image, MR image
    :params coordinates: tuple, (x, y, z) - coordinates of slices which we want to see
    :params label: str, name of object for which we plot slices
    """

    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if (len(img.shape) > 3):
            img = img[0, :, :, :]
    elif isinstance(img, nibabel.nifti1.Nifti1Image):
        img = img.get_fdata()

    coordinate_sagital, coordinate_coronal, coordinate_axial = coordinates
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    fig.suptitle(f'Certain cuts', fontsize=16)
    if label:
        fig.suptitle(f'Certain cuts of {label}', fontsize=16)

    axes[0].imshow(img[coordinate_sagital, :, :], cmap='gray')
    axes[0].set_title(f'coordinate sagital = {coordinate_sagital}')
    axes[1].imshow(img[:, coordinate_coronal, :], cmap='gray')
    axes[1].set_title(f'coordinate coronal = {coordinate_coronal}')
    axes[2].imshow(img[:, :, coordinate_axial], cmap='gray')
    axes[2].set_title(f'coordinate axial = {coordinate_axial}')

    plt.show()

def get_center_coord_of_mask(img):
    """
    Function finds central coordinates of mask

    :params img: torch.tensor or nibabel.nifti1.Nifti1Image, mask of FCD

    :outputs coordinates: tuple, (sagital_center_coord, coronal_center_coord, axial_center_coord) - corresponding central coordinates of mask
    """

    if isinstance(img, torch.Tensor):
        img = img.numpy()
        if (len(img.shape) > 3):
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

    return sagital_center_coord, coronal_center_coord, axial_center_coord