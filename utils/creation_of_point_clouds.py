import torch
import numpy as np
from .data_processor import *


def brain_and_mask_to_point_cloud_and_labels(brain, mask, size):
    """
    Transforms 3d tensor of brain into point-cloud with labels.
      Args:
          brain: torch tensor of size 'size' with intensity value at the positions with brain and 0 otherwise
          mask: torch tensor of size 'size' with the value 1 at the positions with FCD and 0 otherwise
          size: size of the input brain tensors
      Output:
          point cloud: torch tensor of size [N, 4] with coordinates and intensities
          labesl: torch tensor of size [N,] with labels
      """

    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.tensor(
            range(
                size[0])), torch.tensor(
            range(
                size[1])), torch.tensor(
            range(
                size[2])))

    raw_point_cloud = torch.cat((grid_x.unsqueeze(-1).float(), grid_y.unsqueeze(-1).float(),
                                 grid_z.unsqueeze(-1).float(), torch.tensor(brain).float().unsqueeze(-1).float()), -1)

    point_cloud_fcd = raw_point_cloud[mask == 1, :]
    fcd_amount = point_cloud_fcd.shape[0]
    random_fcd_idxs = np.random.choice(range(fcd_amount), fcd_amount // 100, replace=False)
    point_cloud_fcd = point_cloud_fcd[random_fcd_idxs]

    point_cloud_no_fcd = raw_point_cloud[(mask == 0) * (brain != 0), :]
    no_fcd_amount = point_cloud_no_fcd.shape[0]
    random_no_fcd_idxs = np.random.choice(range(no_fcd_amount), no_fcd_amount // 100, replace=False)
    point_cloud_no_fcd = point_cloud_no_fcd[random_no_fcd_idxs]

    return torch.cat([point_cloud_fcd, point_cloud_no_fcd]), np.array([1] * fcd_amount + [0] * no_fcd_amount)


def filenames_to_point_cloud_and_labels(file, file_mask, size):
    """
    Procceses filenames of brain and mask into point cloud with labels
      Args:
          file: path to brain file
          file_mask: path to mask file
          size: size of the input tensors
      Output:
          point cloud: torch tensor of size [N, 4] with coordinates and intensities
          labesl: torch tensor of size [N,] with labels
      """

    brain = load_nii_to_array(file)
    mask = load_nii_to_array(file_mask)
    point_cloud, labels = brain_and_mask_to_point_cloud_and_labels(brain, mask, size=size)

    return point_cloud, labels


def bran_and_seg_to_point_cloud_and_labels_grey(brain, seg, size=(260, 311, 260)):
    """
    Transforms 3d tensors of brain into pointcloud and labels for it
      Args:
          brain: torch tensor of size 'size' with intensity value at the positions with brain and 0 otherwise
          size: size of the input tensors
          seg: torch tensor of size 'size' with segmentation codes at all the positions of brain
      Output:
          point cloud: torch tensor of size [N, 4] with coordinates and intensities
          labesl: torch tensor of size [N,] with labels
      """
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.tensor(
            range(
                size[0])), torch.tensor(
            range(
                size[1])), torch.tensor(
            range(
                size[2])))

    raw_point_cloud = torch.cat((grid_x.unsqueeze(-1).float(), grid_y.unsqueeze(-1).float(),
                                 grid_z.unsqueeze(-1).float(), torch.tensor(brain).float().unsqueeze(-1).float()), -1)

    pc_grey_matter = raw_point_cloud[seg >= 1000, :]
    grey_matter_amount = pc_grey_matter.shape[0]
    random_grey_matter_idx = np.random.choice(range(grey_matter_amount), grey_matter_amount // 100, replace=False)
    pc_grey_matter = pc_grey_matter[random_grey_matter_idx]

    pc_no_grey_matter = raw_point_cloud[(seg < 1000) * (brain != 0), :]
    no_grey_matter_amount = pc_no_grey_matter.shape[0]
    random_no_grey_matter_idx = np.random.choice(
        range(no_grey_matter_amount),
        no_grey_matter_amount // 100,
        replace=False)
    pc_no_grey_matter = pc_no_grey_matter[random_no_grey_matter_idx]

    return torch.cat([pc_grey_matter, pc_no_grey_matter]), np.array(
        [1] * grey_matter_amount + [0] * no_grey_matter_amount)


def grey_filenames_to_pc_and_labels(file_brain, file_segmentation, size=(260, 311, 260)):
    """
    Procceses filename of brain and segmentation into point cloud with labels
      Args:
          file_brain: path to brain file
          file_segmentation: path to segmentation file
          size: size of the input tensors
      Output:
          point cloud: torch tensor of size [N, 4] with coordinates and intensities
          labesl: torch tensor of size [N,] with labels
      """

    brain = load_nii_to_array(file_brain)
    seg = load_nii_to_array(file_segmentation)
    point_cloud, labels = bran_and_seg_to_point_cloud_and_labels_grey(brain, seg, size=size)

    return point_cloud, labels
