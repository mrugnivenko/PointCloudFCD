import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import griddata
from tqdm import tqdm


def points2mri(points, labels, mri_shape=(241, 293, 247), method="nearest"):

    """
    method=["nearest", "linear"]
    """

    x = np.arange(0, mri_shape[0])
    y = np.arange(0, mri_shape[1])
    z = np.arange(0, mri_shape[2])
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    A = set(map(tuple, np.column_stack((x.ravel(), y.ravel(), z.ravel())).tolist()))
    points = np.rint(points)

    for i in range(3):
        points[:, i] = np.clip(points[:, i], 0, mri_shape[i] - 1)

    points = points.astype(int).tolist()
    xi = np.array(list(A - set(map(tuple, points))))
    x, y, z = np.concatenate([xi, points]).T
    mri_3d = np.zeros(mri_shape)

    xi_labels = griddata(points=points, values=labels, xi=xi, method=method)
    mri_3d[x, y, z] = np.concatenate([xi_labels, labels])

    return mri_3d
