'''
Description of the following functions:
    * load_nii_to_array
'''

import numpy as np
import nibabel as nib


def load_nii_to_array(nii_path):
    """
    Function returns np.array data from the *.nii file

    Arguments:
        * nii_path (str): path to *.nii file with data

    Output:
        * data (np.array): data obtained from nii_path
    """

    try:
        data = np.asanyarray(nib.load(nii_path).dataobj)
        return (data)

    except OSError:
        print(FileNotFoundError(f'No such file or no access: {nii_path}'))
        return('')
