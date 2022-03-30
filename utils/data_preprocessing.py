"""This module contains utility functions for data preprocessing."""
import nibabel as nib
import nilearn
import numpy as np
from nilearn import image


def load_nii_to_array(nii_path: str):
    """
    Function returns np.array data from the *.nii file.

    Parameters
        nii_path: path to *.nii file with data;

    Returns
        data: data obtained from nii_path.
    """
    try:
        data = np.asanyarray(nib.load(nii_path).dataobj)
        return data
    except OSError:
        print(FileNotFoundError(f"No such file or no access: {nii_path}"))
        return ""


def register_image_to_template(
    path_to_image: str,
    template_path: str,
    transform_path: str,
    output_path: str,
    interpolation: str = "Linear",
):
    """
    Register image to the template.

    To use run this lines from server terminal:
    * docker run -p 8777:8777 --rm -it -v ~/fcd_detection:/input --entrypoint=bash nipreps/fmriprep:20.2.1
    * cd ..
    * cd input
    * pip install jupyter
    * jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 8777
    Then go to the - http://ironman:8777 and use this function.
    Sometimes it takes several times to repeat this procedure to kernel starts working.

    Parameters:
        path_to_image: path to an image;
        template_path: path to a template;
        transform_path: path to a transformation;
        output_path: path to wrtite output;
        interpolation: type of interpolation;

    Returns:
        Registered image.
    """
    at = ApplyTransforms()
    at.inputs.input_image = path_to_image
    at.inputs.reference_image = template_path
    at.inputs.transforms = transform_path
    at.inputs.output_image = output_path
    at.inputs.interpolation = interpolation
    at.run()


def resize_image_to_template(
    path_to_image: str,
    template_path: str,
    output_path: str,
):
    """
    Resize image to the template shape.

    Parameters:
        path_to_image: path to an image;
        template_path: path to a template;
        output_path: path to wrtite output.

    Returns:
        Resized image.
    """
    image = nib.load(path_to_image)
    template = nib.load(template_path)
    resized_image = nilearn.image.resample_to_img(image, template)
    nib.save(resized_image, output_path)
