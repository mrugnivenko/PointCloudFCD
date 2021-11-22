import numpy as np
import pandas as pd
from data_processor import load_nii_to_array
from crop import *

def score(descending_probability_and_amount_of_fcd: list) -> np.float:
    idx = next((index for index,
                value in enumerate(descending_probability_and_amount_of_fcd) if value != 0), None)
    if idx == None:
        return 0
    else:
        return (10-idx)*10
    
def calculate_score(brain_path: str,
                    prediction_path: str,
                    label_path: str,
                    crop_size = 64,
                    step_size = 64) -> float:
    
    brain = np.load(brain_path)
    prediction = load_nii_to_array(prediction_path)
    label = np.load(label_path)
    
    if label.sum() == 0:
        return None
    
    sagittal_shape, coronal_shape, axial_shape = brain.shape
    
    deltas = []
    for shape in [sagittal_shape, coronal_shape, axial_shape]:
        if shape % crop_size != 0:
            deltas.append((0, (shape // crop_size + 1) * crop_size - shape))
        else:
            deltas.append((0, 0))

    brain = np.pad(brain, deltas, "constant", constant_values=0)
    prediction = np.pad(prediction, deltas, "constant", constant_values=0)
    label = np.pad(label, deltas, "constant", constant_values=0)

    _, center_coords = get_inference_crops(
        {'brains': brain}, crop_size=crop_size, step_size=step_size
    )
    
    df = pd.DataFrame(columns=['probability', 'is_fcd'])

    for i, crop_point in enumerate(center_coords):
        pred_crop = prediction[crop_point[0]:crop_point[0]+crop_size,
                               crop_point[1]:crop_point[1]+crop_size,
                               crop_point[2]:crop_point[2]+crop_size,
                              ]

        label_crop = label[crop_point[0]:crop_point[0]+crop_size,
                         crop_point[1]:crop_point[1]+crop_size,
                         crop_point[2]:crop_point[2]+crop_size,
                        ]
        
        df.loc[i] = [pred_crop.sum(), label_crop.sum()]
        
    df = df.sort_values(by='probability', ascending=False)[:10]
    return score(df.is_fcd.tolist())