import numpy as np
import pandas as pd

from utils.crop import *
from utils.data_processor import load_nii_to_array

def calculate_contrast(points_labels_flats, pred_soft_flats):
    """
    Function calculates contrast metric 
    
    :params points_labels_flats: list, list with points labels - 1 for FCD point and 0 for no-FCD point
    :params pred_soft_flats: list, list with probabilities of each point to be FCD point 
    
    :outputs contrast: float, contrast metric
    """
    
    probability_inside_fcd = np.dot(points_labels_flats, pred_soft_flats) / np.sum(points_labels_flats)
    probability_outside_fcd = np.dot(np.ones(len(points_labels_flats)) - np.array(points_labels_flats), pred_soft_flats,) / (len(points_labels_flats) - np.sum(points_labels_flats))
    
    return (probability_inside_fcd - probability_outside_fcd) / (probability_inside_fcd + probability_outside_fcd)

def top10_f(pred, label, coords, crop_size = 64):
    """
        pred - (N,)
        label - (N,)
        coords = (N, 3)
    """

    df = pd.DataFrame({'pred':pred,'label':label,'coord1':coords[:,0],'coord2':coords[:,1],'coord3':coords[:,2]})
    
    for i in range(1,4):
        df[f'coord{i}'] = df[f'coord{i}'] // crop_size

    df = df.groupby([f'coord{i}' for i in range(1,4)]).mean().reset_index()
    df.label = 1 - df.label

    df = df.sort_values(['pred','label'], ascending = False).reset_index(drop = True).head(10)
    df = df[df.label!=1]
    if df.shape[0] == 0:
        top10 = 0.0
    else:
        top10 = 1 - float(df.head(1).index.values[0]) / 10
    return top10

def percent_of_high_score(df, threshold):
    
    result = []
    for column in df.columns:
        result.append(df[df[column] >= threshold][column].count()/df[column].count())
    
    return result

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