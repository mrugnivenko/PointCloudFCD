import json
import torch
import numpy as np 
from tqdm import tqdm 

from utils.crop import *
from utils.model_etc import *
import datasets.data_utils as d_utils

def inference(config, all_data_dict, kf, num_votes=1, repeat=1):
    
    softmax = torch.nn.Softmax(dim=0)

    transform_for_vote = d_utils.BatchPointcloudScaleAndJitter(
        scale_low=config.scale_low,
        scale_high=config.scale_high,
        std=config.noise_std,
        clip=config.noise_clip
    )
    
    with torch.no_grad():
        
        for e, (_, test_idxs) in tqdm(enumerate(kf.split(all_data_dict['brains'])), total=kf.n_splits):  
            test_dicts = [{key: all_data_dict[key][idx] for key in all_data_dict} for idx in test_idxs]

            model, _ = build_multi_part_segmentation(config=config)
            model.load_state_dict(torch.load(f'experiments/{config.EXP_NAME}/weights/{e + 1}_fold.pth'))
            model.eval()
            model.cuda()

            data_loaders = [BrainDataSegCrop(config=config, num_points=config.num_points,
                                            task='test',
                                            data_dict = test_dict,
                                            crop_size = config.CROP_SIZE,
                                            return_air_mask_test = True,
                                            return_abs_coords = config.IS_RETURN_ABS_COORDS,
                                            return_pc_without_air_points = config.IS_RETURN_PC_WITHOUT_AIR_POINTS, 
                                            MEANS = config.MEANS,
                                            STDS = config.STDS
                                           ) for test_dict in test_dicts]

            for test_brain, test_idx in tqdm(zip(data_loaders, test_idxs), total=len(test_idxs)):

                subject = all_data_dict['brains'][test_idx].split('/')[-1].split('.')[0]
                
                points_orig_flats = []
                points_labels_flats = []
                pred_soft_flats = []
                air_masks_flats = []
                center_coords_flats = []
                
                for crop in test_brain:
                    for _ in range(repeat):
                        points_orig, mask, points_labels, air_mask = [
                            crop[key] for key in ["current_points", "mask", "current_points_labels", 'air_mask']]
                        
                        vote_logits = None
                        vote_points_labels = None
                        vote_masks = None
                        
                        points_orig = points_orig.unsqueeze(0)
                        mask = mask.unsqueeze(0)
                        points_labels = points_labels.unsqueeze(0)
                        
                        preds = []
                        
                        for v in range(num_votes):
                            
                            batch_logits = []
                            batch_points_labels = []
                            batch_masks = []
                            
                            if v > 0:
                                points = transform_for_vote(points_orig)
                            else:
                                points = points_orig
                           
                            features = points.transpose(1, 2).contiguous()
                            features = features.cuda(non_blocking=True)
                            points = points[:, :, :3].cuda(non_blocking=True)
                            mask = mask.cuda(non_blocking=True)
                            points_labels = points_labels.cuda(non_blocking=True)
                            
                            pred = model(points, mask, features)
                            preds.append(pred[0])
                            
                        preds = torch.cat(preds).mean(dim=0)

                        points_orig = points_orig.squeeze(0)                        
                        pred_soft_flats += softmax(preds)[1,:].reshape(-1).detach().cpu().numpy().tolist()
                        points_labels_flats += points_labels.reshape(-1).detach().cpu().numpy().tolist()
                        air_masks_flats += air_mask.reshape(-1).tolist()
                        
                        if config.IS_RETURN_ABS_COORDS:
                            size_abs = (241, 336, 283)
                            means = np.array([x // 2 for x in size_abs])
                            half_range = np.array([x // 2 for x in size_abs])
                            points_orig_flats += (points_orig[:, :3].detach().cpu().numpy() * half_range + means).tolist()
                        else:
                            pass
                        
                    result = {'coordinates': points_orig_flats,
                              'predictions': pred_soft_flats,
                              'labels': points_labels_flats,
                              'air_maks': air_masks_flats
                             }
                    
                with open(f'experiments/{config.EXP_NAME}/predictions/{subject}.json', 'w') as file:
                    json.dump(result, file)
            del data_loaders, test_dicts