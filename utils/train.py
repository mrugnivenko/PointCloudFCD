import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as data
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc

import datasets.data_utils as d_utils
from utils.util import AverageMeter, get_confusion_matrix
from utils.lr_scheduler import get_scheduler
from utils.crop import get_loader_crop
from utils.model_etc import *
from utils.data_processor import *


def find_optimal_cutoff(target, predicted):
    """
    Find the optimal probability cutoff point for a classification model related to event rate
    
    :params target: ###################################
    :params predicted: ###################################
    
    :outputs ###################################
    """
    
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {
            "tf": pd.Series(tpr - (1 - fpr), index=i),
            "tpr": pd.Series(tpr, index=i),
            "fpr": pd.Series(fpr, index=i),
            "threshold": pd.Series(threshold, index=i),
        }
    )
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return (
        np.mean(list(roc_t["threshold"])),
        list(roc_t["tpr"])[0],
        list(roc_t["fpr"])[0],
    )

def train_one_epoch(
    epoch,
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    config
):
    """
    Training one epoch
    
    :params epoch: int, epoch number
    :params train_loader: torch.utils.data.dataloader.DataLoader, loader for train data
    :params model: model 
    :params criterion: loss function
    :params optimizer: optimizer 
    :params scheduler: scheduler
    :params config: dict, dictionary with all the parameters of model
    
    :outputs avarage loss over one epoch, optimal cutoff for binarization, ROC-AUC score
    """
    
    model.train()
    loss_meter = AverageMeter()
    softmax = torch.nn.Softmax(dim=1)
    
    pred_soft_flats = []
    points_labels_flats = []
    
    for idx, batch in enumerate(train_loader):
        points, mask, points_labels, _ = [batch[key] for key in ["current_points", "mask", "current_points_labels", "label"]]
        
        bsz = points.size(0)
        features = points.transpose(1, 2).contiguous()
        
        points = points[:, :, :3].cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)
        
        pred = model(points, mask, features) # [bsz, 2, num_points]
        loss = criterion(pred, points_labels)

        pred_soft_flats += list(softmax(pred[0])[:, 1, :].reshape(-1).detach().cpu().numpy())
        points_labels_flats += list(points_labels.reshape(-1).detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), bsz)
        
        del points, mask, points_labels, loss, batch
        
    precision, recall, thresholds = precision_recall_curve(points_labels_flats, pred_soft_flats)
    auc_precision_recall = auc(recall, precision)
    
    return loss_meter.avg, auc_precision_recall

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
        
def validate_one_epoch(
    epoch,
    test_loader,
    model,
    criterion,
    config,
    num_votes=1
):
    """
    Validating one epoch
    
    :params epoch: int, epoch number
    :params test_loader: torch.utils.data.dataloader.DataLoader, loader for test data
    :params model: model
    :params criterion: criterion
    :params config:  dict, dictionary with all the parameters of model
    :params num_votes: int, #######################################
    :params log_confusion_matrix: bool, #######################################
    
    :outputs #######################################
    """

    model.eval()
    losses = AverageMeter()
    softmax = torch.nn.Softmax(dim=1)
    
    transform = d_utils.BatchPointcloudScaleAndJitter(
            scale_low=config.scale_low,
            scale_high=config.scale_high,
            std=config.noise_std,
            clip=config.noise_clip,
        )
    
    with torch.no_grad():
        
        all_logits = []
        all_points_labels = []
        all_shape_labels = []
        all_masks = []
        
        pred_soft_flats = []
        points_labels_flats = []
        
        for idx, batch in enumerate(test_loader):
            
            points_orig, mask, points_labels, shape_labels = [batch[key] for 
                                                              key in ["current_points", "mask", "current_points_labels", "label"]]
            
            vote_logits = None
            vote_points_labels = None
            vote_shape_labels = None
            vote_masks = None
            
            for vote in range(num_votes):
                
                batch_logits = []
                batch_points_labels = []
                batch_shape_labels = []
                batch_masks = []
                
                # augment for voting
                if vote > 0:
                    points = transform(points_orig)
                else:
                    points = points_orig
                    
                # forward
                bsz = points.size(0)
                features = points.transpose(1, 2).contiguous()
                
                points = points[:, :, :3].cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                features = features.cuda(non_blocking=True)
                points_labels = points_labels.cuda(non_blocking=True)
                shape_labels = shape_labels.cuda(non_blocking=True)

                pred = model(points, mask, features)
                loss = criterion(pred, points_labels)
                
                losses.update(loss.item(), bsz)

                pred_soft_flats += list(np.array(softmax(pred[0])[:, 1, :].reshape(-1).detach().cpu()))
                points_labels_flats += list(np.array(points_labels.reshape(-1).detach().cpu()))

                # collect
                for ib in range(bsz):
                    
                    sl = shape_labels[ib]
                    logits = pred[sl][ib]
                    pl = points_labels[ib]
                    pmk = mask[ib]
                    
                    batch_logits.append(logits.cpu().numpy())
                    batch_points_labels.append(pl.cpu().numpy())
                    batch_shape_labels.append(sl.cpu().numpy())
                    batch_masks.append(pmk.cpu().numpy().astype(np.bool))

                if vote_logits is None:
                    vote_logits = batch_logits
                    vote_points_labels = batch_points_labels
                    vote_shape_labels = batch_shape_labels
                    vote_masks = batch_masks
                else:
                    for i in range(len(vote_logits)):
                        vote_logits[i] = vote_logits[i] + (batch_logits[i] - vote_logits[i]) / (vote + 1)

            all_logits += vote_logits
            all_points_labels += vote_points_labels
            all_shape_labels += vote_shape_labels
            all_masks += vote_masks
            
            del points_orig, mask, points_labels, shape_labels, loss
        

    confs = get_confusion_matrix(
        config.num_classes,
        config.num_parts,
        all_shape_labels,
        all_logits,
        all_points_labels,
        all_masks
    )
    
    IoU = confs[1, 1] / (confs[1, 1] + confs[1, 0] + confs[0, 1])
    DICE_score = 2 * confs[1, 1] / (2 * confs[1, 1] + confs[1, 0] + confs[0, 1]) 
    contrast = calculate_contrast(points_labels_flats, pred_soft_flats)
    precision, recall, thresholds = precision_recall_curve(points_labels_flats, pred_soft_flats)
    auc_precision_recall = auc(recall, precision)
    recall = confs[1, 1] / (confs[1, 1] + confs[1, 0]) 
    
    metrics = {
        "IoU": IoU,
        "dice": DICE_score,
        "contrast": contrast,
        "recall": recall,
        "AUC-PR": auc_precision_recall,
        "loss": losses.avg,
    }
    
    return metrics, confs

def log_metrics(experiment, metrics_names, metrics_values, epoch):
    for name, value in zip(metrics_names, metrics_values):
        experiment.log_metric(name, value, epoch=epoch)
        
def train(data_dict, kf, config, experiment, IS_EXPERIMENT):
    
    for e, (train_idxs, test_idxs) in tqdm(enumerate(kf.split(data_dict['brains']))):  
        train_dict = {key: [data_dict[key][idx] for idx in train_idxs] for key in data_dict}
        test_dict = {key: [data_dict[key][idx] for idx in test_idxs] for key in data_dict}

        out = get_loader_crop(config=config,
                              batch_size=config.BATCH_SIZE,
                              num_points=config.num_points,
                              train_dict=train_dict,
                              test_dict=test_dict,
                              crop_size=config.CROP_SIZE,
                              return_abs_coords=config.IS_RETURN_ABS_COORDS,
                              return_pc_without_air_points=config.IS_RETURN_PC_WITHOUT_AIR_POINTS,
                              coin_flip_threshold = config.COIN_FLIP_THRESHOLD,
                              weighted_loss = config.WEIGHTED_LOSS
                             )

        if config.WEIGHTED_LOSS:
            train_loader, test_loader, weights = out
        else:
            train_loader, test_loader = out
            weights = None
        print(f"size of train dataset: {len(train_loader.dataset)}")
        print(f"size of test dataset: {len(test_loader.dataset)}")

        model, criterion = build_multi_part_segmentation(config=config, weights=weights, type=config.LOSS_TYPE)
        if config.FINE_TUNE:
            model.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH))
        model.cuda()
        criterion.cuda()    
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, len(train_loader), config)

        for epoch in tqdm(range(1, config.EPOCHS + 1)):
            loss, pr_auc = train_one_epoch(epoch, train_loader, model, criterion, optimizer, scheduler, config)

            if IS_EXPERIMENT:
                log_metrics(experiment, 
                            [f"train PR-AUC (fold #{e + 1})",
                             f"learning rate (fold #{e + 1})",
                             f"train loss (fold #{e + 1})"], 
                            [pr_auc,
                             optimizer.param_groups[0]['lr'],
                             loss],
                             epoch
                           )

            if epoch == 1 or epoch % config.LOGGING_PERIOD == 0:
                metrics_dict, confusion_matrix = validate_one_epoch(epoch, test_loader, model, criterion, config)
                if IS_EXPERIMENT:   
                    for name, value in metrics_dict.items():
                        experiment.log_metric(f"test {name} (fold #{e + 1})", value, epoch=epoch)
                    experiment.log_confusion_matrix(title=f"Test confusion matrix", 
                                                    file_name=f"confusion-matrix-fold_{e+1}-epoch_{epoch}.json",
                                                    matrix=confusion_matrix, labels=['No FCD', 'FCD'])

        torch.save(model.state_dict(), f'experiments/{config.EXP_NAME}/weights/{e + 1}_fold.pth')  

        del model, train_loader, test_loader
        
    experiment.end()