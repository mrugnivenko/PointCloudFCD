import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import (auc, precision_recall_curve)
from tqdm import tqdm

from torch.utils.data import DataLoader
from models.losses import MultiShapeCrossEntropy

import utils.data_utils as d_utils
from utils.crop_new_2 import get_loader_crop
from utils.data_preprocessing import *
from utils.lr_scheduler import get_scheduler
from utils.model_etc import *
from utils.util import AverageMeter, get_confusion_matrix
from utils.config import Config
from utils.metrics import calculate_contrast
from utils.validation import validate_one_epoch
from utils.logger import Logger


def train(
    data_dict: dict, 
    kf: dict, 
    config: Config, 
    experiment, 
    IS_EXPERIMENT: bool
) -> None:
    """
    Perform training procedure.
    
    Parameters:
        data_dict:
        kf:
        config: 
        experiment: 
        IS_EXPERIMENT:     
    """
    logger = Logger(experiment)
    dict_fold_metrics = {}
    dict_avg_metrics = {
        "train_loss": np.zeros(config.epochs),
        "val_loss": np.zeros(1 + config.epochs // config.val_freq),
        "train_pr_auc": np.zeros(config.epochs),
        "val_AUC-PR": np.zeros(1 + config.epochs // config.val_freq),
        "val_contrast": np.zeros(1 + config.epochs // config.val_freq),
        "val_recall": np.zeros(1 + config.epochs // config.val_freq),
        "val_IoU": np.zeros(1 + config.epochs // config.val_freq),
        "val_dice": np.zeros(1 + config.epochs // config.val_freq),
    }

    for e in tqdm(kf.keys()):

        train_dict = {
            key: [x for x in data_dict[key] if x.split("/")[-1][:-7] in kf[str(e)][0]]
            for key in data_dict
        }
        test_dict = {
            key: [x for x in data_dict[key] if x.split("/")[-1][:-7] in kf[str(e)][1]]
            for key in data_dict
        }

        out = get_loader_crop(
            config=config,
            batch_size=config.batch_size,
            num_points=config.num_points,
            train_dict=train_dict,
            test_dict=test_dict,
            crop_size=config.crop_size,
            return_abs_coords=config.is_return_absolute_coordinates,
            return_pc_without_air_points=config.get_rid_of_air_points,
            coin_flip_threshold=config.coin_flip_threshold,
            weighted_loss=config.weighted_loss,
        )

        if config.weighted_loss:
            train_loader, test_loader, weights = out
        else:
            train_loader, test_loader = out
            weights = None
        print(f"size of train dataset: {len(train_loader.dataset)}")
        print(f"size of test dataset: {len(test_loader.dataset)}")

        model, criterion = build_multi_part_segmentation(
            config=config, weights=weights, type=config.loss
        )
        #         if config.FINE_TUNE:
        #             model.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH))
        model.cuda()
        criterion.cuda()
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, len(train_loader), config)

        dict_train_metrics = {
            "loss": [],
            "pr_auc": [],
        }

        dict_val_metrics = {
            "loss": [],
            "AUC-PR": [],
            "IoU": [],
            "dice": [],
            "contrast": [],
            "recall": [],
        }

        for epoch in tqdm(range(1, config.epochs + 1)):
            training_result = train_one_epoch(
                epoch, train_loader, model, criterion, optimizer, scheduler, config
            )

            dict_train_metrics["loss"].append(training_result[0])
            dict_train_metrics["pr_auc"].append(training_result[1])

            if config.is_experiment:
                logger.log_train_results(training_result, epoch, int(e))

            if epoch == 1 or epoch % config.val_freq == 0:
                metrics_dict, confusion_matrix = validate_one_epoch(
                    epoch, test_loader, model, criterion, config
                )
                for name, value in metrics_dict.items():
                    dict_val_metrics[name].append(value)

                if config.is_experiment:
                    logger.log_val_results(metrics_dict, confusion_matrix, epoch, int(e))

        torch.save(
            model.state_dict(),
            f"experiments/{config.name_of_experiment}/weights/{int(e) + 1}_fold.pth",
        )

        del model, train_loader, test_loader

        dict_fold_metrics[int(e)] = {
            "train": dict_train_metrics,
            "val": dict_val_metrics,
        }

    if config.is_experiment:

        for k in range(5):
            for mode in ["train", "val"]:
                for metric in dict_fold_metrics[k][mode]:
                    dict_avg_metrics[f"{mode}_{metric}"] += np.array(
                        dict_fold_metrics[k][mode][metric]
                    )
        for metric in dict_avg_metrics:
            dict_avg_metrics[metric] /= 5

        for name, value in dict_avg_metrics.items():
            if "train" in name:
                number = config.epochs + 1
            else:
                number = config.epochs // config.val_freq + 2
            for t, epoch in enumerate(range(1, number)):
                experiment.log_metric(f"avg {name}", value[t], step=epoch)

    experiment.end()

def train_one_epoch(
    epoch: int, 
    train_loader: DataLoader, 
    model: MultiPartSegmentationModel, 
    criterion: MultiShapeCrossEntropy, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler, 
    config: Config,
) -> tuple:
    """
    Perform training on single epoch.

    Parameters:
        epoch: Epoch number;
        train_loader: Loader for train data;
        model: Model;
        criterion: Loss function;
        optimizer: Optimizer;
        scheduler: Scheduler;
        config: Dictionary with all the parameters of model.

    Returns:
        Avarage loss over one epoch and ROC-AUC score.
    """

    model.train()
    loss_meter = AverageMeter()
    softmax = torch.nn.Softmax(dim=1)

    pred_soft_flats = []
    points_labels_flats = []

    for idx, batch in enumerate(train_loader):
        points, mask, points_labels, _ = [
            batch[key]
            for key in ["current_points", "mask", "current_points_labels", "label"]
        ]
        bsz = points.size(0)
        features = points.transpose(1, 2).contiguous()

        points = points[:, :, :3].cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)

        pred = model(points, mask, features)  # [bsz, 2, num_points]
        loss = criterion(pred, points_labels)

        pred_soft_flats += list(
            softmax(pred[0])[:, 1, :].reshape(-1).detach().cpu().numpy()
        )
        points_labels_flats += list(points_labels.reshape(-1).detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), bsz)

        del points, mask, points_labels, loss, batch

    precision, recall, thresholds = precision_recall_curve(
        points_labels_flats, pred_soft_flats
    )
    auc_precision_recall = auc(recall, precision)

    return [loss_meter.avg, auc_precision_recall, optimizer.param_groups[0]["lr"]]
