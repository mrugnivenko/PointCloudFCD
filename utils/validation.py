import torch
import math
import numpy as np
from sklearn.metrics import (auc, precision_recall_curve)
import utils.data_utils as d_utils
from utils.util import AverageMeter, get_confusion_matrix
from utils.metrics import calculate_contrast

def validate_one_epoch(epoch, test_loader, model, criterion, config, num_votes=1):
    """
    Validating one epoch

     epoch: int, epoch number
     test_loader: torch.utils.data.dataloader.DataLoader, loader for test data
     model: model
     criterion: criterion
     config:  dict, dictionary with all the parameters of model
     num_votes: int, #######################################
     log_confusion_matrix: bool, #######################################

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

            points_orig, mask, points_labels, shape_labels = [
                batch[key]
                for key in ["current_points", "mask", "current_points_labels", "label"]
            ]
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

                pred_soft_flats += list(
                    np.array(softmax(pred[0])[:, 1, :].reshape(-1).detach().cpu())
                )
                points_labels_flats += list(
                    np.array(points_labels.reshape(-1).detach().cpu())
                )

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
                        vote_logits[i] = vote_logits[i] + (
                            batch_logits[i] - vote_logits[i]
                        ) / (vote + 1)

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
        all_masks,
    )

    IoU = confs[1, 1] / (confs[1, 1] + confs[1, 0] + confs[0, 1] + 1e-8)
    DICE_score = 2 * confs[1, 1] / (2 * confs[1, 1] + confs[1, 0] + confs[0, 1] + 1e-8)

    contrast = calculate_contrast(points_labels_flats, pred_soft_flats)
    if math.isnan(contrast):
        contrast = 0

    precision, recall, thresholds = precision_recall_curve(
        points_labels_flats, pred_soft_flats
    )
    auc_precision_recall = auc(recall, precision)
    if math.isnan(auc_precision_recall):
        auc_precision_recall = 0

    recall = confs[1, 1] / (confs[1, 1] + confs[1, 0] + 1e-8)

    metrics = {
        "IoU": IoU,
        "dice": DICE_score,
        "contrast": contrast,
        "recall": recall,
        "AUC-PR": auc_precision_recall,
        "loss": losses.avg,
    }

    return metrics, confs
