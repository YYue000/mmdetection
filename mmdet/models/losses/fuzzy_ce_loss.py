import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def _expand_onehot_labels_fz(labels, label_weights, label_channels, ignore_index, fz=None):
    """Expand onehot labels to match the size of prediction."""
    assert label_channels > 1
    if fz is None:
        _f = 0
        fg_score = 1
        bg_score = 0
    elif fz == 'mixup_0.5':
        eps = 0.1
        fg_score = 0.5
        bg_score = 1-fg_score-eps
        if label_channels == 2:
            fg_score = fg_score+eps
            _f = 0
        else:
            _f = eps/(label_channels-2)
    else:
        raise NotImplementedError

    bin_labels = torch.ones((labels.size(0), label_channels), device=labels.device)*_f
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = fg_score
        bin_labels[inds, 0] = bg_score

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def fz_binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         fz=None):
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels_fz(label, weight, pred.size(-1),
                                              ignore_index, fz)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


