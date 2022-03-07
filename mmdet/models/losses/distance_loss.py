# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

from .kd_loss import KnowledgeDistillationKLDivLoss 


@LOSSES.register_module()
class SymKnowledgeDistillationKLDivLoss(KnowledgeDistillationKLDivLoss):

    def __init__(self, reduction='mean', loss_weight=1.0, T=10, detach_target=True):
        super(SymKnowledgeDistillationKLDivLoss, self).__init__(reduction, loss_weight, T, detach_target)

    def forward(self,
                pred1,
                pred2,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        l1 = super().forward(pred1, pred2, weight, avg_factor, reduction_override)
        l2 = super().forward(pred2, pred1, weight, avg_factor, reduction_override)
        return l1+l2



@LOSSES.register_module()
class DistanceLoss(nn.Module):
    def __init__(self,  loss_type, loss_weight=1.0,  *args, **kwargs):
        super(DistanceLoss, self).__init__()
        loss_func = getattr(F, loss_type)
        self.loss = loss_func(*args, **kwargs)
        self.loss_weight = loss_weight
        print(self.loss)

    def forward(self, *args, **kwargs):
        print()
        xx
        return self.loss(*args, **kwargs)*self.loss_weight


