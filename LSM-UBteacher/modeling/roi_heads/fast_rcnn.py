# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs,
)

# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas, (scale_scores, _) = predictions
        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            scale_scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()

        return losses


class FastRCNNFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        scale_pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        num_classes=80,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes
        self.scale_pred_class_logits = scale_pred_class_logits
        # print('self_num_classes:', self.num_classes)
    def losses(self):
        return {
            "loss_cls": self.comput_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
            # "loss_cls_distill": self.comput_distill_loss() * 0.2,
        }

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            # print(self.gt_scores)
            mask = self.gt_scores > 0.7
            total_loss = FC_loss(input=self.pred_class_logits[mask], target=self.gt_classes[mask])
            total_loss = total_loss / self.gt_classes[mask].shape[0]

            return total_loss
    def comput_distill_loss(self):
        mask = self.gt_scores <= 0.8
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        elif len(self.pred_class_logits[mask]) > 0:
            
            dis_label = F.softmax(self.scale_pred_class_logits[mask], 1)
            dis_label = dis_label.detach()
            logit = self.pred_class_logits[mask]
            logit = F.softmax(logit, -1)
            CE = - torch.log(logit) * dis_label
            CE = CE.sum(-1)
            p = torch.exp(-CE)
            loss = (1 - p) * CE
            return loss.sum() / dis_label.shape[0]
        else:
            return 0
class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()
