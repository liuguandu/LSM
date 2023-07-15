# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
import torch.nn.functional as F
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from ubteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        features2: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, features2, proposals, compute_loss, compute_val_loss, branch
            )
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, features2, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        features2: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)

            # box_in_fea2 = ['p5', 'p2', 'p3', 'p4']
            # features2 = [features2[f] for f in box_in_fea2]
            # box_features2 = self.box_pooler(features2, [x.proposal_boxes for x in proposals])
            # box_features2 = self.box_head(box_features2)
            # # box_features2 = torch.cat((box_features2, big_feature), 1)
            # # box_features2 = self.cross_layer(box_features2)
            # # print('box_features2: ', box_features2.size())
            # predictions2 = self.box_predictor(box_features2)
            # pred_logit1, pred_delta1, _ = predictions
            # # _, _, (pred_logit2, pred_delta2) = predictions2
            # pred_logit2, pred_delta2, _ = predictions2
            # # print('pred_logit1:', box_features.size(), 'pred_logit2:', box_features2.size())
            # proposal2 = torch.cat([p.gt_classes for p in proposals], dim=0)
            # gt_scores = torch.cat([p.gt_scores for p in proposals], dim=0)
            # gt_scores = gt_scores[int(len(pred_logit1)/4):]
            # mask = gt_scores > 0.7
            # klloss = F.cross_entropy(pred_logit2[int(len(pred_logit1)/4):, :], proposal2[int(len(pred_logit1)/4):], reduction='mean')
            # # p = torch.exp(-klloss)
            # # k_loss = (1 - p) ** 1 * klloss
            # # losses['loss_twoKL'] = k_loss.sum() / proposal2[int(len(pred_logit1)/4):].size(0) * 0.2
            # losses['loss_twoKL'] = klloss * 0.5
            
            # box_type = type(proposals[0].proposal_boxes)
            # gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            # proposal_boxes = box_type.cat([p.proposal_boxes for p in proposals])
            # box_transform = Box2BoxTransform((10.0, 10.0, 5.0, 5.0))
            # gt_pred_deltas = box_transform.get_deltas(proposal_boxes.tensor, gt_boxes.tensor)
            # pred_delta2 = pred_delta2[int(len(pred_delta2)/4):, :]
            # gt_pred_deltas = gt_pred_deltas[int(len(gt_pred_deltas)/4):, :]
            # proposal2 = proposal2[int(len(pred_logit1)/4):]
            # # lscore = lscore[int(len(pred_logit1)/4):]
            # fg_inds = nonzero_tuple((proposal2 >= 0) & (proposal2 < 80))[0]
            # fg_gt_classes = proposal2[fg_inds]
            # gt_class_cols = 4 * fg_gt_classes[:, None] + torch.arange(4, device=pred_logit2.device)
            # loss_box_reg = smooth_l1_loss(
            #     pred_delta2[fg_inds[:, None], gt_class_cols],
            #     gt_pred_deltas[fg_inds],
            #     0.0,
            #     reduction="sum",
            # )
            # losses['loss_twoKL_reg'] = loss_box_reg * 0.5 / max(proposal2.numel(), 1.0)
            # del features2, box_features2, predictions2, pred_delta2, pred_logit2

            box_in_fea2 = ['p5', 'p2', 'p3', 'p4']
            features2 = [features2[f] for f in box_in_fea2]
            box_features2 = self.box_pooler(features2, [x.proposal_boxes for x in proposals])
            box_features2 = self.box_head(box_features2)
            # box_features2 = torch.cat((box_features2, big_feature), 1)
            # box_features2 = self.cross_layer(box_features2)
            # print('box_features2: ', box_features2.size())
            predictions2 = self.box_predictor(box_features2)
            pred_logit1, pred_delta1, _ = predictions
            pred_logit2, pred_delta2, _ = predictions2
            # print('pred_logit1:', box_features.size(), 'pred_logit2:', box_features2.size())
            proposal2 = torch.cat([p.gt_classes for p in proposals], dim=0)
            klloss = F.cross_entropy(pred_logit2[int(len(pred_logit1)/4):, :], proposal2[int(len(pred_logit1)/4):], reduction='mean')
            # p = torch.exp(-klloss)
            # k_loss = (1 - p) ** 1 * klloss
            # losses['loss_twoKL'] = k_loss.sum() / proposal2[int(len(pred_logit1)/4):].size(0) * 0.2
            losses['loss_twoKL'] = klloss * 0.5
            
            box_type = type(proposals[0].proposal_boxes)
            gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            proposal_boxes = box_type.cat([p.proposal_boxes for p in proposals])
            box_transform = Box2BoxTransform((10.0, 10.0, 5.0, 5.0))
            gt_pred_deltas = box_transform.get_deltas(proposal_boxes.tensor, gt_boxes.tensor)
            pred_delta2 = pred_delta2[int(len(pred_delta2)/4):, :]
            gt_pred_deltas = gt_pred_deltas[int(len(gt_pred_deltas)/4):, :]
            proposal2 = proposal2[int(len(pred_logit1)/4):]
            # lscore = lscore[int(len(pred_logit1)/4):]
            fg_inds = nonzero_tuple((proposal2 >= 0) & (proposal2 < 20))[0]
            fg_gt_classes = proposal2[fg_inds]
            gt_class_cols = 4 * fg_gt_classes[:, None] + torch.arange(4, device=pred_logit2.device)
            loss_box_reg = smooth_l1_loss(
                pred_delta2[fg_inds[:, None], gt_class_cols],
                gt_pred_deltas[fg_inds],
                0.0,
                reduction="sum",
            )
            losses['loss_twoKL_reg'] = loss_box_reg * 0.5 / max(proposal2.numel(), 1.0)
            del features2, box_features2, predictions2, pred_delta2, pred_logit2

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            # print('class:', targets_per_image.gt_classes)
            # print('score:', targets_per_image.gt_scores)
            if targets_per_image.gt_scores.numel() > 0:
                gt_scores = targets_per_image.gt_scores[matched_idxs]
            else:
                gt_scores = torch.zeros_like(matched_idxs) + self.num_classes
            gt_scores = gt_scores[sampled_idxs]
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_scores = gt_scores
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
