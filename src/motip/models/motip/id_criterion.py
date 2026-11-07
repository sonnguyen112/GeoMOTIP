# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import is_distributed, distributed_world_size, labels_to_one_hot
from models.motip.spatial_priors import spatial_id_loss_components


class IDCriterion(nn.Module):
    def __init__(
            self,
            weight: float,
            use_focal_loss: bool,
            use_spatial_id_loss: bool = False,
            spatial_giou_weight: float = 2.0,
            spatial_l1_weight: float = 5.0,
    ):
        super().__init__()
        self.weight = weight
        self.use_focal_loss = use_focal_loss
        self.use_spatial_id_loss = bool(use_spatial_id_loss)
        self.spatial_giou_weight = float(spatial_giou_weight)
        self.spatial_l1_weight = float(spatial_l1_weight)
        self.last_spatial_l1_loss = 0.0
        self.last_spatial_giou_loss = 0.0
        if self.spatial_giou_weight < 0 or self.spatial_l1_weight < 0:
            raise ValueError("Spatial ID loss weights must be non-negative")
        if self.use_spatial_id_loss and self.use_focal_loss:
            raise ValueError("MOTIP2-style spatial ID loss is defined with softmax cross-entropy")

        if not self.use_focal_loss:
            self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        return

    def forward(
            self,
            id_logits,
            id_labels,
            id_masks,
            predicted_boxes=None,
            gt_boxes=None,
    ):
        spatial_logits_input = id_logits
        spatial_labels_input = id_labels
        spatial_masks_input = id_masks
        # _B, _G, _T, _N = id_logits.shape
        # Remove the first T for supervision:
        id_logits = id_logits[:, :, 1:, :, :]
        id_labels = id_labels[:, :, 1:, :]
        id_masks = id_masks[:, :, 1:, :]
        pass

        # Flatten:
        id_logits_flatten = einops.rearrange(id_logits, "b g t n c -> (b g t n) c")
        id_labels_flatten = einops.rearrange(id_labels, "b g t n -> (b g t n)")
        id_masks_flatten = einops.rearrange(id_masks, "b g t n -> (b g t n)")
        # Filter out the invalid id labels:
        id_logits_flatten = id_logits_flatten[~id_masks_flatten]
        id_labels_flatten = id_labels_flatten[~id_masks_flatten]
        # Calculate the loss:
        if self.use_focal_loss:
            id_labels_flatten_one_hot = labels_to_one_hot(id_labels_flatten, class_num=id_logits_flatten.shape[-1])
            id_labels_flatten_one_hot = torch.tensor(id_labels_flatten_one_hot, device=id_logits.device)
            loss = sigmoid_focal_loss(inputs=id_logits_flatten, targets=id_labels_flatten_one_hot).sum()
        else:
            loss = self.ce_loss(id_logits_flatten, id_labels_flatten).sum()
        num_ids = torch.as_tensor([len(id_logits_flatten)], dtype=torch.float, device=id_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_ids)
        num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()
        loss = loss / num_ids

        self.last_spatial_l1_loss = 0.0
        self.last_spatial_giou_loss = 0.0
        if self.use_spatial_id_loss:
            if predicted_boxes is None or gt_boxes is None:
                raise ValueError("Spatial ID loss requires predicted_boxes and gt_boxes")
            spatial_l1, spatial_giou, spatial_count = spatial_id_loss_components(
                id_logits=spatial_logits_input,
                id_labels=spatial_labels_input,
                id_masks=spatial_masks_input,
                predicted_boxes=predicted_boxes,
                gt_boxes=gt_boxes,
            )
            spatial_count = spatial_count.reshape(1).to(dtype=torch.float, device=id_logits.device)
            if is_distributed():
                torch.distributed.all_reduce(spatial_count)
            spatial_normalizer = torch.clamp(
                spatial_count / distributed_world_size(), min=1
            ).item()
            spatial_l1 = spatial_l1 / spatial_normalizer
            spatial_giou = spatial_giou / spatial_normalizer
            loss = loss + self.spatial_l1_weight * spatial_l1 + self.spatial_giou_weight * spatial_giou
            self.last_spatial_l1_loss = float(spatial_l1.detach().item())
            self.last_spatial_giou_loss = float(spatial_giou.detach().item())

        return loss


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        use_focal_loss=config["USE_FOCAL_LOSS"],
        use_spatial_id_loss=config.get("USE_SPATIAL_ID_LOSS", False),
        spatial_giou_weight=config.get("SPATIAL_ID_GIOU_WEIGHT", 2.0),
        spatial_l1_weight=config.get("SPATIAL_ID_L1_WEIGHT", 5.0),
    )
