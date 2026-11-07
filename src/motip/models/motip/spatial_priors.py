"""Spatial-prior components adapted from the MOTIP2 paper.

These components implement the paper's spatially weighted ID objective and a
learned box-to-token position cue for the existing Deformable-DETR/MOTIP
backbone. The latter is an adaptation: this DETR implementation does not
expose MOTIP2's detector-produced final-box positional embedding.
"""

from __future__ import annotations

import torch
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy


class LearnedBoxPositionEmbedding(nn.Module):
    """Map a normalized final box into an additive feature-space position cue."""

    def __init__(self, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(hidden_dim or 2 * feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, boxes: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if boxes.shape[-1] != 4 or boxes.shape[:-1] != masks.shape:
            raise ValueError("Position embedding expects boxes [...,4] and masks [...]")
        # Do not detach: the paper reuses a learned detector-side position cue,
        # so this adaptation allows association loss to train the box pathway.
        safe_boxes = boxes.masked_fill(masks.unsqueeze(-1), 0)
        embedded = self.mlp(safe_boxes)
        return embedded.masked_fill(masks.unsqueeze(-1), 0)


def _pairwise_giou_aligned(boxes1_cxcywh: torch.Tensor, boxes2_cxcywh: torch.Tensor) -> torch.Tensor:
    """Elementwise GIoU for aligned pairs of cxcywh boxes."""
    boxes1 = box_cxcywh_to_xyxy(boxes1_cxcywh.float())
    boxes2 = box_cxcywh_to_xyxy(boxes2_cxcywh.float())
    lt_inter = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb_inter = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh_inter = (rb_inter - lt_inter).clamp(min=0)
    intersection = wh_inter[..., 0] * wh_inter[..., 1]

    wh1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0)
    wh2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0)
    area1 = wh1[..., 0] * wh1[..., 1]
    area2 = wh2[..., 0] * wh2[..., 1]
    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=1e-7)

    lt_enclosure = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    rb_enclosure = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    wh_enclosure = (rb_enclosure - lt_enclosure).clamp(min=0)
    enclosure = wh_enclosure[..., 0] * wh_enclosure[..., 1]
    return iou - (enclosure - union) / enclosure.clamp(min=1e-7)


def spatial_id_loss_components(
    id_logits: torch.Tensor,
    id_labels: torch.Tensor,
    id_masks: torch.Tensor,
    predicted_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute MOTIP2-style expected spatial costs over present IDs + NEW.

    Inputs are one ID-decoder output stack, before the normal MOTIP criterion
    removes the first context frame. ``id_logits`` and ``id_labels`` may contain
    concatenated auxiliary decoder layers along dimension 0. Box tensors remain
    one copy per batch item and are repeated to match that convention.

    The candidate box for an existing local ID is its current-frame GT box.
    For NEW, the candidate box is the current detection's own GT box, matching
    the paper's stated convention. Probabilities are renormalized across only
    IDs present in the current frame plus NEW. Predicted boxes are detached so
    these terms do not train the detector regression head.

    Returns summed L1 cost, summed GIoU cost, and the count of valid queries.
    """
    if id_logits.ndim != 5 or id_labels.ndim != 4 or id_masks.shape != id_labels.shape:
        raise ValueError("Expected logits [B,G,T,N,C], labels/masks [B,G,T,N]")
    if id_logits.shape[:-1] != id_labels.shape:
        raise ValueError("ID logits and labels are not aligned")

    if predicted_boxes.shape[-1] != 4 or gt_boxes.shape != predicted_boxes.shape:
        raise ValueError("Predicted/GT boxes must have aligned [...,4] shapes")
    if predicted_boxes.shape != (*id_labels.shape, 4):
        if predicted_boxes.shape[0] == 0 or id_labels.shape[0] % predicted_boxes.shape[0] != 0:
            raise ValueError("Box batches cannot be expanded to match auxiliary ID logits")
        repeats = id_labels.shape[0] // predicted_boxes.shape[0]
        predicted_boxes = predicted_boxes.repeat(repeats, 1, 1, 1, 1)
        gt_boxes = gt_boxes.repeat(repeats, 1, 1, 1, 1)
        if predicted_boxes.shape != (*id_labels.shape, 4):
            raise ValueError("Predicted/GT boxes do not align with ID logits after layer expansion")

    class_count = id_logits.shape[-1]
    new_class = class_count - 1
    if new_class < 1:
        raise ValueError("ID vocabulary must contain existing IDs and a NEW class")

    # MOTIP's ID criterion intentionally excludes the first frame in the clip.
    logits = id_logits[:, :, 1:, :, :].float()
    labels = id_labels[:, :, 1:, :].long()
    masks = id_masks[:, :, 1:, :].bool()
    pred = predicted_boxes[:, :, 1:, :, :].detach().float()
    gt = gt_boxes[:, :, 1:, :, :].float()
    if logits.shape[2] == 0:
        zero = id_logits.sum() * 0
        return zero, zero, zero.detach()

    # Present ID labels + the always-available newborn class form the spatial
    # loss candidate set. Accumulation handles repeated labels safely.
    valid = ~masks
    if torch.any(valid & ((labels < 0) | (labels >= class_count))):
        raise ValueError("A valid ID label lies outside the ID vocabulary")
    safe_labels = labels.clamp(min=0, max=new_class)
    present = torch.zeros(
        (*labels.shape[:3], class_count), dtype=torch.long, device=labels.device
    )
    present.scatter_add_(-1, safe_labels, valid.long())
    present = present > 0
    present[..., new_class] = True

    # Build one GT box per existing local ID. Padded/NEW queries are directed
    # to the NEW slot, then that slot is replaced per query below.
    is_existing = valid & (labels >= 0) & (labels < new_class)
    box_indices = torch.where(is_existing, labels, torch.full_like(labels, new_class))
    box_sources = torch.where(is_existing.unsqueeze(-1), gt, torch.zeros_like(gt))
    boxes_by_id = gt.new_zeros((*gt.shape[:-2], class_count, 4))
    boxes_by_id.scatter_(-2, box_indices.unsqueeze(-1).expand(*box_indices.shape, 4), box_sources)

    candidate_boxes = boxes_by_id.unsqueeze(-3).expand(*gt.shape[:-1], class_count, 4).clone()
    candidate_boxes[..., new_class, :] = gt

    allowed = present.unsqueeze(-2).expand(*labels.shape, class_count)
    logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=-1)

    pred_expanded = pred.unsqueeze(-2).expand_as(candidate_boxes)
    l1_cost = (pred_expanded - candidate_boxes).abs().sum(dim=-1)
    giou_cost = 1.0 - _pairwise_giou_aligned(pred_expanded, candidate_boxes)
    l1_per_query = (probabilities * l1_cost).sum(dim=-1)
    giou_per_query = (probabilities * giou_cost).sum(dim=-1)

    l1_sum = l1_per_query[valid].sum()
    giou_sum = giou_per_query[valid].sum()
    count = valid.sum().to(dtype=logits.dtype)
    return l1_sum, giou_sum, count
