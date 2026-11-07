"""Lightweight diagnostic helpers for cached appearance compatibility."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def candidate_appearance_cosines(
        detection_features: torch.Tensor,
        trajectory_features: torch.Tensor,
        trajectory_id_labels: torch.Tensor,
        trajectory_masks: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_active: torch.Tensor,
):
    """Compute optional appearance similarities for active track candidates.

    Returns detection-by-candidate cosine matrices against each candidate
    track's latest observed, mean observed, and maximum historical similarity.
    Missing histories remain NaN and are explicitly marked invalid. This is a
    diagnostic-only calculation and never contributes to association logits.
    """
    candidate_ids = candidate_ids.detach().cpu().long()
    candidate_active = candidate_active.detach().cpu().bool()
    query = F.normalize(detection_features.detach().float(), dim=-1).cpu()
    n, k = candidate_ids.shape
    scores = {
        name: torch.full((n, k), float("nan"), dtype=torch.float32)
        for name in ("last", "mean", "max")
    }
    valid = torch.zeros((n, k), dtype=torch.bool)
    if trajectory_features.numel() == 0 or trajectory_id_labels.numel() == 0:
        return scores, valid

    history = trajectory_features.detach().float().cpu()
    labels = trajectory_id_labels.detach().cpu().long()
    masks = trajectory_masks.detach().cpu().bool()
    latest_labels = labels[-1]
    for column, slot_value in enumerate(latest_labels.tolist()):
        slot = int(slot_value)
        selected = candidate_active & (candidate_ids == slot)
        if not selected.any():
            continue
        observed = (~masks[:, column]).nonzero(as_tuple=False).flatten()
        if not observed.numel():
            continue
        features = F.normalize(history[observed, column], dim=-1)
        cosine = query @ features.T
        rows, cols = torch.where(selected)
        scores["last"][rows, cols] = cosine[rows, -1]
        scores["mean"][rows, cols] = cosine[rows].mean(dim=-1)
        scores["max"][rows, cols] = cosine[rows].max(dim=-1).values
        valid[selected] = True
    return scores, valid
