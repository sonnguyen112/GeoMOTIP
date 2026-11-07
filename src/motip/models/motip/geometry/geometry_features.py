"""Shared G4 multi-lag geometry feature construction.

The offline probe and the online V3 branch intentionally use this module.  A
history sample is selected only when its stored timestamp equals the requested
current timestamp minus the requested lag; missing samples are zero-filled and
represented by an explicit validity bit.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


DEFAULT_GEOMETRY_LAGS = (1, 2, 4, 8)
GEOMETRY_FEATURE_DIM = 66
EPS = 1e-5


def _as_lags(lags: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(x) for x in lags)
    if values != DEFAULT_GEOMETRY_LAGS:
        raise ValueError(
            "The V3 G4 descriptor is defined for lags (1, 2, 4, 8); "
            f"received {values}."
        )
    return values


def _validate_pair_inputs(
    current: torch.Tensor,
    lag_boxes: torch.Tensor,
    lag_valid: torch.Tensor,
) -> None:
    if current.shape[-1] != 4:
        raise ValueError(f"current boxes must end in 4, got {tuple(current.shape)}")
    if lag_boxes.shape[-2:] != (len(DEFAULT_GEOMETRY_LAGS), 4):
        raise ValueError(
            "lag_boxes must have shape [..., 4, 4], "
            f"got {tuple(lag_boxes.shape)}"
        )
    if lag_valid.shape[-1] != len(DEFAULT_GEOMETRY_LAGS):
        raise ValueError(
            "lag_valid must have shape [..., 4], "
            f"got {tuple(lag_valid.shape)}"
        )
    if lag_boxes.shape[:-1] != lag_valid.shape:
        raise ValueError(
            "lag_boxes and lag_valid are incompatible: "
            f"boxes={tuple(lag_boxes.shape)}, valid={tuple(lag_valid.shape)}"
        )
    if current.shape[:-1] != lag_boxes.shape[:-2]:
        raise ValueError(
            "current and lag boxes must share pair dimensions: "
            f"current={tuple(current.shape)}, lag_boxes={tuple(lag_boxes.shape)}"
        )


def build_g4_features(
    current: torch.Tensor,
    lag_boxes: torch.Tensor,
    lag_valid: torch.Tensor,
    lags: Sequence[int] = DEFAULT_GEOMETRY_LAGS,
) -> torch.Tensor:
    """Build the exact 66-dimensional offline G4 descriptor.

    Args:
        current: ``[..., 4]`` normalized ``cx, cy, w, h`` boxes.
        lag_boxes: ``[..., 4, 4]`` boxes at exact lags ``1,2,4,8``.
        lag_valid: ``[..., 4]`` exact-observation validity bits.

    Returns:
        ``[..., 66]`` geometry features.  Invalid lag blocks are zeroed, while
        validity channels remain one-bit indicators.
    """
    _as_lags(lags)
    _validate_pair_inputs(current, lag_boxes, lag_valid)
    valid = lag_valid.to(dtype=torch.bool)
    current_scale = current[..., 2:].clamp_min(EPS)
    centers = lag_boxes[..., :, :2]

    raw = torch.where(valid[..., None], lag_boxes, torch.zeros_like(lag_boxes))
    parts = [current, raw.flatten(-2), valid.to(current.dtype)]

    # Relative displacement and log-scale change for all four lags.
    relative_center = current[..., None, :2] / current_scale[..., None, :] - centers / current_scale[..., None, :]
    relative_scale = torch.log(
        (current[..., None, 2:] + EPS) / (lag_boxes[..., :, 2:] + EPS)
    )
    relative = torch.cat((relative_center, relative_scale), dim=-1)
    relative = torch.where(valid[..., None], relative, torch.zeros_like(relative))

    # Adjacent historical velocities, normalized by current object scale.
    velocities = []
    velocity_valid = []
    for left, right, gap in ((0, 1, 1.0), (1, 2, 2.0), (2, 3, 4.0)):
        value = (centers[..., left, :] - centers[..., right, :]) / current_scale / gap
        is_valid = valid[..., left] & valid[..., right]
        velocities.append(torch.where(is_valid[..., None], value, torch.zeros_like(value)))
        velocity_valid.append(is_valid)
    recent = velocities[0]
    extrapolation = (
        current[..., :2] - (centers[..., 0, :] + recent * current_scale)
    ) / current_scale
    extrapolation_valid = valid[..., 0] & valid[..., 1]
    extrapolation = torch.where(
        extrapolation_valid[..., None], extrapolation, torch.zeros_like(extrapolation)
    )
    velocity_values = torch.stack((*velocities, extrapolation), dim=-2).flatten(-2)
    velocity_valid_values = torch.stack((*velocity_valid, extrapolation_valid), dim=-1)
    parts += [velocity_values, velocity_valid_values.to(current.dtype)]

    # Compact acceleration: differences of adjacent historical velocities.
    accelerations = []
    acceleration_valid = []
    for index in (0, 1):
        value = velocities[index] - velocities[index + 1]
        is_valid = velocity_valid[index] & velocity_valid[index + 1]
        accelerations.append(torch.where(is_valid[..., None], value, torch.zeros_like(value)))
        acceleration_valid.append(is_valid)
    parts += [
        torch.cat(accelerations, dim=-1),
        torch.stack(acceleration_valid, dim=-1).to(current.dtype),
        relative.flatten(-2),
    ]

    area_change = relative[..., 2] + relative[..., 3]
    area_change = torch.where(valid, area_change, torch.zeros_like(area_change))
    lag_code = current.new_tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    lag_code = lag_code.expand(current.shape[:-1] + (len(DEFAULT_GEOMETRY_LAGS),))
    parts += [area_change, lag_code]
    features = torch.cat(parts, dim=-1)
    if features.shape[-1] != GEOMETRY_FEATURE_DIM:
        raise RuntimeError(f"G4 feature dimension drifted to {features.shape[-1]}")
    return features


def build_pairwise_g4_features(
    current_boxes: torch.Tensor,
    current_times: torch.Tensor,
    history_boxes: torch.Tensor,
    history_times: torch.Tensor,
    history_masks: torch.Tensor,
    lags: Sequence[int] = DEFAULT_GEOMETRY_LAGS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct pairwise G4 features from an exact timestamped history.

    Args:
        current_boxes: ``[B,G,Tq,Q,4]`` current detection boxes.
        current_times: ``[B,G,Tq,Q]`` current timestamps.
        history_boxes: ``[B,G,Th,K,4]`` candidate-track history boxes.
        history_times: ``[B,G,Th,K]`` timestamps for the history boxes.
        history_masks: ``[B,G,Th,K]`` true for missing/padded observations.

    Returns:
        ``features`` with shape ``[B,G,Tq,Q,K,66]`` and an exact-lag validity
        tensor ``[B,G,Tq,Q,K,4]``.  No timestamp at or after the current frame
        can match a requested positive lag.
    """
    lags = _as_lags(lags)
    if current_boxes.shape[-1] != 4 or history_boxes.shape[-1] != 4:
        raise ValueError("All boxes must be normalized cxcywh with final dimension 4")
    if history_boxes.shape[:-1] != history_masks.shape:
        raise ValueError("history_boxes/history_masks shape mismatch")
    if history_times.shape != history_masks.shape:
        raise ValueError("history_times/history_masks shape mismatch")
    if current_boxes.shape[:-1] != current_times.shape:
        raise ValueError("current_boxes/current_times shape mismatch")

    # [B,G,Tq,Q,Th,K], selecting each candidate column independently.
    current_time = current_times[..., None, None]
    stored_time = history_times[:, :, None, None, :, :]
    stored_valid = ~history_masks[:, :, None, None, :, :]
    lag_features = []
    lag_validities = []
    for lag in lags:
        exact = stored_valid & (stored_time == current_time - int(lag))
        boxes = history_boxes[:, :, None, None, :, :, :]
        selected = (boxes * exact[..., None].to(boxes.dtype)).sum(dim=-3)
        is_valid = exact.any(dim=-2)
        lag_features.append(selected)
        lag_validities.append(is_valid)
    lag_boxes = torch.stack(lag_features, dim=-2)
    lag_valid = torch.stack(lag_validities, dim=-1)
    current = current_boxes[..., None, :].expand_as(lag_boxes[..., 0, :])
    features = build_g4_features(current, lag_boxes, lag_valid, lags=lags)
    return features, lag_valid
