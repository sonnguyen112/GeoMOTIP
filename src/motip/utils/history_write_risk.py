"""Shared offline/runtime features for ranking unreliable trajectory writes.

All geometric inputs are XYWH boxes in one common coordinate scale. Runtime
boxes are converted from normalized CXCYWH; offline audit boxes are pixel
XYWH. The descriptors are invariant to a shared image-scale factor.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch


LAGS = (1, 2, 4, 8)
EPS = 1e-6
RISK_FEATURE_NAMES = [
    "age", "time_since_seen", "valid_lag_count",
    *[f"lag{lag}_{suffix}" for lag in LAGS
      for suffix in ("valid", "dx", "dy", "logw", "logh", "iou")],
]
ALL_FEATURE_NAMES = [*RISK_FEATURE_NAMES, "velocity_x", "velocity_y",
                     "extrap_residual_x", "extrap_residual_y"]


def _as_list(value) -> list[float]:
    if torch.is_tensor(value):
        return [float(v) for v in value.detach().to(device="cpu").reshape(-1).tolist()]
    return [float(v) for v in value]


def cxcywh_to_xywh(box) -> list[float]:
    cx, cy, width, height = _as_list(box)
    return [cx - width / 2, cy - height / 2, width, height]


def box_iou_xywh(a, b) -> float:
    ax, ay, aw, ah = _as_list(a)
    bx, by, bw, bh = _as_list(b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
    return inter / max(union, EPS)


def build_history_write_features(
        current_xywh,
        history_by_lag: Mapping[int, object | None],
        age: int,
        time_since_seen: int,
) -> dict[str, float]:
    """Build the exact fixed feature vector used in the sequence-risk probe."""
    current = _as_list(current_xywh)
    if len(current) != 4 or current[2] <= 0 or current[3] <= 0:
        raise ValueError("current_xywh must be a valid positive-size XYWH box")
    current_center = [current[0] + current[2] / 2, current[1] + current[3] / 2]
    centers = {0: current_center}
    lag_boxes: dict[int, list[float] | None] = {}
    features: dict[str, float] = {
        "age": float(age),
        "time_since_seen": float(time_since_seen),
    }
    valid_count = 0
    for lag in LAGS:
        raw = history_by_lag.get(lag)
        hist = _as_list(raw) if raw is not None else None
        if hist is not None and (len(hist) != 4 or hist[2] <= 0 or hist[3] <= 0):
            hist = None
        lag_boxes[lag] = hist
        valid = hist is not None
        valid_count += int(valid)
        features[f"lag{lag}_valid"] = float(valid)
        if valid:
            hist_center = [hist[0] + hist[2] / 2, hist[1] + hist[3] / 2]
            centers[lag] = hist_center
            dx = (current_center[0] - hist_center[0]) / max(hist[2], EPS)
            dy = (current_center[1] - hist_center[1]) / max(hist[3], EPS)
            logw = math.log(max(current[2], EPS) / max(hist[2], EPS))
            logh = math.log(max(current[3], EPS) / max(hist[3], EPS))
            overlap = box_iou_xywh(current, hist)
        else:
            dx = dy = logw = logh = overlap = 0.0
        features.update({f"lag{lag}_dx": dx, f"lag{lag}_dy": dy,
                         f"lag{lag}_logw": logw, f"lag{lag}_logh": logh,
                         f"lag{lag}_iou": overlap})
    features["valid_lag_count"] = float(valid_count)

    if 1 in centers and 2 in centers:
        hist1 = lag_boxes[1]
        scale_w, scale_h = max(hist1[2], EPS), max(hist1[3], EPS)
        velocity = [centers[1][0] - centers[2][0], centers[1][1] - centers[2][1]]
        residual = [current_center[0] - (centers[1][0] + velocity[0]),
                    current_center[1] - (centers[1][1] + velocity[1])]
        features["velocity_x"] = velocity[0] / scale_w
        features["velocity_y"] = velocity[1] / scale_h
        features["extrap_residual_x"] = residual[0] / scale_w
        features["extrap_residual_y"] = residual[1] / scale_h
    else:
        features.update(velocity_x=0.0, velocity_y=0.0,
                        extrap_residual_x=0.0, extrap_residual_y=0.0)
    return features


def snapshot_runtime_history(tracker) -> dict[str, torch.Tensor]:
    """Copy the pre-write track history once for one policy invocation."""
    return {
        "labels": tracker.trajectory_id_labels.detach().to(device="cpu"),
        "boxes": tracker.trajectory_boxes.detach().to(device="cpu"),
        "times": tracker.trajectory_frame_times.detach().to(device="cpu"),
        "masks": tracker.trajectory_masks.detach().to(device="cpu"),
    }


def runtime_history_write_features(snapshot: dict[str, torch.Tensor], frame: int,
                                   current_cxcywh, slot: int, age: int) -> dict[str, float]:
    """Extract exact-lag features from RuntimeTracker's pre-update state.

    Missing lags stay missing; no nearest-frame substitution is performed.
    A future/equal-frame observation is rejected to prevent leakage.
    """
    labels, boxes = snapshot["labels"], snapshot["boxes"]
    times, masks = snapshot["times"], snapshot["masks"]
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        raise KeyError(f"No existing trajectory history for local slot {slot}")
    columns = (labels[0] == int(slot)).nonzero(as_tuple=False).flatten()
    if columns.numel() == 0:
        raise KeyError(f"No existing trajectory history for local slot {slot}")
    if columns.numel() != 1:
        raise KeyError(f"Expected exactly one active history column for slot {slot}, got {columns.numel()}")
    column = int(columns.item())
    valid_rows = (~masks[:, column]).nonzero(as_tuple=False).flatten()
    history_at_time: dict[int, list[float]] = {}
    for row_idx in valid_rows.tolist():
        stamp = int(times[row_idx, column].item())
        if stamp >= int(frame):
            raise AssertionError(f"Future/current history timestamp {stamp} at frame {frame}")
        if stamp in history_at_time:
            raise AssertionError(f"Duplicate committed box for slot {slot} at frame {stamp}")
        history_at_time[stamp] = cxcywh_to_xywh(boxes[row_idx, column])
    last_seen = max(history_at_time) if history_at_time else None
    time_since_seen = int(frame) - last_seen if last_seen is not None else int(frame)
    history_by_lag = {lag: history_at_time.get(int(frame) - lag) for lag in LAGS}
    return build_history_write_features(
        cxcywh_to_xywh(current_cxcywh), history_by_lag, age, time_since_seen,
    )


class LogisticHistoryCommitPolicy:
    """Use a frozen logistic risk model to abstain from existing-track writes.

    ``payload`` is one of the JSON records emitted by the reference fitting
    tool; the threshold must be selected outside the locked evaluation group.
    A new instance should be passed to each sequence (RuntimeTracker resets
    it via ``reset_sequence``).
    """

    def __init__(self, payload: dict, threshold: float):
        self.feature_names = list(payload["features"])
        if self.feature_names != RISK_FEATURE_NAMES:
            raise ValueError("History risk model feature order does not match runtime feature builder")
        self.mean = [float(x) for x in payload["standard_scaler_mean"]]
        self.scale = [max(float(x), EPS) for x in payload["standard_scaler_scale"]]
        self.coef = [float(x) for x in payload["logistic_coefficients"]]
        self.intercept = float(payload["logistic_intercept"])
        if not (len(self.mean) == len(self.scale) == len(self.coef) == len(self.feature_names)):
            raise ValueError("Invalid history risk model parameter dimensions")
        if not math.isfinite(float(threshold)):
            raise ValueError("Risk threshold must be finite")
        self.threshold = float(threshold)
        self.reset_sequence()

    def reset_sequence(self) -> None:
        self.birth_frame_by_global_id: dict[int, int] = {}

    def _probability(self, features: dict[str, float]) -> float:
        z = self.intercept
        for name, mean, scale, coef in zip(self.feature_names, self.mean, self.scale, self.coef):
            z += ((float(features[name]) - mean) / scale) * coef
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def __call__(self, *, tracker, boxes, id_labels, is_newborn, frame_index, **_unused):
        keep = torch.ones_like(id_labels, dtype=torch.bool)
        snapshot = snapshot_runtime_history(tracker)
        slots = id_labels.detach().to(device="cpu").tolist()
        newborns = is_newborn.detach().to(device="cpu").bool().tolist()
        box_rows = boxes.detach().to(device="cpu")
        for index, (slot, newborn) in enumerate(zip(slots, newborns)):
            global_id = tracker.id_label_to_id.get(int(slot))
            if global_id is None:
                raise KeyError(f"No global ID mapping for local slot {slot}")
            birth = self.birth_frame_by_global_id.setdefault(int(global_id), int(frame_index))
            if newborn:
                continue
            age = int(frame_index) - birth + 1
            try:
                features = runtime_history_write_features(
                    snapshot, int(frame_index), box_rows[index], int(slot), age,
                )
            except KeyError as error:
                # A slot can be newly allocated/recycled by MOTIP while the
                # current prediction is not marked NEW by the decoder. With
                # no prior history there is nothing to contaminate; retain
                # this first write so the slot can establish its trajectory.
                if error.args and str(error.args[0]).startswith("No existing trajectory history"):
                    continue
                raise
            if self._probability(features) >= self.threshold:
                keep[index] = False
        return keep
