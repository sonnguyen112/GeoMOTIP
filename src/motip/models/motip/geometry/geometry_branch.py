"""Candidate-wise geometry compatibility branch for Trail-MOTIP V3."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .geometry_features import (
    DEFAULT_GEOMETRY_LAGS,
    GEOMETRY_FEATURE_DIM,
    build_pairwise_g4_features,
)


class GeometryCompatibilityBranch(nn.Module):
    """Small MLP that scores a current detection against each active track."""

    def __init__(
        self,
        feature_dim: int = GEOMETRY_FEATURE_DIM,
        hidden_dim: int = 128,
        num_id_vocabulary: int = 50,
        lags: tuple[int, ...] = DEFAULT_GEOMETRY_LAGS,
        fixed_lambda: float = 0.01,
        aux_weight: float = 1.0,
        fused_weight: float = 1.0,
        validate_alignment: bool = True,
        input_clip: float = 50.0,
        input_normalize: bool = True,
        input_stats: torch.Tensor | None = None,
        store_candidate_metadata: bool = False,
    ):
        super().__init__()
        if feature_dim != GEOMETRY_FEATURE_DIM:
            raise ValueError("V3 currently uses the fixed 66-dimensional G4 descriptor")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_id_vocabulary = int(num_id_vocabulary)
        self.lags = tuple(int(x) for x in lags)
        self.fixed_lambda = float(fixed_lambda)
        self.aux_weight = float(aux_weight)
        self.fused_weight = float(fused_weight)
        self.validate_alignment = bool(validate_alignment)
        self.input_clip = float(input_clip)
        self.store_candidate_metadata = bool(store_candidate_metadata)
        self.input_normalize = bool(input_normalize)
        if input_stats is not None:
            if tuple(input_stats.shape) != (2, self.feature_dim):
                raise ValueError(
                    "geometry input stats must have shape [2, feature_dim], "
                    f"got {tuple(input_stats.shape)}"
                )
            self.register_buffer("input_mean", input_stats[0].detach().float().clone())
            self.register_buffer(
                "input_std", input_stats[1].detach().float().clamp_min(1e-3).clone()
            )
        else:
            self.register_buffer("input_mean", torch.empty(0))
            self.register_buffer("input_std", torch.empty(0))
        self.input_norm = nn.LayerNorm(self.feature_dim, elementwise_affine=False) if self.input_normalize else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.last_pair_scores = None
        self.last_pair_valid = None
        self.last_candidate_ids = None
        self.last_candidate_active = None
        self.last_exact_lag_valid = None
        self.last_geometry_logits = None
        self.last_geometry_loss = None
        self.last_stats = {}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def score_pairs(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        scores = self.mlp(features).squeeze(-1)
        return scores.masked_fill(~valid, 0.0)

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply offline-probe feature normalization, or safe legacy fallback."""
        if self.input_mean.numel():
            mean = self.input_mean.to(device=features.device, dtype=features.dtype)
            std = self.input_std.to(device=features.device, dtype=features.dtype)
            return ((features - mean) / std).clamp(-self.input_clip, self.input_clip)
        if self.input_clip > 0:
            features = features.clamp(-self.input_clip, self.input_clip)
        return self.input_norm(features)

    @staticmethod
    def _candidate_ids(
        history_labels: torch.Tensor,
        history_times: torch.Tensor,
        history_masks: torch.Tensor,
        current_times: torch.Tensor,
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return each candidate's last valid pre-current local ID."""
        prior = (~history_masks[:, :, None, None, :, :]) & (
            history_times[:, :, None, None, :, :] < current_times[..., None, None]
        )
        score_times = history_times[:, :, None, None, :, :].masked_fill(~prior, -1)
        index = score_times.argmax(dim=-2)
        candidate_ids = history_labels[:, :, None, None, :, :].expand_as(score_times)
        candidate_ids = candidate_ids.gather(-2, index.unsqueeze(-2)).squeeze(-2)
        active = prior.any(dim=-2)
        candidate_ids = torch.where(
            active & (candidate_ids >= 0) & (candidate_ids < num_classes),
            candidate_ids,
            torch.full_like(candidate_ids, num_classes),
        )
        return candidate_ids, active

    def _scatter_logits(
        self,
        pair_scores: torch.Tensor,
        pair_valid: torch.Tensor,
        candidate_ids: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        logits = pair_scores.new_zeros((*pair_scores.shape[:-1], num_classes + 1))
        safe_ids = candidate_ids.clamp(min=0, max=num_classes)
        values = pair_scores * pair_valid.to(pair_scores.dtype)
        if self.validate_alignment:
            active_ids = safe_ids.masked_fill(~pair_valid, num_classes + 1)
            ordered = active_ids.sort(dim=-1).values
            duplicate = (ordered[..., 1:] == ordered[..., :-1]) & (
                ordered[..., 1:] < num_classes
            )
            if duplicate.any():
                raise AssertionError("Active geometry candidates contain duplicate local ID slots")
        logits.scatter_add_(-1, safe_ids, values)
        # Geometry never assigns a score to the newborn/special class.
        logits[..., num_classes] = 0.0
        return logits

    def forward(self, seq: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # MOTIP keeps ``trajectory_times``/``unknown_times`` as local causal
        # positions for its relative positional encoding.  Geometry, however,
        # needs the actual frame numbers: sampled clips can have a temporal
        # stride and an observation at local index ``t-1`` is not necessarily
        # one frame before the query.  Training/runtime provide the explicit
        # frame-time tensors; retain the local-time fallback for old callers.
        current_times = seq.get("unknown_frame_times", seq["unknown_times"])
        history_times = seq.get("trajectory_frame_times", seq["trajectory_times"])
        features, lag_valid = build_pairwise_g4_features(
            current_boxes=seq["unknown_boxes"],
            current_times=current_times,
            history_boxes=seq["trajectory_boxes"],
            history_times=history_times,
            history_masks=seq["trajectory_masks"],
            lags=self.lags,
        )
        if not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite G4 geometry feature encountered")
        raw_feature_abs_max = float(features.detach().float().abs().max().item())
        features = self.normalize_features(features)
        feature_abs_max = float(features.detach().float().abs().max().item())
        candidate_ids, active = self._candidate_ids(
            history_labels=seq["trajectory_id_labels"],
            history_times=history_times,
            history_masks=seq["trajectory_masks"],
            current_times=current_times,
            num_classes=self.num_id_vocabulary,
        )
        pair_valid = lag_valid.any(dim=-1) & active & ~seq["unknown_masks"][..., None]
        pair_scores = self.score_pairs(features, pair_valid)
        logits = self._scatter_logits(
            pair_scores=pair_scores,
            pair_valid=pair_valid,
            candidate_ids=candidate_ids,
            num_classes=self.num_id_vocabulary,
        )
        labels = seq.get("unknown_id_labels")
        masks = seq["unknown_masks"]
        target_present = torch.zeros((), device=logits.device)
        existing_fraction = torch.zeros((), device=logits.device)
        if labels is not None:
            valid_queries = ~masks
            existing = valid_queries & (labels < self.num_id_vocabulary)
            target_candidate = (
                (candidate_ids == labels[..., None]) & pair_valid
            ).any(dim=-1)
            if existing.any():
                target_present = target_candidate[existing].float().mean()
                existing_fraction = existing.float().mean()
            # A query whose target slot has no exact requested lag cannot be
            # scored by this branch.  Exclude it from the auxiliary CE rather
            # than pushing the all-zero newborn logit or inventing a nearest
            # timestamp. Newborn targets are likewise handled exclusively by
            # MOTIP's baseline branch.
            supervised = existing & target_candidate
            self.last_geometry_loss = F.cross_entropy(
                logits[supervised], labels[supervised]
            ) if supervised.any() else logits.sum() * 0.0
        else:
            self.last_geometry_loss = logits.sum() * 0.0
        self.last_pair_scores = pair_scores.detach()
        self.last_pair_valid = pair_valid.detach()
        self.last_candidate_ids = candidate_ids.detach() if self.store_candidate_metadata else None
        self.last_candidate_active = active.detach() if self.store_candidate_metadata else None
        self.last_exact_lag_valid = lag_valid.detach() if self.store_candidate_metadata else None
        self.last_geometry_logits = logits
        base_norm = logits.detach().float().norm(dim=-1).mean().item()
        self.last_stats = {
            "geometry_feature_abs_max": feature_abs_max,
            "geometry_raw_feature_abs_max": raw_feature_abs_max,
            "geometry_logit_norm": float(base_norm),
            "geometry_pair_valid_fraction": float(pair_valid.float().mean().item()),
            "geometry_score_mean": float(pair_scores.detach().float().mean().item()),
            "geometry_score_std": float(pair_scores.detach().float().std().item()),
            "geometry_target_candidate_recall": float(target_present.item()),
            "geometry_existing_query_fraction": float(existing_fraction.item()),
            "geometry_newborn_query_fraction": float(
                ((labels == self.num_id_vocabulary) & ~masks).float().mean().item()
                if labels is not None else 0.0
            ),
        }
        return logits, pair_scores, pair_valid
