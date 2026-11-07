# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.misc import _get_clones, label_to_one_hot
from models.ffn import FFN
from models.motip.geometry import GeometryCompatibilityBranch, GEOMETRY_FEATURE_DIM
from models.motip.trail.trail_residual_branch import TrailResidualBranch
from models.motip.spatial_anchor import BoxSpatialAnchor
from models.motip.spatial_priors import LearnedBoxPositionEmbedding


def _geometry_logit_scale_diagnostics(
        baseline_logits: torch.Tensor,
        geometry_logits: torch.Tensor,
        scale: torch.Tensor,
        base_temperature: float,
        geometry_temperature: float,
) -> dict[str, float]:
    """Return raw and actually fused per-query logit norm diagnostics.

    ``contribution_ratio`` is retained as the historical, untempered ratio.
    ``calibrated_contribution_ratio`` reflects the norms after applying the
    exact temperature and residual scale used by candidate-logit fusion.
    """
    base = baseline_logits.detach().float()
    geometry = geometry_logits.detach().float()
    effective = scale.detach().float() * geometry
    base_norm = base.norm(dim=-1)
    effective_norm = effective.norm(dim=-1)
    calibrated_base_norm = (base / float(base_temperature)).norm(dim=-1)
    calibrated_geometry_norm = (effective / float(geometry_temperature)).norm(dim=-1)
    eps = 1e-6
    return {
        "geometry_logit_norm": float(geometry.norm(dim=-1).mean().item()),
        "base_logit_norm": float(base_norm.mean().item()),
        "effective_geometry_logit_norm": float(effective_norm.mean().item()),
        "contribution_ratio": float((effective_norm / (base_norm + eps)).mean().item()),
        "calibrated_base_logit_norm": float(calibrated_base_norm.mean().item()),
        "calibrated_geometry_logit_norm": float(calibrated_geometry_norm.mean().item()),
        "calibrated_contribution_ratio": float(
            (calibrated_geometry_norm / (calibrated_base_norm + eps)).mean().item()
        ),
    }


def _candidate_geometry_logit_diagnostics(
        baseline_logits: torch.Tensor,
        geometry_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_active: torch.Tensor,
        unknown_masks: torch.Tensor,
        scale: torch.Tensor,
        base_temperature: float,
        geometry_temperature: float,
) -> dict[str, float]:
    """Measure branch scale/ranking on each query's real active IDs plus NEW.

    Unlike ``_geometry_logit_scale_diagnostics``, this masks vocabulary slots
    that are not active candidates and centers each score vector over that
    query's candidate set. Geometry's NEW score remains zero, matching the
    actual residual-fusion policy. Auxiliary decoder layers, when present,
    share the same candidate set and query masks.
    """
    if baseline_logits.shape != geometry_logits.shape:
        raise ValueError("base and geometry logits must have identical shapes")
    if baseline_logits.shape[-1] < 2:
        return {}
    num_classes = baseline_logits.shape[-1] - 1  # final slot is NEW
    prefix = baseline_logits.shape[:-1]
    ids, active = candidate_ids, candidate_active.bool()
    if ids.shape != active.shape:
        raise ValueError("candidate_ids and candidate_active must have identical shapes")
    if ids.shape[:-1] != prefix:
        if ids.ndim + 1 == baseline_logits.ndim and ids.shape[:-1] == prefix[1:]:
            ids = ids.unsqueeze(0).expand((prefix[0],) + tuple(ids.shape))
            active = active.unsqueeze(0).expand_as(ids)
        else:
            raise ValueError(
                f"candidate prefix {tuple(ids.shape[:-1])} does not match logits {tuple(prefix)}"
            )
    query_masks = unknown_masks.bool()
    if query_masks.shape != prefix:
        if query_masks.ndim + 1 == len(prefix) and query_masks.shape == prefix[1:]:
            query_masks = query_masks.unsqueeze(0).expand(prefix)
        else:
            raise ValueError(
                f"query-mask shape {tuple(query_masks.shape)} does not match logits {tuple(prefix)}"
            )

    active = active & (ids >= 0) & (ids < num_classes) & ~query_masks[..., None]
    safe_ids = ids.clamp(min=0, max=num_classes)
    presence = torch.zeros_like(baseline_logits, dtype=torch.int32)
    presence.scatter_add_(-1, safe_ids, active.to(torch.int32))
    candidate_mask = presence > 0
    candidate_mask[..., num_classes] = True  # NEW is always included, never geometry-scored.
    valid_query = active.any(dim=-1)
    if not valid_query.any():
        return {}

    base = baseline_logits.detach().float()
    geom = geometry_logits.detach().float()
    scale_value = scale.detach().float()
    calibrated_base = base / float(base_temperature)
    calibrated_geom = geom * scale_value / float(geometry_temperature)

    def centered_rows(scores: torch.Tensor):
        masked_scores = torch.where(candidate_mask, scores, torch.zeros_like(scores))
        count = candidate_mask.sum(dim=-1).clamp_min(1).to(scores.dtype)
        mean = masked_scores.sum(dim=-1, keepdim=True) / count[..., None]
        centered = torch.where(candidate_mask, scores - mean, torch.zeros_like(scores))
        norm = centered.norm(dim=-1)
        top2 = scores.masked_fill(~candidate_mask, -torch.inf).topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        return centered, norm, margin

    base_centered, base_norm, base_margin = centered_rows(base)
    geom_centered, geom_norm, geom_margin = centered_rows(geom)
    fused_centered, fused_norm, fused_margin = centered_rows(calibrated_base + calibrated_geom)
    calibrated_base_centered, calibrated_base_norm, _ = centered_rows(calibrated_base)
    calibrated_geom_centered, calibrated_geom_norm, _ = centered_rows(calibrated_geom)
    eps = 1e-6
    base_valid = valid_query
    uncalibrated_ratio = geom_norm / (base_norm + eps)
    calibrated_ratio = calibrated_geom_norm / (calibrated_base_norm + eps)
    corr = (base_centered * geom_centered).sum(dim=-1) / (
        base_norm * geom_norm + eps
    )

    def mean_valid(values: torch.Tensor) -> float:
        return float(values[base_valid].mean().item())

    return {
        "candidate_count_mean": mean_valid(candidate_mask.sum(dim=-1).float()),
        "base_candidate_centered_norm": mean_valid(base_norm),
        "geometry_candidate_centered_norm": mean_valid(geom_norm),
        "fused_candidate_centered_norm": mean_valid(fused_norm),
        "geometry_candidate_centered_ratio": mean_valid(uncalibrated_ratio),
        "geometry_calibrated_candidate_centered_ratio": mean_valid(calibrated_ratio),
        "base_candidate_margin": mean_valid(base_margin),
        "geometry_candidate_margin": mean_valid(geom_margin),
        "fused_candidate_margin": mean_valid(fused_margin),
        "base_geometry_candidate_correlation": mean_valid(corr),
        "calibrated_base_candidate_centered_norm": mean_valid(calibrated_base_norm),
        "calibrated_geometry_candidate_centered_norm": mean_valid(calibrated_geom_norm),
        "calibrated_base_geometry_candidate_correlation": mean_valid(
            (calibrated_base_centered * calibrated_geom_centered).sum(dim=-1)
            / (calibrated_base_norm * calibrated_geom_norm + eps)
        ),
    }


class IDDecoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            id_dim: int,
            ffn_dim_ratio: int,
            num_layers: int,
            head_dim: int,
            num_id_vocabulary: int,
            rel_pe_length: int,
            use_aux_loss: bool,
            use_shared_aux_head: bool,
            use_trail: bool = False,
            use_token_type: bool = False,
            trail_fusion_mode: str = "shared_attention",
            trail_freeze_baseline: bool = False,
            trail_residual_init: float = 0.0,
            use_geometry: bool = False,
            geometry_dim: int = GEOMETRY_FEATURE_DIM,
            geometry_hidden_dim: int = 128,
            geometry_lags: tuple[int, ...] = (1, 2, 4, 8),
            geometry_fixed_lambda: float = 0.01,
            geometry_base_temperature: float = 1.0,
            geometry_temperature: float = 1.0,
            geometry_aux_weight: float = 1.0,
            geometry_fused_weight: float = 1.0,
            geometry_learned_lambda: bool = False,
            geometry_freeze_baseline: bool = False,
            geometry_validate_alignment: bool = True,
            geometry_input_clip: float = 50.0,
            geometry_input_normalize: bool = True,
            geometry_input_stats: torch.Tensor | None = None,
            geometry_candidate_diagnostics: bool = False,
            use_spatial_anchor: bool = False,
            use_motip2_spatial_anchor: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.n_heads = (self.feature_dim + self.id_dim) // self.head_dim
        self.num_id_vocabulary = num_id_vocabulary
        self.rel_pe_length = rel_pe_length

        self.use_aux_loss = use_aux_loss
        self.use_shared_aux_head = use_shared_aux_head

        self.word_to_embed = nn.Linear(self.num_id_vocabulary + 1, self.id_dim, bias=False)
        embed_to_word = nn.Linear(self.id_dim, self.num_id_vocabulary + 1, bias=False)

        if self.use_aux_loss and not self.use_shared_aux_head:
            self.embed_to_word_layers = _get_clones(embed_to_word, self.num_layers)
        else:
            self.embed_to_word_layers = nn.ModuleList([embed_to_word for _ in range(self.num_layers)])
        pass

        # Related Position Embeddings:
        self.rel_pos_embeds = nn.Parameter(
            torch.zeros((self.num_layers, self.rel_pe_length, self.n_heads), dtype=torch.float32)
        )
        # Prepare others for rel pe:
        t_idxs = torch.arange(self.rel_pe_length, dtype=torch.int64)
        curr_t_idxs, traj_t_idxs = torch.meshgrid([t_idxs, t_idxs])
        self.rel_pos_map = (curr_t_idxs - traj_t_idxs)      # [curr_t_idx, traj_t_idx] -> rel_pos, like [1, 0] = 1
        pass

        self_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        self_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)
        cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        cross_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)
        ffn = FFN(
            d_model=self.feature_dim + self.id_dim,
            d_ffn=(self.feature_dim + self.id_dim) * self.ffn_dim_ratio,
            activation=nn.GELU(),
        )
        ffn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)

        self.self_attn_layers = _get_clones(self_attn, self.num_layers - 1)
        self.self_attn_norm_layers = _get_clones(self_attn_norm, self.num_layers - 1)
        self.cross_attn_layers = _get_clones(cross_attn, self.num_layers)
        self.cross_attn_norm_layers = _get_clones(cross_attn_norm, self.num_layers)
        self.ffn_layers = _get_clones(ffn, self.num_layers)
        self.ffn_norm_layers = _get_clones(ffn_norm, self.num_layers)

        # Init parameters:
        for n, p in self.named_parameters():
            if p.dim() > 1 and "rel_pos_embeds" not in n:
                nn.init.xavier_uniform_(p)

        self.use_trail = use_trail
        self.trail_fusion_mode = str(trail_fusion_mode).lower()
        if self.trail_fusion_mode not in {"shared_attention", "residual_logits"}:
            raise ValueError(f"Unknown trail fusion mode: {trail_fusion_mode}")
        self.trail_attention_mass = 0.0
        self.trail_context_scale = 1.0
        self.trail_logit_norm = 0.0
        self.baseline_logit_norm = 0.0
        self.effective_trail_logit_norm = 0.0
        # Runtime-only diagnostics.  Disabled by default so the baseline and
        # training memory footprint remain unchanged.  When enabled we retain
        # scalar attention summaries (never the full attention tensors).
        self.collect_attention_diagnostics = False
        self.trail_attention_stats = []
        self.trail_projection = None
        self.token_type = None
        self.trail_residual_branch = None
        self.residual_scale = None
        if use_trail and self.trail_fusion_mode == "shared_attention":
            # Reuse the original dictionary. W_h starts as identity on [z;i].
            with torch.random.fork_rng(devices=[]):
                self.trail_projection = nn.Linear(feature_dim + id_dim, feature_dim + id_dim, bias=False)
            nn.init.eye_(self.trail_projection.weight)
            self.token_type = nn.Parameter(torch.zeros(2, feature_dim + id_dim)) if use_token_type else None
        elif use_trail and self.trail_fusion_mode == "residual_logits":
            self.trail_residual_branch = TrailResidualBranch(
                feature_dim=feature_dim,
                id_dim=id_dim,
                head_dim=head_dim,
                num_id_vocabulary=num_id_vocabulary,
                ffn_dim_ratio=ffn_dim_ratio,
            )
            self.residual_scale = nn.Parameter(torch.tensor(float(trail_residual_init)))

        self.use_geometry = bool(use_geometry)
        self.geometry_branch = None
        self.geometry_lambda = None
        self.geometry_fixed_lambda = float(geometry_fixed_lambda)
        self.geometry_base_temperature = float(geometry_base_temperature)
        self.geometry_temperature = float(geometry_temperature)
        if self.geometry_base_temperature <= 0 or self.geometry_temperature <= 0:
            raise ValueError("Geometry/base temperatures must be positive")
        self.geometry_aux_weight = float(geometry_aux_weight)
        self.geometry_fused_weight = float(geometry_fused_weight)
        self.compute_geometry_candidate_diagnostics = bool(geometry_candidate_diagnostics)
        self.geometry_aux_loss = None
        self.geometry_logit_norm = 0.0
        self.geometry_base_logit_norm = 0.0
        self.geometry_effective_logit_norm = 0.0
        self.geometry_contribution_ratio = 0.0
        self.geometry_calibrated_base_logit_norm = 0.0
        self.geometry_calibrated_effective_logit_norm = 0.0
        self.geometry_calibrated_contribution_ratio = 0.0
        self.geometry_logit_correlation = 0.0
        self.geometry_margin = 0.0
        self.geometry_candidate_diagnostics = {}
        self.last_base_logits = None
        self.last_geometry_logits = None
        self.last_fused_logits = None
        if self.use_geometry:
            # Optional G4 parameters must not shift initialization/data RNG for
            # shared MOTIP modules in a matched ablation. The branch keeps the
            # same deterministic initialization for a given seed, while its
            # random draws are isolated from the caller's RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.geometry_branch = GeometryCompatibilityBranch(
                    feature_dim=geometry_dim,
                    hidden_dim=geometry_hidden_dim,
                    num_id_vocabulary=num_id_vocabulary,
                    lags=geometry_lags,
                    fixed_lambda=geometry_fixed_lambda,
                    aux_weight=geometry_aux_weight,
                    fused_weight=geometry_fused_weight,
                    validate_alignment=geometry_validate_alignment,
                    input_clip=geometry_input_clip,
                    input_normalize=geometry_input_normalize,
                    input_stats=geometry_input_stats,
                    store_candidate_metadata=geometry_candidate_diagnostics,
                )
            if geometry_learned_lambda:
                self.geometry_lambda = nn.Parameter(torch.tensor(float(geometry_fixed_lambda)))

        if trail_freeze_baseline:
            self.freeze_baseline_branch()
        if geometry_freeze_baseline:
            self.freeze_geometry_baseline()

        self.use_spatial_anchor = bool(use_spatial_anchor)
        self.use_motip2_spatial_anchor = bool(use_motip2_spatial_anchor)
        self.spatial_anchor = None
        self.motip2_spatial_anchor = None
        if self.use_spatial_anchor and self.use_motip2_spatial_anchor:
            raise ValueError("Select either the simple BoxAnchor ablation or the MOTIP2-style box embedding")
        if self.use_spatial_anchor:
            if self.use_geometry or self.use_trail:
                raise ValueError("BoxAnchor is a standalone E0 ablation, not a V3/trail modification")
            # Enabling the prior must not change original weight initialization
            # or the subsequent augmentation/sampling RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.spatial_anchor = BoxSpatialAnchor(feature_dim)
        if self.use_motip2_spatial_anchor:
            if self.use_trail:
                raise ValueError("MOTIP2-style position embedding cannot be combined with VisualTrail")
            # The current Deformable-DETR code does not expose the D-FINE
            # final-box positional embedding used by MOTIP2. This learned
            # box-to-token mapping is an explicitly documented adaptation.
            with torch.random.fork_rng(devices=[]):
                self.motip2_spatial_anchor = LearnedBoxPositionEmbedding(feature_dim)

        pass

    def forward(self, seq_info, use_decoder_checkpoint):
        if seq_info["unknown_features"].shape[-2] == 0:
            return self._empty_output(seq_info)
        if self.use_spatial_anchor:
            # Do not mutate caller-owned tensors or accumulate positional
            # embeddings across repeated diagnostic calls.
            seq_info = dict(seq_info)
            for prefix in ("trajectory", "unknown"):
                features = seq_info[f"{prefix}_features"]
                anchor = self.spatial_anchor(seq_info[f"{prefix}_boxes"], seq_info[f"{prefix}_masks"])
                if features.shape != anchor.shape:
                    raise ValueError(f"{prefix} feature/box alignment mismatch")
                seq_info[f"{prefix}_features"] = features + anchor.to(features.dtype)
        if self.use_motip2_spatial_anchor:
            seq_info = dict(seq_info)
            for prefix in ("trajectory", "unknown"):
                features = seq_info[f"{prefix}_features"]
                position = self.motip2_spatial_anchor(
                    seq_info[f"{prefix}_boxes"], seq_info[f"{prefix}_masks"]
                )
                if features.shape != position.shape:
                    raise ValueError(f"{prefix} feature/position alignment mismatch")
                seq_info[f"{prefix}_features"] = features + position.to(features.dtype)
        if self.use_geometry:
            if self.use_trail:
                raise ValueError("V3 geometry cannot be combined with the visual-trail decoder path")
            return self._forward_geometry(seq_info, use_decoder_checkpoint)
        if self.use_trail and "trail_tokens" in seq_info:
            if self.trail_fusion_mode == "residual_logits":
                return self._forward_residual_logits(seq_info, use_decoder_checkpoint)
            return self._forward_with_trails(seq_info, use_decoder_checkpoint)
        return self._forward_baseline(seq_info, use_decoder_checkpoint)

    def freeze_baseline_branch(self):
        """Freeze original MOTIP decoder parameters, leaving only trail add-ons."""
        trail_names = ("trail_projection", "token_type", "trail_residual_branch", "residual_scale")
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(trail_names)

    def freeze_geometry_baseline(self):
        """Freeze MOTIP while retaining only the V3 geometry branch."""
        geometry_names = ("geometry_branch.", "geometry_lambda")
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(geometry_names)

    def trainability_report(self):
        report = {}
        for name, parameter in self.named_parameters():
            if name.startswith(("geometry_branch", "geometry_lambda")):
                group = "geometry"
            elif name.startswith("spatial_anchor."):
                group = "spatial_anchor"
            elif name.startswith(("trail_projection", "token_type", "trail_residual_branch", "residual_scale")):
                group = "trail"
            else:
                group = "baseline"
            item = report.setdefault(group, {"parameters": 0, "trainable": 0, "names": []})
            item["parameters"] += parameter.numel()
            if parameter.requires_grad:
                item["trainable"] += parameter.numel()
                item["names"].append(name)
        return report

    def _forward_residual_logits(self, seq_info, use_decoder_checkpoint):
        baseline_logits, labels, masks = self._forward_baseline(seq_info, use_decoder_checkpoint)
        trail_logits = self.trail_residual_branch(
            seq_info, self.id_label_to_embed, self.generate_empty_id_embed,
        )
        if self.training and self.use_aux_loss:
            trail_logits = trail_logits.repeat(self.num_layers, 1, 1, 1, 1)
        scale = self.residual_scale.to(dtype=baseline_logits.dtype)
        self.baseline_logit_norm = float(baseline_logits.detach().float().norm(dim=-1).mean().item())
        self.trail_logit_norm = float(trail_logits.detach().float().norm(dim=-1).mean().item())
        self.effective_trail_logit_norm = float((scale * trail_logits).detach().float().norm(dim=-1).mean().item())
        return baseline_logits + scale * trail_logits, labels, masks

    def _forward_geometry(self, seq_info, use_decoder_checkpoint):
        # The original decoder is deliberately evaluated unchanged.  Geometry
        # is a candidate-wise residual in ID-logit space, never a replacement
        # for the MOTIP history pathway.
        baseline_logits, labels, masks = self._forward_baseline(
            seq_info, use_decoder_checkpoint
        )
        geometry_logits, _, _ = self.geometry_branch(seq_info)
        if self.training and self.use_aux_loss:
            geometry_for_fusion = geometry_logits.repeat(self.num_layers, 1, 1, 1, 1)
        else:
            geometry_for_fusion = geometry_logits
        if self.geometry_lambda is None:
            scale = baseline_logits.new_tensor(self.geometry_fixed_lambda)
        else:
            scale = self.geometry_lambda.to(dtype=baseline_logits.dtype)
        self.geometry_aux_loss = self.geometry_branch.last_geometry_loss * self.geometry_aux_weight
        effective = scale * geometry_for_fusion
        scale_stats = _geometry_logit_scale_diagnostics(
            baseline_logits,
            geometry_for_fusion,
            scale,
            self.geometry_base_temperature,
            self.geometry_temperature,
        )
        self.geometry_logit_norm = scale_stats["geometry_logit_norm"]
        self.geometry_base_logit_norm = scale_stats["base_logit_norm"]
        self.geometry_effective_logit_norm = scale_stats["effective_geometry_logit_norm"]
        self.geometry_contribution_ratio = scale_stats["contribution_ratio"]
        self.geometry_calibrated_base_logit_norm = scale_stats["calibrated_base_logit_norm"]
        self.geometry_calibrated_effective_logit_norm = scale_stats["calibrated_geometry_logit_norm"]
        self.geometry_calibrated_contribution_ratio = scale_stats["calibrated_contribution_ratio"]
        self.geometry_candidate_diagnostics = (
            _candidate_geometry_logit_diagnostics(
                baseline_logits=baseline_logits,
                geometry_logits=geometry_for_fusion,
                candidate_ids=self.geometry_branch.last_candidate_ids,
                candidate_active=self.geometry_branch.last_candidate_active,
                unknown_masks=seq_info["unknown_masks"],
                scale=scale,
                base_temperature=self.geometry_base_temperature,
                geometry_temperature=self.geometry_temperature,
            )
            if self.compute_geometry_candidate_diagnostics else {}
        )
        base_flat = baseline_logits.detach().float().reshape(-1, baseline_logits.shape[-1])
        geom_flat = geometry_for_fusion.detach().float().reshape(-1, geometry_for_fusion.shape[-1])
        base_center = base_flat - base_flat.mean(dim=-1, keepdim=True)
        geom_center = geom_flat - geom_flat.mean(dim=-1, keepdim=True)
        correlation = (base_center * geom_center).sum(-1) / (
            base_center.norm(dim=-1) * geom_center.norm(dim=-1) + 1e-6
        )
        self.geometry_logit_correlation = float(correlation.mean().item())
        top2 = torch.topk(geometry_logits[..., :self.num_id_vocabulary], k=min(2, self.num_id_vocabulary), dim=-1).values
        self.geometry_margin = float((top2[..., 0] - top2[..., -1]).detach().float().mean().item())
        # Temperature scaling is an inference/calibration diagnostic.  The
        # raw branch logits remain exposed above; only the residual fusion
        # path is calibrated.  With USE_GEOMETRY=false the original MOTIP
        # path is untouched.
        fused_logits = (
            baseline_logits / self.geometry_base_temperature
            + scale * geometry_for_fusion / self.geometry_temperature
        )
        self.last_base_logits = baseline_logits.detach()
        self.last_geometry_logits = geometry_for_fusion.detach()
        self.last_fused_logits = fused_logits.detach()
        return fused_logits, labels, masks

    def _forward_baseline(self, seq_info, use_decoder_checkpoint):
        if seq_info["unknown_features"].shape[-2] == 0:
            return self._empty_output(seq_info)
        trajectory_features = seq_info["trajectory_features"]
        unknown_features = seq_info["unknown_features"]
        trajectory_id_labels = seq_info["trajectory_id_labels"]
        unknown_id_labels = seq_info["unknown_id_labels"] if "unknown_id_labels" in seq_info else None
        trajectory_times = seq_info["trajectory_times"]
        unknown_times = seq_info["unknown_times"]
        trajectory_masks = seq_info["trajectory_masks"]
        unknown_masks = seq_info["unknown_masks"]
        _B, _G, _T, _N, _ = trajectory_features.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_features.shape

        trajectory_id_embeds = self.id_label_to_embed(id_labels=trajectory_id_labels)
        unknown_id_embeds = self.generate_empty_id_embed(unknown_features=unknown_features)

        trajectory_embeds = torch.cat([trajectory_features, trajectory_id_embeds], dim=-1)
        unknown_embeds = torch.cat([unknown_features, unknown_id_embeds], dim=-1)

        # Prepare some common variables:
        self_attn_key_padding_mask = einops.rearrange(unknown_masks, "b g t n -> (b g t) n").contiguous()
        cross_attn_key_padding_mask = einops.rearrange(trajectory_masks, "b g t n -> (b g) (t n)").contiguous()
        _trajectory_times_flatten = einops.rearrange(trajectory_times, "b g t n -> (b g) (t n)")
        _unknown_times_flatten = einops.rearrange(unknown_times, "b g t n -> (b g) (t n)")
        cross_attn_mask = _trajectory_times_flatten[:, None, :] >= _unknown_times_flatten[:, :, None]
        cross_attn_mask = einops.repeat(cross_attn_mask, "bg tn1 tn2 -> (bg n_heads) tn1 tn2", n_heads=self.n_heads).contiguous()
        # Prepare for rel PE:
        self.rel_pos_map = self.rel_pos_map.to(trajectory_features.device)
        rel_pe_idx_pairs = torch.stack([
            torch.stack(
                torch.meshgrid([_unknown_times_flatten[_], _trajectory_times_flatten[_]]), dim=-1
            )
            for _ in range(len(_trajectory_times_flatten))
        ], dim=0)       # (B*G, T*N of curr, T*N of traj, 2)
        rel_pe_idx_pairs = rel_pe_idx_pairs.to(trajectory_features.device)
        rel_pe_idxs = self.rel_pos_map[rel_pe_idx_pairs[..., 0], rel_pe_idx_pairs[..., 1]]      # (B*G, T_curr, T_traj)
        pass
        # Change Cross-Attn key_padding_mask and attn_mask to float:
        cross_attn_key_padding_mask = torch.masked_fill(
            cross_attn_key_padding_mask.float(),
            mask=cross_attn_key_padding_mask,
            value=float("-inf"),
        ).to(self.dtype)
        cross_attn_mask = torch.masked_fill(
            cross_attn_mask.float(),
            mask=cross_attn_mask,
            value=float("-inf"),
        ).to(self.dtype)
        pass

        all_unknown_id_logits = None
        all_unknown_id_labels = None
        all_unknown_id_masks = None

        for layer in range(self.num_layers):
            # Predict ID logits:
            if use_decoder_checkpoint:
                unknown_embeds = checkpoint(
                    self._forward_a_layer,
                    layer,
                    unknown_embeds, trajectory_embeds,
                    self_attn_key_padding_mask, cross_attn_key_padding_mask,
                    cross_attn_mask, rel_pe_idxs,
                    use_reentrant=False,
                )
            else:
                unknown_embeds = self._forward_a_layer(
                    layer=layer,
                    unknown_embeds=unknown_embeds,
                    trajectory_embeds=trajectory_embeds,
                    self_attn_key_padding_mask=self_attn_key_padding_mask,
                    cross_attn_key_padding_mask=cross_attn_key_padding_mask,
                    cross_attn_mask=cross_attn_mask,
                    rel_pe_idx=rel_pe_idxs,
                )

            _unknown_id_logits = self.embed_to_word_layers[layer](unknown_embeds[..., -self.id_dim:])
            _unknown_id_masks = unknown_masks.clone()
            _unknown_id_labels = None if not self.training else unknown_id_labels
            if all_unknown_id_logits is None:
                all_unknown_id_logits = _unknown_id_logits
                all_unknown_id_labels = _unknown_id_labels
                all_unknown_id_masks = _unknown_id_masks
            else:
                all_unknown_id_logits = torch.cat([all_unknown_id_logits, _unknown_id_logits], dim=0)
                all_unknown_id_labels = torch.cat([all_unknown_id_labels, _unknown_id_labels], dim=0) if _unknown_id_labels is not None else None
                all_unknown_id_masks = torch.cat([all_unknown_id_masks, _unknown_id_masks], dim=0)

        if self.training and self.use_aux_loss:
            return all_unknown_id_logits, all_unknown_id_labels, all_unknown_id_masks
        else:
            return _unknown_id_logits, _unknown_id_labels, _unknown_id_masks

    def _forward_a_layer(
            self,
            layer: int,
            unknown_embeds: torch.Tensor,
            trajectory_embeds: torch.Tensor,
            self_attn_key_padding_mask: torch.Tensor,
            cross_attn_key_padding_mask: torch.Tensor,
            cross_attn_mask: torch.Tensor,
            rel_pe_idx: torch.Tensor,
            trail_key_count: int = 0,
    ):
        _B, _G, _T, _N, _ = trajectory_embeds.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_embeds.shape
        if layer > 0:   # use self-attention to transfer information between unknown features (same time step)
            self_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g t) n c").contiguous()
            self_out, _ = self.self_attn_layers[layer - 1](
                query=self_unknown_embeds, key=self_unknown_embeds, value=self_unknown_embeds,
                key_padding_mask=self_attn_key_padding_mask,
            )
            self_out = self_unknown_embeds + self_out
            self_out = self.self_attn_norm_layers[layer - 1](self_out)
            unknown_embeds = einops.rearrange(self_out, "(b g t) n c -> b g t n c", b=_B, g=_G, t=_curr_T)

        # Cross-attention for in-context decoding:
        cross_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        cross_trajectory_embeds = einops.rearrange(trajectory_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        # Prepare attn_mask:
        rel_pe_mask = self.rel_pos_embeds[layer][rel_pe_idx]
        cross_attn_mask_with_rel_pe = cross_attn_mask + einops.rearrange(rel_pe_mask, "bg l1 l2 n -> (bg n) l1 l2")
        # Apply cross-attention:
        collect_attention = bool(getattr(self, "collect_attention_diagnostics", False))
        cross_out, attention = self.cross_attn_layers[layer](
            query=cross_unknown_embeds, key=cross_trajectory_embeds, value=cross_trajectory_embeds,
            key_padding_mask=cross_attn_key_padding_mask,
            attn_mask=cross_attn_mask_with_rel_pe,
            average_attn_weights=not collect_attention,
        )
        if trail_key_count and (layer == self.num_layers - 1 or collect_attention):
            # ``add_zero_attn`` appends a final null key.  The previous
            # implementation reported the sum of all trail-key probabilities
            # for the final query snapshot, averaged over query positions.  It
            # is a valid *total* mass, but it hides token-count imbalance.  In
            # diagnostic mode also retain history/trail masses, valid token
            # counts, per-token masses and per-head values.
            weights = attention.detach()
            if weights.ndim == 3:  # PyTorch's default head-averaged weights.
                weights = weights.unsqueeze(1)
            source_count = cross_trajectory_embeds.shape[1]
            history_count = source_count - trail_key_count
            if history_count < 0:
                raise ValueError("trail_key_count exceeds decoder context length")
            trail_start, trail_end = history_count, source_count
            history = weights[..., :history_count]
            trail = weights[..., trail_start:trail_end]
            # Padding masks are represented by finite zero / -inf values and
            # are expanded over all query positions by MultiheadAttention.
            valid = torch.isfinite(cross_attn_key_padding_mask)
            valid_history = valid[:, :history_count]
            valid_trail = valid[:, trail_start:trail_end]
            h_count = valid_history.sum(-1).clamp_min(1)
            t_count = valid_trail.sum(-1).clamp_min(1)
            h_mass = history.sum(-1)
            t_mass = trail.sum(-1)
            null_mass = weights[..., source_count:].sum(-1)
            self.trail_attention_mass = t_mass.mean().item()
            if collect_attention:
                # [bg, heads, query] -> scalar/per-head summaries.
                self.trail_attention_stats.append({
                    "layer": int(layer),
                    "batch_groups": int(weights.shape[0]),
                    "query_tokens": int(weights.shape[2]),
                    "history_tokens": int(history_count),
                    "trail_tokens": int(trail_key_count),
                    "history_valid_tokens_mean": float(valid_history.sum(-1).float().mean().item()),
                    "trail_valid_tokens_mean": float(valid_trail.sum(-1).float().mean().item()),
                    "history_mass_total": float(h_mass.mean().item()),
                    "trail_mass_total": float(t_mass.mean().item()),
                    "null_mass_total": float(null_mass.mean().item()),
                    "history_mass_per_valid_token": float((h_mass / h_count[:, None, None]).mean().item()),
                    "trail_mass_per_valid_token": float((t_mass / t_count[:, None, None]).mean().item()),
                    "history_mass_per_head": [float(x) for x in h_mass.mean((0, 2)).cpu().tolist()],
                    "trail_mass_per_head": [float(x) for x in t_mass.mean((0, 2)).cpu().tolist()],
                    "history_mass_per_query": [float(x) for x in h_mass.mean((0, 1)).cpu().tolist()],
                    "trail_mass_per_query": [float(x) for x in t_mass.mean((0, 1)).cpu().tolist()],
                })
        cross_out = cross_unknown_embeds + cross_out
        cross_out = self.cross_attn_norm_layers[layer](cross_out)
        # Feed-forward network:
        cross_out = cross_out + self.ffn_layers[layer](cross_out)
        cross_out = self.ffn_norm_layers[layer](cross_out)
        # Re-shape back to original shape:
        unknown_embeds = einops.rearrange(cross_out, "(b g) (t n) c -> b g t n c", b=_B, g=_G, t=_curr_T)

        return unknown_embeds

    def _forward_with_trails(self, seq, use_checkpoint):
        """One causal trail snapshot per query time; history is retained in full.

        Avoid T copies of prefix tokens as keys to all T queries. Each time's
        query attends to T*N original tokens and K*P additional trail tokens.
        This is mathematically the same query-wise masked union, with smaller
        attention masks/activations on the 16GB target GPU.
        """
        features = seq["trajectory_features"]
        if getattr(self, "collect_attention_diagnostics", False):
            self.trail_attention_stats = []
        else:
            self.trail_attention_stats = []
        self.trail_attention_mass = 0.0
        b, g, t, n, _ = features.shape
        bg = b * g
        appearance = torch.cat((features, self.id_label_to_embed(seq["trajectory_id_labels"])), -1)
        appearance = appearance.reshape(b, g, 1, t * n, -1)
        history_times = seq["trajectory_times"].reshape(bg, t * n)
        history_masks = seq["trajectory_masks"].reshape(bg, t * n)
        queries = torch.cat((seq["unknown_features"], self.generate_empty_id_embed(seq["unknown_features"])), -1)
        if self.token_type is not None:
            appearance = appearance + self.token_type[0]
        all_times = []
        for q in range(queries.shape[2]):
            tokens = seq["trail_tokens"][:, :, q]
            k, p = tokens.shape[2:4]
            ids = self.id_label_to_embed(seq["trail_slot_ids"][:, :, q])
            ids = ids.unsqueeze(-2).expand(b, g, k, p, self.id_dim)
            trail = self.trail_projection(torch.cat((tokens, ids), -1))
            if self.token_type is not None:
                trail = trail + self.token_type[1]
            trail = trail * float(getattr(self, "trail_context_scale", 1.0))
            context = torch.cat((appearance, trail.reshape(b, g, 1, k * p, -1)), 3)
            query = queries[:, :, q:q+1]
            qt = seq["unknown_times"][:, :, q].reshape(bg, -1)
            mask_history = history_times[:, None, :] >= qt[:, :, None]
            mask_trail = torch.zeros((bg, qt.shape[-1], k * p), device=qt.device, dtype=torch.bool)
            causal = torch.cat((mask_history, mask_trail), -1)
            padding = torch.cat((history_masks, seq["trail_masks"][:, :, q, :, None].expand(b, g, k, p).reshape(bg, k*p)), -1)
            # Original relative PE for history; trail snapshot is at q-1.
            rel_history = qt[:, :, None] - history_times[:, None, :]
            rel_history = rel_history.clamp(-self.rel_pe_length + 1, self.rel_pe_length - 1)
            rel = torch.cat((rel_history, torch.ones_like(mask_trail, dtype=torch.long)), -1).long()
            attn_mask = causal.to(self.dtype).masked_fill(causal, float("-inf"))
            attn_mask = attn_mask.repeat_interleave(self.n_heads, 0)
            pad_mask = padding.to(self.dtype).masked_fill(padding, float("-inf"))
            self_mask = seq["unknown_masks"][:, :, q].reshape(bg, -1)
            layers = []
            for layer in range(self.num_layers):
                args = (layer, query, context, self_mask, pad_mask, attn_mask, rel, k*p)
                query = checkpoint(self._forward_a_layer, *args, use_reentrant=False) if use_checkpoint \
                    else self._forward_a_layer(*args)
                layers.append(self.embed_to_word_layers[layer](query[..., -self.id_dim:]))
            all_times.append(layers)
        if self.training and self.use_aux_loss:
            logits = torch.cat([torch.cat([x[layer] for x in all_times], 2) for layer in range(self.num_layers)], 0)
            labels = seq["unknown_id_labels"].repeat(self.num_layers, 1, 1, 1)
            masks = seq["unknown_masks"].repeat(self.num_layers, 1, 1, 1)
        else:
            logits = torch.cat([x[-1] for x in all_times], 2)
            labels = seq.get("unknown_id_labels") if self.training else None
            masks = seq["unknown_masks"]
        return logits, labels, masks

    def _empty_output(self, seq):
        """Return an empty ID prediction while preserving auxiliary-loss layout."""
        output = seq["unknown_features"].new_empty(
            (*seq["unknown_features"].shape[:-1], self.num_id_vocabulary + 1)
        )
        labels = seq.get("unknown_id_labels") if self.training else None
        masks = seq["unknown_masks"]
        self.trail_attention_mass = 0.0
        self.trail_attention_stats = []
        self.geometry_aux_loss = None
        if self.training and self.use_aux_loss:
            output = output.repeat(self.num_layers, 1, 1, 1, 1)
            masks = masks.repeat(self.num_layers, 1, 1, 1)
            if labels is not None:
                labels = labels.repeat(self.num_layers, 1, 1, 1)
        return output, labels, masks

    def id_label_to_embed(self, id_labels):
        id_words = label_to_one_hot(id_labels, self.num_id_vocabulary + 1, dtype=self.dtype)
        id_embeds = self.word_to_embed(id_words)
        return id_embeds

    def generate_empty_id_embed(self, unknown_features):
        _shape = unknown_features.shape[:-1]
        empty_id_labels = self.num_id_vocabulary * torch.ones(_shape, dtype=torch.int64, device=unknown_features.device)
        empty_id_embeds = self.id_label_to_embed(id_labels=empty_id_labels)
        return empty_id_embeds

    def shuffle(self):
        shuffle_index = torch.randperm(self.num_id_vocabulary, device=self.word_to_embed.weight.device)
        shuffle_index = torch.cat([shuffle_index, torch.tensor([self.num_id_vocabulary], device=self.word_to_embed.weight.device)])
        self.word_to_embed.weight.data = self.word_to_embed.weight.data[:, shuffle_index]
        self.embed_to_word.weight.data = self.embed_to_word.weight.data[shuffle_index, :]
        pass

    @property
    def dtype(self):
        return self.word_to_embed.weight.dtype
