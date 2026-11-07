"""Small frozen-branch fusion experiments; no tracker/lifecycle changes.

Features are deliberately inference-available only.  GT ownership, target
age/motion, labels and numerical identity indices are not accepted as inputs.
All native vocabulary logits (including NEW and inactive slots) are retained.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES = (
    'base_score', 'geometry_score', 'base_probability', 'geometry_probability',
    'base_rank', 'geometry_rank', 'geometry_valid', 'log_track_age',
    'log_time_since_seen', 'valid_lag_fraction', 'lag1', 'lag2', 'lag4', 'lag8',
    'scale_normalized_displacement', 'log_current_area', 'log_current_aspect',
    'base_margin', 'base_entropy', 'base_pmax', 'base_new_probability',
    'base_mapped_new_mass', 'geometry_margin', 'geometry_entropy', 'geometry_pmax',
    'top1_agreement', 'log_candidate_count', 'log_geometry_candidate_count',
)
METHODS = ('GLOBAL', 'GATE', 'RESIDUAL_CONTEXT')


def _confidence(scores, valid):
    """Masked branch probabilities and confidence; empty sets return zeros."""
    count = valid.sum(-1)
    safe = valid.clone()
    safe[~safe.any(-1), -1] = True
    values = scores.masked_fill(~safe, -torch.inf)
    p = values.softmax(-1).masked_fill(~valid, 0.)
    top = values.topk(min(2, scores.shape[-1]), dim=-1).values
    margin = top[..., 0] - top[..., -1]
    margin = torch.where(count > 1, margin, torch.zeros_like(margin))
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    # Expected ranks under ties, never tie-breaking by the numeric ID slot.
    comparisons = scores[:, None, :] > scores[:, :, None]
    ties = scores[:, None, :] == scores[:, :, None]
    ranks = 1 + (comparisons & valid[:, None, :]).sum(-1)
    ranks = ranks + .5 * ((ties & valid[:, None, :]).sum(-1) - 1)
    ranks = torch.where(valid, ranks / count[:, None].clamp_min(1), 0.)
    best = valid & (scores == values.max(-1, keepdim=True).values)
    return p, margin, entropy, p.max(-1).values, ranks, best


def build_fusion_features(base, geometry, slots, history, valid, age, gap, current):
    """Shared offline/online builder. Inputs contain no annotations.

    base/geometry:[N,V], current:[N,4], slots:[K], history:[K,8,4],
    valid:[K,8], age/gap:[K], actual-frame predicted-track lifecycle only.
    NEW is V-1. G4 support is any exact lag1/2/4/8, not nearest observation.
    """
    n, vocab = base.shape
    if geometry.shape != base.shape or current.shape != (n, 4):
        raise ValueError('Query/logit shape mismatch')
    k = slots.numel()
    if history.shape != (k, 8, 4) or valid.shape != (k, 8):
        raise ValueError('History shape mismatch')
    if slots.unique().numel() != k or (slots < 0).any() or (slots >= vocab-1).any():
        raise AssertionError('Invalid/duplicate local track slot')
    if not torch.isfinite(base).all() or not torch.isfinite(geometry).all():
        raise FloatingPointError('Nonfinite fusion logits')
    active = torch.zeros_like(base, dtype=torch.bool)
    active[:, slots] = True
    usable = torch.zeros_like(active)
    sparse = valid[:, [7, 6, 4, 0]]
    usable[:, slots] = sparse.any(-1)[None]
    if (geometry.masked_select(~usable) != 0).any():
        raise AssertionError('Geometry must be zero for NEW/inactive/unavailable candidates')
    all_classes = torch.ones_like(active)
    bp, bm, be, bmax, br, bbest = _confidence(base, all_classes)
    gp, gm, ge, gmax, gr, gbest = _confidence(geometry, usable)
    mapped_new = (~active).to(bp.dtype).mul(bp).sum(-1)
    agreement = (bbest & gbest).any(-1).to(base.dtype)
    age_full = torch.zeros_like(base)
    gap_full = torch.zeros_like(base)
    age_full[:, slots] = age.to(base.dtype).log1p()[None]
    gap_full[:, slots] = gap.to(base.dtype).log1p()[None]
    lag = base.new_zeros(n, vocab, 4)
    lag[:, slots] = sparse.to(base.dtype)[None]
    motion = torch.zeros_like(base)
    if k:
        history = torch.where(valid[..., None], history, torch.zeros_like(history))
        last = torch.where(valid, torch.arange(8, device=valid.device), -1).max(-1).values.clamp_min(0)
        anchor = history[torch.arange(k, device=valid.device), last]
        displacement = (current[:, None, :2] - anchor[None, :, :2]) / anchor[None, :, 2:].clamp_min(1e-4)
        distance = displacement.norm(dim=-1).clamp_max(100.)
        motion[:, slots] = torch.where(valid.any(-1)[None], distance, 0.)
    area = current[:, 2:].clamp_min(1e-6).log().sum(-1)
    aspect = current[:, 2].clamp_min(1e-6).log()-current[:, 3].clamp_min(1e-6).log()
    broadcast = lambda v: v[:, None].expand_as(base)
    values = [base, geometry, bp, gp, br, gr, usable.to(base.dtype), age_full,
        gap_full, lag.mean(-1), *lag.unbind(-1), motion, broadcast(area), broadcast(aspect),
        broadcast(bm), broadcast(be), broadcast(bmax), broadcast(bp[:, -1]),
        broadcast(mapped_new), broadcast(gm), broadcast(ge), broadcast(gmax),
        broadcast(agreement), broadcast(active.sum(-1).to(base.dtype).log1p()),
        broadcast(usable.sum(-1).to(base.dtype).log1p())]
    features = torch.stack(values, -1)
    assert features.shape == (n, vocab, len(FEATURE_NAMES))
    if not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite inference-available fusion feature')
    return features, active, usable


def training_statistics(features, active):
    samples = features[active].float()
    if len(samples) < 2:
        raise ValueError('Insufficient official-train feature samples')
    return torch.stack((samples.mean(0), samples.std(0).clamp_min(1e-4)))


class LearnedCandidateFusion(nn.Module):
    """Three predefined alternatives, initialized without a zero-gradient gate."""
    def __init__(self, method, stats):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.register_buffer('mean', stats[0].clone())
        self.register_buffer('std', stats[1].clone())
        if method in ('GLOBAL', 'GATE'):
            # Initialization equals the validated calibration; both coefficients
            # are learned, not separately swept temperatures/lambda bounds.
            self.base_raw = nn.Parameter(torch.tensor(math.log(math.expm1(.5))))
        if method == 'GLOBAL':
            self.geometry_raw = nn.Parameter(torch.tensor(math.log(math.expm1(2.))))
        elif method == 'GATE':
            self.gate = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.normal_(self.gate[-1].weight, std=.001)
            nn.init.constant_(self.gate[-1].bias, math.log(math.expm1(2.)))
        else:
            self.encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.GELU())
            self.correction = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
            # Small, nonzero final weights: upstream gradients exist on the
            # first backward pass. There is no multiplicative zero lambda.
            nn.init.normal_(self.correction[-1].weight, std=.001)
            nn.init.zeros_(self.correction[-1].bias)

    def forward(self, base, geometry, features, active, usable):
        if features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError('Feature contract changed')
        if self.method == 'GLOBAL':
            proposed = F.softplus(self.base_raw)*base + F.softplus(self.geometry_raw)*geometry
        else:
            z = ((features-self.mean)/self.std).clamp(-20., 20.)
            if self.method == 'GATE':
                weights = F.softplus(self.gate(z).squeeze(-1))
                proposed = F.softplus(self.base_raw)*base + weights*geometry
            else:
                h = self.encoder(z)
                masked = torch.where(active[..., None], h, torch.zeros_like(h))
                context = masked.sum(1)/active.sum(1, keepdim=True).clamp_min(1)
                context = context[:, None].expand_as(h)
                delta = self.correction(torch.cat((h, context), -1)).squeeze(-1)
                proposed = base + delta.masked_fill(~usable, 0.)
        # A query with no usable geometry falls back EXACTLY to native MOTIP,
        # including its confidence and NEW evidence. No GT controls this mask.
        return torch.where(usable.any(-1, keepdim=True), proposed, base)

    def coefficients(self):
        if self.method == 'GLOBAL':
            return dict(base=float(F.softplus(self.base_raw).detach()),
                        geometry=float(F.softplus(self.geometry_raw).detach()))
        if self.method == 'GATE':
            return dict(base=float(F.softplus(self.base_raw).detach()))
        return dict()


def semantic_cross_entropy(logits, targets, active):
    """Exact native distribution with inactive+NEW as one semantic NEW event.

    Existing targets can be set-valued for the diagnostic; primary training
    excludes ambiguous owners before calling this function. Unmatched queries
    have no fabricated label and must also be excluded by the caller.
    """
    allowed = targets.clone()
    allowed |= targets[:, -1, None] & ~active
    if not allowed.any(-1).all():
        raise ValueError('Unlabelled query entered supervised fusion loss')
    return (torch.logsumexp(logits, -1) - torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), -1)).mean()
