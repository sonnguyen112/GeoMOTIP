"""Opt-in full-train motion Transformer; the production V3 path is untouched.

Installed only by tools.school_gpu.motion_transformer_train.  The architecture
matches the frozen TRANSFORMER_DENSE probe, but history is encoded once per
query timestamp/track rather than redundantly for every detection-track pair.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .geometry_branch import GeometryCompatibilityBranch


def relative_target(current, anchor):
    scale = anchor[..., 2:].clamp_min(1e-4)
    return torch.cat(((current[..., :2] - anchor[..., :2]) / scale,
                      torch.log(current[..., 2:].clamp_min(1e-4) / scale)), -1).clamp(-8, 8)


class DenseMotionTransformer(nn.Module):
    """Exactly eight past bbox tokens, 2 layers, d=32, 4 heads, FFN=64."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(6, 32)
        self.lag_embedding = nn.Embedding(8, 32)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0.,
                                       activation='gelu', batch_first=True)
            for _ in range(2)
        ])
        self.score = nn.Sequential(nn.Linear(40, 128), nn.GELU(), nn.Linear(128, 1))

    def encode_history(self, history, valid):
        history = torch.where(valid[..., None], history, torch.zeros_like(history))
        ids = torch.arange(8, device=history.device)
        last = torch.where(valid, ids, -1).max(1).values.clamp_min(0)
        anchor = history[torch.arange(len(history), device=history.device), last]
        scale = anchor[:, 2:].clamp_min(1e-4)
        delta = (history[:, :, :2] - anchor[:, None, :2]) / scale[:, None]
        sizes = torch.log(history[:, :, 2:].clamp_min(1e-4) / scale[:, None])
        lag = torch.arange(8, 0, -1, device=history.device, dtype=history.dtype)
        lag = (lag / 8)[None, :, None].expand(len(history), -1, -1)
        values = torch.cat((delta, sizes, lag, valid[..., None].to(history.dtype)), -1).clamp(-8, 8)
        values = torch.where(valid[..., None], values, torch.zeros_like(values))
        tokens = self.projection(values) + self.lag_embedding(ids)[None]
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        safe_valid = valid.clone()
        safe_valid[~safe_valid.any(1), 0] = True
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=~safe_valid)
        state = torch.where(valid[..., None], tokens, torch.zeros_like(tokens)).sum(1)
        state = state / valid.sum(1, keepdim=True).clamp_min(1)
        return state, anchor

    def forward(self, batch):
        state, anchor = self.encode_history(batch['history'], batch['valid'])
        target = relative_target(batch['current'], anchor)
        return self.score(torch.cat((state, target, anchor), -1)).flatten()


def exact_dense_history(boxes, times, masks, query_times):
    """[B,G,Th,K,4] -> [B,G,Tq,K,8,4], exact chronological lags8..1."""
    if boxes.shape[:-1] != times.shape or times.shape != masks.shape:
        raise ValueError('Historical box/timestamp/mask shapes disagree')
    results, validity = [], []
    for lag in range(8, 0, -1):
        exact = (~masks[:, :, None]) & (times[:, :, None] == query_times[..., None, None] - lag)
        if (exact.sum(-2) > 1).any():
            raise AssertionError('Duplicate observations for an exact timestamp and track slot')
        # where, not multiplication: invalid NaN payloads must be neutral.
        chosen = torch.where(exact[..., None], boxes[:, :, None], 0.).sum(-3)
        results.append(chosen)
        validity.append(exact.any(-2))
    return torch.stack(results, -2), torch.stack(validity, -1)


class TransformerCompatibilityBranch(GeometryCompatibilityBranch):
    """V3 loss/scatter interface, with a dense motion encoder instead of G4."""
    def __init__(self, *args, **kwargs):
        if kwargs.get('input_stats') is not None:
            raise ValueError('G4 normalization must not be applied to Transformer bbox tokens')
        super().__init__(*args, **kwargs)
        del self.mlp
        self.motion = DenseMotionTransformer()
        self.history_chunk = 256
        self.pair_chunk = 4096

    def _encode_chunk(self, history, valid):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.motion.encode_history, history, valid, use_reentrant=False)
        return self.motion.encode_history(history, valid)

    def _score_chunk(self, features):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.motion.score, features, use_reentrant=False)
        return self.motion.score(features)

    def forward(self, seq):
        current = seq['unknown_boxes']
        current_times = seq.get('unknown_frame_times', seq['unknown_times'])
        history_times = seq.get('trajectory_frame_times', seq['trajectory_times'])
        masks = seq['unknown_masks']
        if not current.shape[-2] or not seq['trajectory_boxes'].shape[-2]:
            raise ValueError('Empty decoder inputs must use IDDecoder._empty_output')
        qtimes = current_times[..., 0]
        if ((current_times != qtimes[..., None]) & ~masks).any():
            raise AssertionError('Detections in one MOTIP time slice must share a frame timestamp')
        history, valid = exact_dense_history(seq['trajectory_boxes'], history_times,
                                            seq['trajectory_masks'], qtimes)
        if not torch.isfinite(current[~masks]).all() or not torch.isfinite(history).all():
            raise FloatingPointError('Nonfinite motion boxes')
        flat_h, flat_v = history.reshape(-1, 8, 4), valid.reshape(-1, 8)
        states, anchors = zip(*(self._encode_chunk(flat_h[i:i+self.history_chunk],
                                                   flat_v[i:i+self.history_chunk])
                                for i in range(0, len(flat_h), self.history_chunk)))
        state = torch.cat(states).reshape(*valid.shape[:-1], 32)
        anchor = torch.cat(anchors).reshape(*valid.shape[:-1], 4)
        # [B,G,T,Q,K,*], encoder state depends only on historical boxes.
        target = relative_target(current[..., None, :], anchor.unsqueeze(-3))
        shape = target.shape[:-1]
        features = torch.cat((state.unsqueeze(-3).expand(*shape, 32), target,
                              anchor.unsqueeze(-3).expand(*shape, 4)), -1)
        flattened = features.reshape(-1, 40)
        scores = torch.cat([self._score_chunk(flattened[i:i+self.pair_chunk])
                            for i in range(0, len(flattened), self.pair_chunk)]).reshape(shape)
        candidate_ids, active = self._candidate_ids(seq['trajectory_id_labels'], history_times,
            seq['trajectory_masks'], current_times, self.num_id_vocabulary)
        pair_valid = valid.any(-1).unsqueeze(-2) & active & ~masks[..., None]
        scores = scores.masked_fill(~pair_valid, 0.)
        logits = self._scatter_logits(scores, pair_valid, candidate_ids, self.num_id_vocabulary)
        labels = seq.get('unknown_id_labels')
        existing = torch.zeros_like(masks)
        target_candidate = torch.zeros_like(masks)
        if labels is not None:
            existing = ~masks & (labels >= 0) & (labels < self.num_id_vocabulary)
            target_candidate = ((candidate_ids == labels[..., None]) & pair_valid).any(-1)
        supervised = existing & target_candidate
        self.last_geometry_loss = (F.cross_entropy(logits[supervised], labels[supervised])
                                   if supervised.any() else logits.sum() * 0.)
        self.last_pair_scores, self.last_pair_valid = scores.detach(), pair_valid.detach()
        self.last_geometry_logits = logits
        self.last_stats = {
            'geometry_feature_abs_max': float(features.detach().abs().max()),
            'geometry_raw_feature_abs_max': float(history.detach().abs().max()),
            'geometry_target_candidate_recall': float(target_candidate[existing].float().mean()) if existing.any() else 0.,
            'geometry_existing_query_fraction': float(existing.float().mean()),
            'geometry_newborn_query_fraction': float(((labels == self.num_id_vocabulary) & ~masks).float().mean()) if labels is not None else 0.,
            'motion_dense_valid_fraction': float(valid.float().mean()),
        }
        return logits, scores, pair_valid


def install_motion_transformer(config):
    """Process-local opt-in; never patch an existing V3 job or saved source."""
    if not config.get('USE_MOTION_TRANSFORMER', False):
        raise ValueError('Dedicated entry point requires USE_MOTION_TRANSFORMER=true')
    if not config.get('USE_GEOMETRY') or config.get('USE_TRAIL') or config.get('USE_SPATIAL_ANCHOR'):
        raise ValueError('Motion Transformer requires only the candidate residual branch')
    if config.get('GEOMETRY_INPUT_STATS_PATH'):
        raise ValueError('Transformer must not load G4 input statistics')
    import models.motip.id_decoder as module
    module.GeometryCompatibilityBranch = TransformerCompatibilityBranch
