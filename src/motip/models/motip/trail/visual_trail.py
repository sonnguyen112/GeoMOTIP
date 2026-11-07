import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .trail_gate import TrailGate
from .trail_encoder import TrailEncoder
from .trail_utils import canonical_slots, check_finite, extract_roi, splat_roi
from .trail_memory_manager import TrailMemoryManager, TrailMemoryState


# Backward-compatible public alias.  Existing callers and V1 tests use
# ``TrailState``; V2 extends it with pending confirmation state and counters.
TrailState = TrailMemoryState


class VisualTrail(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = int(config.get("TRAIL_CHANNELS", 16))
        self.height = int(config.get("TRAIL_HEIGHT", 12))
        self.width = int(config.get("TRAIL_WIDTH", 20))
        self.roi_size = tuple(config.get("TRAIL_ROI_SIZE", [8, 4]))
        self.feature_dim = config["FEATURE_DIM"]
        self.vocabulary = config["NUM_ID_VOCABULARY"]
        self.detach_history = config.get("TRAIL_DETACH_HISTORY", True)
        self.checkpoint = config.get("GRADIENT_CHECKPOINTING", False)
        self.check_numerics = config.get("TRAIL_CHECK_FINITE", True)
        self.drop = float(config.get("TRAIL_RANDOM_DROP", 0))
        self.jitter = float(config.get("TRAIL_SPATIAL_JITTER", 0))
        self.switch = float(config.get("TRAIL_ID_SWITCH", 0))
        self.memory_mode = str(config.get("TRAIL_MEMORY_MODE", "naive")).lower()
        self.commit_threshold = float(config.get("TRAIL_COMMIT_THRESHOLD", 0.7))
        self.confirm_frames = int(config.get("TRAIL_CONFIRM_FRAMES", 2))
        decays = config.get("TRAIL_DECAYS", [.3, .7, .95])
        if not decays or any(not 0 <= x < 1 for x in decays):
            raise ValueError("TRAIL_DECAYS must be nonempty values in [0,1)")
        if min(self.channels, self.height, self.width, *self.roi_size) <= 0:
            raise ValueError("Trail dimensions must be positive")
        if not all(0 <= x <= 1 for x in (self.drop, self.switch)) or self.jitter < 0:
            raise ValueError("Invalid trail augmentation probability/jitter")
        self.register_buffer("decays", torch.tensor(decays).view(1, 1, -1, 1, 1, 1))
        self.memory_manager = TrailMemoryManager(
            self.decays[0, 0], mode=self.memory_mode,
            commit_threshold=self.commit_threshold,
            confirm_frames=self.confirm_frames,
        )
        self.source_projection = nn.Conv2d(config["DETR_HIDDEN_DIM"], self.channels, 1)
        self.gate = TrailGate(config["DETR_HIDDEN_DIM"]) if config.get("TRAIL_USE_GATE", True) else None
        self.encoder = TrailEncoder(self.channels, len(decays), self.feature_dim,
                                    config.get("TRAIL_NUM_TOKENS", 8))
        self.stats = {}
        # Optional inference probe; disabled for normal training/evaluation.
        self.collect_feature_diagnostics = False
        self.feature_diagnostic_rows = []
        self.memory_stats = {}
        # Runtime diagnostics: status for each attempted slot write in the
        # most recent frame.  This is intentionally separate from the
        # cumulative counters so ownership precision can be audited without
        # changing the inference decision.
        self.last_write_statuses = {}

    def extract(self, source, boxes, valid_ratios):
        if self.detach_history:
            source, boxes = source.detach(), boxes.detach()
        projected = self.source_projection(source.to(self.source_projection.weight.dtype))
        return extract_roi(projected, boxes, valid_ratios, self.roi_size)

    def observations(self, rois, boxes, features, scores, invalid):
        shape = boxes.shape[:-1]
        if self.detach_history:
            boxes, features, scores = boxes.detach(), features.detach(), scores.detach()
        if self.training and self.jitter:
            boxes = boxes.clone()
            boxes[..., :2] = (boxes[..., :2] + torch.randn_like(boxes[..., :2]) * self.jitter).clamp(0, 1)
        gate = self.gate(features.to(self.source_projection.weight.dtype), scores.to(self.source_projection.weight.dtype)) \
            if self.gate is not None else torch.ones_like(scores)
        valid = ~invalid
        if self.training and self.drop:
            valid = valid & (torch.rand_like(scores) >= self.drop)
        stamps = splat_roi(rois.reshape(-1, self.channels, *self.roi_size), boxes.reshape(-1, 4), self.height, self.width)
        stamps = stamps.reshape(*shape, self.channels, self.height, self.width)
        if getattr(self, "collect_feature_diagnostics", False):
            with torch.no_grad():
                roi_energy = rois.float().abs().mean(dim=tuple(range(1, rois.ndim)))
                canvas_energy = stamps.float().abs().mean(dim=-3)
                support = (canvas_energy > 1e-6).float().mean(dim=(-2, -1))
                area = (boxes[..., 2].float().clamp_min(0) *
                        boxes[..., 3].float().clamp_min(0))
                self.feature_diagnostic_rows.append({
                    "observations": int(roi_energy.numel()),
                    "roi_abs_mean": float(roi_energy.mean().item()) if roi_energy.numel() else 0.0,
                    "roi_abs_std": float(roi_energy.std(unbiased=False).item()) if roi_energy.numel() else 0.0,
                    "splat_support_fraction": float(support.mean().item()) if support.numel() else 0.0,
                    "box_area_fraction": float(area.mean().item()) if area.numel() else 0.0,
                    "valid_fraction": float(valid.float().mean().item()) if valid.numel() else 0.0,
                })
        return stamps * (gate * valid)[..., None, None, None], gate

    def encode(self, state):
        if self.training and self.checkpoint and torch.is_grad_enabled():
            return checkpoint(self.encoder, state, use_reentrant=False)
        return self.encoder(state)

    def forward(self, seq):
        """Build strictly causal prefix tokens for each query time.

        Input history [B,G,T,N,...], ROI [B,G,T,N,C,8,4]. Output
        tokens [B,G,Tq,K,P,F], IDs/masks [B,G,Tq,K]. K is compact assigned
        dictionary slots, not raw dataset IDs. Recurrence graphs span only the
        sampled clip; TRAIL_DETACH_HISTORY detaches DETR, NOT trainable updates.
        """
        b, g, t, n = seq["trajectory_masks"].shape
        bg = b * g
        masks = seq["trajectory_masks"].reshape(bg, t, n)
        labels = seq["trajectory_id_labels"].reshape(bg, t, n)
        slots, mapping = canonical_slots(labels, masks, self.vocabulary)
        k = slots.shape[-1]
        state = seq["trajectory_boxes"].new_zeros(bg, k, self.decays.numel(), self.channels, self.height, self.width).float()
        seen = torch.zeros((bg, k), device=state.device, dtype=torch.bool)
        # MOTIP uses clip-relative frame times, equally across groups/objects.
        tq = seq["unknown_features"].shape[2]
        query_times = seq["unknown_times"].reshape(bg, tq, -1)
        source_times = seq["trajectory_times"].reshape(bg, t, n)
        tokens, token_masks, gates = [], [], []
        pending = [[[] for _ in range(k)] for _ in range(bg)]
        pending_frames = [[[] for _ in range(k)] for _ in range(bg)]
        pending_gates = [[[] for _ in range(k)] for _ in range(bg)]
        memory_stats = {
            "trail_write_attempts": 0,
            "trail_write_committed": 0,
            "trail_write_rejected_low_confidence": 0,
            "trail_write_rejected_identity_change": 0,
            "temporary_state_resets": 0,
            "confirmation_events": 0,
        }
        source_idx = 0
        for query_idx in range(tq):
            current = query_times[:, query_idx].max(-1).values
            # Training: 0..T-1; runtime uses an explicit persistent state path.
            while source_idx < t:
                times = source_times[:, source_idx].max(-1).values
                if not (times < current).all():
                    break
                valid = ~masks[:, source_idx]
                obs, gate = self.observations(
                    seq["trajectory_rois"][:, :, source_idx].reshape(bg, n, self.channels, *self.roi_size),
                    seq["trajectory_boxes"][:, :, source_idx].reshape(bg, n, 4),
                    seq["trajectory_features"][:, :, source_idx].reshape(bg, n, -1),
                    seq["trajectory_scores"][:, :, source_idx].reshape(bg, n), masks[:, source_idx])
                idx = mapping[:, source_idx]
                if self.training and self.switch and n > 1:
                    # Corrupt only trail observations; assigned MOTIP IDs stay intact.
                    swap = torch.rand(bg, device=state.device) < self.switch
                    obs = torch.where(swap[:, None, None, None, None], obs.roll(1, 1), obs)
                # Advance persistent memory every source frame.  In confirmed
                # mode uncommitted observations live only in the local pending
                # lists and cannot affect ``state``.
                state = state * self.decays
                update = torch.zeros_like(state)
                committed_now = torch.zeros_like(seen)
                scores_all = seq.get("trajectory_scores")
                if scores_all is None:
                    score_frame = torch.ones_like(gate)
                else:
                    score_frame = scores_all.reshape(bg, t, n)[:, source_idx]
                for batch_idx in range(bg):
                    observed_slots = set()
                    for object_idx in range(n):
                        if not bool(valid[batch_idx, object_idx]):
                            continue
                        slot = int(idx[batch_idx, object_idx].item())
                        observed_slots.add(slot)
                        confidence = float(score_frame[batch_idx, object_idx].detach().item())
                        stamp = obs[batch_idx, object_idx]
                        gate_value = float(gate[batch_idx, object_idx].detach().item())
                        memory_stats["trail_write_attempts"] += 1
                        if self.memory_mode == "naive":
                            update[batch_idx, slot] = update[batch_idx, slot] + stamp
                            memory_stats["trail_write_committed"] += 1
                            continue
                        if confidence < self.commit_threshold:
                            if pending[batch_idx][slot]:
                                pending[batch_idx][slot].clear()
                                pending_frames[batch_idx][slot].clear()
                                pending_gates[batch_idx][slot].clear()
                                memory_stats["temporary_state_resets"] += 1
                            memory_stats["trail_write_rejected_low_confidence"] += 1
                            continue
                        if pending_frames[batch_idx][slot] and pending_frames[batch_idx][slot][-1] != source_idx - 1:
                            pending[batch_idx][slot].clear()
                            pending_frames[batch_idx][slot].clear()
                            pending_gates[batch_idx][slot].clear()
                            memory_stats["temporary_state_resets"] += 1
                            memory_stats["trail_write_rejected_identity_change"] += 1
                        pending[batch_idx][slot].append(stamp)
                        pending_frames[batch_idx][slot].append(source_idx)
                        pending_gates[batch_idx][slot].append(gate_value)
                        if len(pending[batch_idx][slot]) >= self.confirm_frames:
                            for delayed_stamp, delayed_frame in zip(
                                    pending[batch_idx][slot], pending_frames[batch_idx][slot]):
                                age = source_idx - delayed_frame
                                decay = self.decays[0, 0].to(
                                    device=delayed_stamp.device, dtype=delayed_stamp.dtype
                                ).pow(age).reshape(-1, 1, 1, 1)
                                update[batch_idx, slot] = update[batch_idx, slot] + delayed_stamp * decay
                            memory_stats["trail_write_committed"] += len(pending[batch_idx][slot])
                            memory_stats["confirmation_events"] += 1
                            committed_now[batch_idx, slot] = True
                            pending[batch_idx][slot].clear()
                            pending_frames[batch_idx][slot].clear()
                            pending_gates[batch_idx][slot].clear()
                    if self.memory_mode == "confirmed":
                        for slot in range(k):
                            if slot not in observed_slots and pending[batch_idx][slot]:
                                pending[batch_idx][slot].clear()
                                pending_frames[batch_idx][slot].clear()
                                pending_gates[batch_idx][slot].clear()
                                memory_stats["temporary_state_resets"] += 1
                state = state + update
                counts = torch.zeros_like(seen, dtype=torch.long).scatter_add(1, idx, valid.long())
                if self.memory_mode == "naive":
                    seen = seen | (counts > 0)
                else:
                    seen = seen | committed_now
                gates.append(gate[valid])
                source_idx += 1
            tokens.append(self.encode(state))
            token_masks.append(~seen)
        result = torch.stack(tokens, 1).reshape(b, g, tq, k, self.encoder.num_tokens, self.feature_dim)
        self.memory_stats = dict(memory_stats)
        self.memory_stats["pending_trails"] = sum(bool(x) for rows in pending for x in rows)
        self.record_stats(state, torch.cat(gates) if gates else state.new_empty(0), result, seen)
        seq["trail_tokens"] = result
        seq["trail_slot_ids"] = slots.reshape(b, g, 1, k).expand(b, g, tq, k)
        seq["trail_masks"] = torch.stack(token_masks, 1).reshape(b, g, tq, k)
        return seq

    def record_stats(self, state, gates, tokens, seen=None):
        if self.check_numerics:
            check_finite("trail memory", state)
            check_finite("trail tokens", tokens)
            check_finite("trail gate", gates)
        with torch.no_grad():
            self.stats = {
                "trail_magnitude": state.abs().mean().item() if state.numel() else 0.,
                "trail_token_norm": tokens.float().norm(dim=-1).mean().item() if tokens.numel() else 0.,
                "trail_active": seen.sum(-1).float().mean().item() if seen is not None else float(len(state)),
                "trail_gate_mean": gates.float().mean().item() if gates.numel() else 0.,
                "trail_gate_std": gates.float().std(unbiased=False).item() if gates.numel() else 0.,
                "trail_gate_min": gates.min().item() if gates.numel() else 0.,
                "trail_gate_max": gates.max().item() if gates.numel() else 0.,
            }

    @torch.no_grad()
    def update_runtime(self, bank, ids, rois, boxes, features, scores,
                       frame_index: int | None = None, identity_confidences=None):
        ids_list = ids.tolist()
        if len(set(ids_list)) != len(ids_list) or any(x < 0 or x >= self.vocabulary for x in ids_list):
            raise ValueError("Runtime trail update requires unique active MOTIP slots")
        obs, gates = self.observations(rois, boxes, features, scores, torch.zeros_like(ids, dtype=torch.bool))
        if identity_confidences is None:
            identity_confidences = scores
        if frame_index is None:
            frame_index = 1 if not getattr(bank, "counters", None).get("_last_frame", 0) else int(bank.counters["_last_frame"]) + 1
        frame_index = int(frame_index)
        bank.counters["_last_frame"] = frame_index
        self.memory_manager.decay(bank)
        self.memory_manager.begin_frame(bank, frame_index, ids.tolist())
        self.last_write_statuses = {}
        for i, slot in enumerate(ids_list):
            self.last_write_statuses[int(slot)] = self.memory_manager.update(
                bank, slot, obs[i], float(gates[i]), float(identity_confidences[i]), frame_index,
            )
        if self.check_numerics and bank.memories:
            check_finite("runtime trail", torch.stack(list(bank.memories.values())))
        self.memory_stats = self.memory_manager.snapshot_counters(bank)

    def runtime_context(self, bank, device):
        ids = sorted(bank.memories)
        if not ids:
            return None
        state = torch.stack([bank.memories[i] for i in ids])
        tokens = self.encode(state)
        self.record_stats(state, state.new_tensor([bank.gates[i] for i in ids]), tokens)
        return {"trail_tokens": tokens[None, None, None],
                "trail_slot_ids": torch.tensor(ids, device=device)[None, None, None],
                "trail_masks": torch.zeros((1, 1, 1, len(ids)), dtype=torch.bool, device=device)}
