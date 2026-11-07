"""Confirmed-write trail memory lifecycle.

The V1 runtime bank wrote every predicted observation directly to the
persistent memory.  This module keeps that legacy path available, but adds a
small pending buffer for V2.  The pending buffer is deliberately not a
second long-lived trail: it is discarded when confidence drops, the predicted
slot changes, or the slot is released.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch


@dataclass
class PendingObservation:
    """A short candidate sequence waiting for identity confirmation."""

    stamps: list[torch.Tensor] = field(default_factory=list)
    gates: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.stamps)


def _new_counters():
    return defaultdict(int)


@dataclass
class TrailMemoryState:
    """Persistent and temporary state for one runtime tracker.

    ``memories`` remains the public persistent-bank field used by V1.  The
    additional fields are intentionally plain Python containers so slot
    release/recycle is explicit and cheap on the inference path.
    """

    memories: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    pending: dict[int, PendingObservation] = field(default_factory=dict)
    counters: defaultdict = field(default_factory=_new_counters)

    @classmethod
    def empty(cls):
        return cls()

    def release(self, slots):
        for slot in slots:
            slot = int(slot)
            self.memories.pop(slot, None)
            self.gates.pop(slot, None)
            self.pending.pop(slot, None)

    def retain(self, slots):
        keep = set(map(int, slots))
        self.release(set(self.memories) - keep)
        self.release(set(self.pending) - keep)

    def reset_pending(self, slot, reason: str | None = None):
        slot = int(slot)
        if slot in self.pending:
            self.pending.pop(slot, None)
            self.counters["temporary_state_resets"] += 1
            if reason:
                self.counters[f"temporary_state_resets_{reason}"] += 1


class TrailMemoryManager:
    """Apply naive or confirmed writes to a persistent multi-decay bank."""

    def __init__(self, decays, mode: str = "naive", commit_threshold: float = 0.7,
                 confirm_frames: int = 2):
        mode = str(mode).lower()
        if mode not in {"naive", "confirmed"}:
            raise ValueError(f"Unknown trail memory mode: {mode}")
        if not 0.0 <= float(commit_threshold) <= 1.0:
            raise ValueError("TRAIL_COMMIT_THRESHOLD must be in [0, 1]")
        if int(confirm_frames) <= 0:
            raise ValueError("TRAIL_CONFIRM_FRAMES must be positive")
        self.mode = mode
        self.commit_threshold = float(commit_threshold)
        self.confirm_frames = int(confirm_frames)
        self.decays = decays

    def _decay_stamp(self, stamp: torch.Tensor, age: int) -> torch.Tensor:
        # A stamp is [C,H,W], while a memory is [R,C,H,W].  Applying the
        # elapsed decay to delayed pending observations makes the confirmation
        # commit equivalent to having written them on their original frames.
        powers = self.decays.to(device=stamp.device, dtype=stamp.dtype).pow(int(age))
        return stamp[None] * powers.reshape(-1, 1, 1, 1)

    def decay(self, state: TrailMemoryState):
        """Advance all persistent memories by one frame."""
        if not state.memories:
            return
        decays = self.decays
        for slot in list(state.memories):
            memory = state.memories[slot]
            state.memories[slot] = memory * decays.to(device=memory.device, dtype=memory.dtype).reshape(-1, 1, 1, 1)

    def begin_frame(self, state: TrailMemoryState, frame_index: int, observed_slots):
        """Expire pending candidates whose track was not observed this frame."""
        observed = set(int(x) for x in observed_slots)
        for slot, pending in list(state.pending.items()):
            if slot not in observed or not pending.frames or pending.frames[-1] != int(frame_index) - 1:
                state.reset_pending(slot, reason="missing_or_gap")

    def _ensure_memory(self, state: TrailMemoryState, slot: int, stamp: torch.Tensor):
        if slot not in state.memories:
            scales = self.decays.numel()
            state.memories[slot] = stamp[None].expand(scales, -1, -1, -1).clone()
            # The expanded zero/current buffer is immediately overwritten by
            # the caller's contribution.  Zero it here so a confirmed write
            # never receives an accidental duplicated current stamp.
            state.memories[slot].zero_()

    def _commit(self, state: TrailMemoryState, slot: int, pending: PendingObservation,
                frame_index: int):
        self._ensure_memory(state, slot, pending.stamps[-1])
        contribution = state.memories[slot].new_zeros(state.memories[slot].shape)
        for stamp, frame in zip(pending.stamps, pending.frames):
            contribution = contribution + self._decay_stamp(stamp, int(frame_index) - int(frame))
        state.memories[slot] = state.memories[slot] + contribution
        if pending.gates:
            state.gates[slot] = float(pending.gates[-1])
        state.counters["trail_write_committed"] += pending.count
        state.counters["confirmation_events"] += 1
        return pending.count

    def update(self, state: TrailMemoryState, slot: int, stamp: torch.Tensor, gate: float,
               confidence: float, frame_index: int):
        """Attempt one write and return a status string.

        ``stamp`` is already gate-weighted by ``VisualTrail.observations``.
        Confidence is used only for the confirmation policy; it never changes
        the prediction returned by MOTIP.
        """
        slot = int(slot)
        confidence = float(confidence)
        frame_index = int(frame_index)
        state.counters["trail_write_attempts"] += 1

        if self.mode == "naive":
            self._ensure_memory(state, slot, stamp)
            state.memories[slot] = state.memories[slot] + stamp[None]
            state.gates[slot] = float(gate)
            state.counters["trail_write_committed"] += 1
            return "committed"

        if confidence < self.commit_threshold:
            if slot in state.pending:
                state.reset_pending(slot, reason="low_confidence")
            state.counters["trail_write_rejected_low_confidence"] += 1
            return "rejected_low_confidence"

        pending = state.pending.get(slot)
        if pending is None or not pending.frames or pending.frames[-1] != frame_index - 1:
            if pending is not None:
                state.reset_pending(slot, reason="identity_change")
            pending = PendingObservation()
            state.pending[slot] = pending
        pending.stamps.append(stamp)
        pending.gates.append(float(gate))
        pending.confidences.append(confidence)
        pending.frames.append(frame_index)

        if pending.count < self.confirm_frames:
            state.counters["trail_write_pending"] += 1
            return "pending"

        self._commit(state, slot, pending, frame_index)
        state.pending.pop(slot, None)
        return "committed"

    def snapshot_counters(self, state: TrailMemoryState) -> dict[str, int]:
        result = {str(k): int(v) for k, v in state.counters.items()}
        result["pending_trails"] = len(state.pending)
        result["active_persistent_trails"] = len(state.memories)
        return result

