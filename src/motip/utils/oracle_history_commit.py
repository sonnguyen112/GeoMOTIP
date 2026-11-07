"""Non-deployable GT intervention for a frozen-tracker causal diagnostic.

GT may select a write AFTER the current prediction. It never changes the
current output, supplies boxes/features, or determines an association target.
Owners are keyed by global ID, not reusable local vocabulary slots.
"""
from collections import Counter
import hashlib

import torch


METHODS = ("baseline", "oracle_first_owner", "oracle_past_majority",
           "confidence_matched", "random_matched")


class OracleHistoryCommitPolicy:
    def __init__(self, method, matcher, sequence, budgets=None, seed=42):
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.matcher, self.sequence = method, matcher, sequence
        self.budgets, self.seed = budgets or {}, int(seed)
        self.reset_sequence()

    def reset_sequence(self):
        self.first_owner = {}
        self.observation_owner = {}
        self.rows, self.frame_rows = [], []
        self.gt_frame, self.gt_timestamp = None, None
        self.last_frame = 0

    def set_frame(self, frame, gt_frame):
        if int(frame) <= self.last_frame:
            raise ValueError("Frames must advance strictly")
        self.gt_timestamp, self.gt_frame = int(frame), gt_frame

    def __call__(self, *, tracker, boxes, id_labels, id_confidences,
                 is_newborn, frame_index, **unused):
        frame = int(frame_index)
        assert frame == self.gt_timestamp and frame > self.last_frame
        result = tracker.current_track_results
        # This hook runs only after assignment and emitted results exist.
        ids = result["id"].detach().cpu().tolist()
        slots = id_labels.detach().cpu().tolist()
        newborn = is_newborn.detach().cpu().tolist()
        confidences = id_confidences.detach().cpu().tolist()
        assert len(ids) == len(slots) == len(boxes)
        assert len(set(ids)) == len(ids) and len(set(slots)) == len(slots)
        matched = self.matcher(self.gt_frame, result, threshold=0.5)
        history_slots = tracker.trajectory_id_labels.detach().cpu()
        times = tracker.trajectory_frame_times.detach().cpu()
        masks = tracker.trajectory_masks.detach().cpu()
        assert bool((times[~masks] < frame).all()), "Future/current history"
        prior = {}
        if history_slots.shape[0]:
            for col, slot in enumerate(history_slots[-1].tolist()):
                stamps = times[~masks[:, col], col].tolist()
                gid = int(tracker.id_label_to_id[int(slot)])
                owners = [self.observation_owner.get((gid, int(t)), -1) for t in stamps]
                known = [x for x in owners if x >= 0]
                count = Counter(known)
                winners = [x for x, n in count.items() if n == max(count.values())] if count else []
                prior[int(slot)] = (len(stamps), winners[0] if len(winners) == 1 else -1)

        records = []
        for i, (gid, slot, born, conf) in enumerate(zip(ids, slots, newborn, confidences)):
            owner = int(matched.get(i, (-1, 0))[0])
            first = self.first_owner.get(gid, -1)
            history_count, majority = prior.get(slot, (0, -1))
            existing = not born and history_count > 0
            eligible = existing and first >= 0 and owner >= 0
            records.append(dict(sequence=self.sequence, frame=frame, global_id=gid, slot=slot,
                matched_gt=owner, first_owner=first, prior_majority=majority,
                is_newborn=bool(born), prior_observations=history_count,
                eligible=eligible, id_confidence=float(conf),
                wrong_first=eligible and owner != first,
                wrong_majority=existing and majority >= 0 and owner >= 0 and owner != majority,
                suppressed=False))

        selected = []
        requested = 0
        if self.method in {"oracle_first_owner", "oracle_past_majority"}:
            field = "wrong_first" if self.method == "oracle_first_owner" else "wrong_majority"
            selected = [i for i, row in enumerate(records) if row[field]]
            requested = len(selected)
        elif self.method in {"confidence_matched", "random_matched"}:
            requested = int(self.budgets.get(frame, 0))
            eligible = [i for i, row in enumerate(records) if row["eligible"]]
            if self.method == "confidence_matched":
                key = lambda i: (records[i]["id_confidence"], records[i]["global_id"])
            else:
                key = lambda i: hashlib.sha256(
                    f"{self.seed}:{self.sequence}:{frame}:{records[i]['global_id']}".encode()).hexdigest()
            selected = sorted(eligible, key=key)[:requested]

        keep = torch.ones(len(records), dtype=torch.bool, device=boxes.device)
        for i in selected:
            assert not records[i]["is_newborn"]
            records[i]["suppressed"] = True
            keep[i] = False
        for row in records:
            gid, owner = row["global_id"], row["matched_gt"]
            # Register only after judging the current write. Never relabel an
            # existing global ID to make a switch appear correct.
            if owner >= 0:
                self.first_owner.setdefault(gid, owner)
            self.observation_owner[(gid, frame)] = owner
        self.rows.extend(records)
        self.frame_rows.append(dict(frame=frame, requested=requested, suppressed=len(selected),
                                    budget_shortfall=max(0, requested-len(selected))))
        self.last_frame = frame
        return keep
