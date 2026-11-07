# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh
from models.misc import get_model
from models.motip.trail import TrailState
from tools.appearance_diagnostics import candidate_appearance_cosines


class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
            trail_diagnostic_mode: str = "normal",
            trail_shuffle_seed: int = 42,
            trail_token_limit: int | None = None,
            trail_context_scale: float = 1.0,
            history_commit_policy=None,
            save_candidate_appearance_diagnostics: bool = False,
            tracklet_confirm_n_init: int = 1,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.trail_diagnostic_mode = str(trail_diagnostic_mode).lower()
        if self.trail_diagnostic_mode not in {"normal", "off", "shuffled", "oracle_update"}:
            raise ValueError(f"Unsupported trail diagnostic mode: {trail_diagnostic_mode}")
        self.trail_shuffle_seed = int(trail_shuffle_seed)
        self.trail_token_limit = trail_token_limit
        self.trail_context_scale = float(trail_context_scale)
        # Optional inference-time policy controlling whether an assigned
        # observation is committed to future trajectory history.  This is
        # deliberately separate from the current-frame association/output:
        # the normal path remains unchanged when no policy is supplied.
        self.history_commit_policy = history_commit_policy
        self.save_candidate_appearance_diagnostics = bool(save_candidate_appearance_diagnostics)
        self.tracklet_confirm_n_init = int(tracklet_confirm_n_init)
        if self.tracklet_confirm_n_init < 1:
            raise ValueError("tracklet_confirm_n_init must be >= 1")
        # Optional MOTIP2-style tentative-track lifecycle. The default of one
        # preserves the original MOTIP path exactly. Pending detections are
        # kept in model history so the next frame can confirm the same slot,
        # but are withheld from output until the required consecutive run.
        self._tentative_tracklets = {}
        self.tracklet_confirmation_stats = {
            "created": 0, "confirmed": 0, "discarded": 0,
        }
        if self.history_commit_policy is not None and hasattr(self.history_commit_policy, "reset_sequence"):
            self.history_commit_policy.reset_sequence()
        if self.trail_token_limit is not None and int(self.trail_token_limit) <= 0:
            raise ValueError("trail_token_limit must be positive when supplied")
        if self.trail_context_scale < 0:
            raise ValueError("trail_context_scale must be non-negative")
        self.num_id_vocabulary = get_model(model).num_id_vocabulary
        self.visual_trail = getattr(get_model(model), "visual_trail", None)
        self.trail_state = TrailState.empty() if self.visual_trail is not None else None
        # These fields are used only by evaluation diagnostics. They are kept
        # outside the model so every sequence starts with fresh memory and the
        # normal path remains unchanged.
        self.oracle_gt_to_slot = {}
        self.oracle_slot_to_gt = {}
        self.audit_rows = []
        self.last_trail_update = {}
        self.last_trail_write_statuses = {}
        self.last_id_pred_confidences = torch.empty(0, device=distributed_device(), dtype=dtype)
        self.last_geometry_diagnostics = {}
        self.last_geometry_logits = None
        self.last_base_logits = None
        self.last_fused_logits = None
        self.last_geometry_candidate_ids = None
        self.last_geometry_candidate_active = None
        self.last_geometry_candidate_valid = None
        self.last_geometry_candidate_lag_valid = None
        self._frame_index = 0
        decoder = getattr(get_model(model), "id_decoder", None)
        if decoder is not None:
            decoder.trail_context_scale = self.trail_context_scale

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        self.id_queue = OrderedSet()
        # Init id_queue:
        for i in range(self.num_id_vocabulary):
            self.id_queue.add(i)
        # All fields are in shape (T, N, ...)
        self.trajectory_features = torch.zeros(
            (0, 0, 256), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=dtype, device=distributed_device(),
        )
        # Absolute frame numbers for geometry matching.  ``trajectory_times``
        # is intentionally reindexed after every update for MOTIP's local
        # causal positional encoding, so it must not be reused for exact
        # temporal lags.
        self.trajectory_frame_times = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )
        # self.trajectory_features = torch.zeros(())

        self.current_track_results = {}
        return

    @torch.no_grad()
    def update(self, image, frame_index: int | None = None, gt_frame=None):
        if frame_index is None:
            self._frame_index += 1
        else:
            self._frame_index = int(frame_index)
        if getattr(self, "tracklet_confirm_n_init", 1) > 1:
            self._prune_stale_tentative_tracklets()
        self.last_geometry_logits = None
        self.last_base_logits = None
        self.last_fused_logits = None
        self.last_geometry_candidate_ids = None
        self.last_geometry_candidate_active = None
        self.last_geometry_candidate_valid = None
        self.last_geometry_candidate_lag_valid = None
        self.last_geometry_diagnostics = {}
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds = self._get_activate_detections(detr_out=detr_out)
        rois = None
        if self.visual_trail is not None:
            rois = self.model(part="trail_extract", detr_outputs=detr_out)[0][self._active_detection_indices]
        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
            id_pred_confidences = scores.clone()
        else:
            id_pred_labels = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds)
            id_pred_confidences = self.last_id_pred_confidences.to(device=boxes.device, dtype=scores.dtype)
            if id_pred_confidences.numel() != id_pred_labels.numel():
                id_pred_confidences = scores.clone()
        # Preserve a branch-comparison cohort before the ID-dependent newborn
        # filtering below.  Otherwise a geometry decision can change which
        # detection rows are cached, biasing offline base-vs-geometry metrics.
        candidate_eval_record = None
        if (
            self.last_base_logits is not None
            and self.last_geometry_logits is not None
            and self.last_fused_logits is not None
            and self.last_geometry_candidate_ids is not None
            and self.last_geometry_candidate_active is not None
            and self.last_geometry_candidate_valid is not None
            and self.last_geometry_candidate_lag_valid is not None
        ):
            query_count = int(boxes.shape[0])
            aligned_rows = {
                "base_logits": self.last_base_logits[0, 0, 0].shape[0],
                "geometry_logits": self.last_geometry_logits[0, 0, 0].shape[0],
                "fused_logits": self.last_fused_logits[0, 0, 0].shape[0],
                "candidate_ids": self.last_geometry_candidate_ids[0, 0, 0].shape[0],
                "candidate_active": self.last_geometry_candidate_active[0, 0, 0].shape[0],
                "candidate_valid": self.last_geometry_candidate_valid[0, 0, 0].shape[0],
                "candidate_lag_valid": self.last_geometry_candidate_lag_valid[0, 0, 0].shape[0],
            }
            mismatched = {key: rows for key, rows in aligned_rows.items() if rows != query_count}
            if mismatched:
                raise RuntimeError(
                    "Pre-filter candidate diagnostics are not aligned with detector queries: "
                    f"expected {query_count} rows, got {mismatched}"
                )
            candidate_shapes = (
                tuple(self.last_geometry_candidate_ids[0, 0, 0].shape),
                tuple(self.last_geometry_candidate_active[0, 0, 0].shape),
                tuple(self.last_geometry_candidate_valid[0, 0, 0].shape),
                tuple(self.last_geometry_candidate_lag_valid[0, 0, 0].shape[:-1]),
            )
            if not (
                candidate_shapes[0]
                == candidate_shapes[1]
                == candidate_shapes[2]
                == candidate_shapes[3]
            ):
                raise RuntimeError(
                    "Candidate IDs, active masks, and valid masks must share shape; "
                    f"got {candidate_shapes}"
                )
            candidate_eval_record = {
                "base_logits": self.last_base_logits[0, 0, 0].detach().cpu(),
                "geometry_logits": self.last_geometry_logits[0, 0, 0].detach().cpu(),
                "fused_logits": self.last_fused_logits[0, 0, 0].detach().cpu(),
                "candidate_ids": self.last_geometry_candidate_ids[0, 0, 0].detach().cpu(),
                "candidate_active": self.last_geometry_candidate_active[0, 0, 0].detach().cpu(),
                "candidate_valid": self.last_geometry_candidate_valid[0, 0, 0].detach().cpu(),
                "candidate_lag_valid": self.last_geometry_candidate_lag_valid[0, 0, 0].detach().cpu(),
                "boxes_xywh": (box_cxcywh_to_xywh(boxes) * self.bbox_unnorm).detach().cpu(),
                "detection_scores": scores.detach().cpu(),
                "categories": categories.detach().cpu(),
                "pre_filter_id_prediction": id_pred_labels.detach().cpu(),
                "pre_filter_id_confidence": id_pred_confidences.detach().cpu(),
            }
            candidate_ids_cpu = candidate_eval_record["candidate_ids"]
            candidate_active_cpu = candidate_eval_record["candidate_active"]
            candidate_global_ids = torch.full_like(candidate_ids_cpu, -1)
            candidate_age_frames = torch.zeros_like(candidate_ids_cpu)
            candidate_time_since_last = torch.full_like(candidate_ids_cpu, -1)
            if self.trajectory_id_labels.shape[0] > 0:
                history_labels = self.trajectory_id_labels[-1].detach().cpu().long()
                history_masks = self.trajectory_masks.detach().cpu()
                history_frame_times = self.trajectory_frame_times.detach().cpu().long()
                slot_metadata = {}
                for column, slot_value in enumerate(history_labels.tolist()):
                    observed = (~history_masks[:, column]).nonzero(as_tuple=False).flatten()
                    if not observed.numel():
                        continue
                    observed_frames = history_frame_times[observed, column]
                    first_seen = int(observed_frames.min().item())
                    last_seen = int(observed_frames.max().item())
                    slot = int(slot_value)
                    slot_metadata[slot] = (
                        int(self.id_label_to_id.get(slot, -1)),
                        max(1, self._frame_index - first_seen + 1),
                        max(0, self._frame_index - last_seen),
                    )
                for slot, (global_id, age, since_last) in slot_metadata.items():
                    selected = candidate_active_cpu & (candidate_ids_cpu == slot)
                    candidate_global_ids[selected] = global_id
                    candidate_age_frames[selected] = age
                    candidate_time_since_last[selected] = since_last
            candidate_eval_record["candidate_global_ids"] = candidate_global_ids
            candidate_eval_record["candidate_age_frames"] = candidate_age_frames
            candidate_eval_record["candidate_time_since_last_observation"] = candidate_time_since_last
            if self.save_candidate_appearance_diagnostics:
                appearance_scores, appearance_valid = candidate_appearance_cosines(
                    detection_features=output_embeds,
                    trajectory_features=self.trajectory_features,
                    trajectory_id_labels=self.trajectory_id_labels,
                    trajectory_masks=self.trajectory_masks,
                    candidate_ids=candidate_ids_cpu,
                    candidate_active=candidate_active_cpu,
                )
                for name, values in appearance_scores.items():
                    candidate_eval_record[f"appearance_cosine_{name}"] = values
                candidate_eval_record["appearance_history_valid"] = appearance_valid
        # Filter out illegal newborn detections:
        keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]
        id_pred_confidences = id_pred_confidences[keep_idxs]
        self._filter_last_decoder_logits(keep_idxs)
        if rois is not None:
            rois = rois[keep_idxs]

        # A hack implementation, before assign new id labels, update the id_queue to ensure the uniqueness of id labels:
        n_activate_id_labels = 0
        n_newborn_targets = 0
        for _ in range(len(id_pred_labels)):
            if id_pred_labels[_].item() != self.num_id_vocabulary:
                n_activate_id_labels += 1
                self.id_queue.add(id_pred_labels[_].item())
            else:
                n_newborn_targets += 1

        # Make sure the length of newborn instances is less than the length of remaining IDs:
        n_remaining_ids = len(self.id_queue) - n_activate_id_labels
        if n_newborn_targets > n_remaining_ids:
            keep_idxs = torch.ones(len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device)
            newborn_idxs = (id_pred_labels == self.num_id_vocabulary)
            newborn_keep_idxs = torch.ones(len(newborn_idxs), dtype=torch.bool, device=newborn_idxs.device)
            newborn_keep_idxs[n_remaining_ids:] = False
            keep_idxs[newborn_idxs] = newborn_keep_idxs
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]
            id_pred_confidences = id_pred_confidences[keep_idxs]
            self._filter_last_decoder_logits(keep_idxs)
            if rois is not None:
                rois = rois[keep_idxs]
        pass

        # Assign new id labels:
        memory_slots_before_assignment = set(self.trail_state.memories) if self.trail_state is not None else set()
        # Allocation rewrites pred_id_labels IN PLACE. Capture NEW now so a
        # history-write policy can still protect true first observations.
        history_write_newborn = (
            id_pred_labels == self.num_id_vocabulary
        ) if self.history_commit_policy is not None else None
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels)
        memory_slots_after_assignment = set(self.trail_state.memories) if self.trail_state is not None else set()
        released_slots = sorted(memory_slots_before_assignment - memory_slots_after_assignment)

        if len(torch.unique(id_labels)) != len(id_labels):
            print(id_labels, id_labels.shape)
            exit(-1)

        # Update the results:
        self.current_track_results = {
            "score": scores,
            "category": categories,
            # "bbox": boxes * self.bbox_unnorm,
            "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
            "id": torch.tensor(
                [self.id_label_to_id[_] for _ in id_labels.tolist()], dtype=torch.int64,
            ),
            "id_confidence": id_pred_confidences.detach().cpu(),
            # Expose the stable local MOTIP slot even when the visual-trail
            # branch is disabled.  This is an internal diagnostic field and
            # does not alter TrackEval output; it lets offline probes align
            # detector boxes with candidate-wise geometry scores.
            "slot_id": id_labels.detach().cpu(),
        }
        if self.last_geometry_logits is not None:
            self.current_track_results.update({
                "geometry_logits": self.last_geometry_logits[0, 0, 0].detach().cpu(),
                "base_logits": self.last_base_logits[0, 0, 0].detach().cpu(),
                "fused_logits": self.last_fused_logits[0, 0, 0].detach().cpu(),
                "geometry_diagnostics": dict(self.last_geometry_diagnostics),
            })
            if self.last_geometry_candidate_ids is not None:
                self.current_track_results.update({
                    "geometry_candidate_ids": self.last_geometry_candidate_ids[0, 0, 0].detach().cpu(),
                    "geometry_candidate_active": self.last_geometry_candidate_active[0, 0, 0].detach().cpu(),
                    "geometry_candidate_valid": self.last_geometry_candidate_valid[0, 0, 0].detach().cpu(),
                    "geometry_candidate_lag_valid": self.last_geometry_candidate_lag_valid[0, 0, 0].detach().cpu(),
                })

        if candidate_eval_record is not None:
            # Opt-in branch cache only: normal tracker output and assignment
            # behavior are unchanged.  The cache writer consumes this nested
            # record to evaluate the same detector-thresholded query cohort
            # across base, geometry, and fused scores.
            self.current_track_results["candidate_eval"] = candidate_eval_record

        # Update id_queue:
        for _ in range(len(id_labels)):
            self.id_queue.add(id_labels[_].item())

        # Update trajectory infos:
        if self.history_commit_policy is None:
            # Keep the reference code path exact for baseline and existing
            # experiments unless a write policy is explicitly injected.
            self._update_trajectory_infos(boxes=boxes, output_embeds=output_embeds, id_labels=id_labels)
        else:
            is_newborn = history_write_newborn
            commit_mask = self.history_commit_policy(
                tracker=self,
                boxes=boxes,
                output_embeds=output_embeds,
                id_labels=id_labels,
                scores=scores,
                id_confidences=id_pred_confidences,
                is_newborn=is_newborn,
                frame_index=self._frame_index,
            )
            if not torch.is_tensor(commit_mask):
                commit_mask = torch.as_tensor(commit_mask, device=id_labels.device)
            if commit_mask.ndim != 1 or commit_mask.shape != id_labels.shape:
                raise ValueError(
                    "history_commit_policy must return one boolean per assigned detection; "
                    f"got {tuple(commit_mask.shape)} for {tuple(id_labels.shape)}"
                )
            commit_mask = commit_mask.to(device=id_labels.device, dtype=torch.bool)
            # A newborn must establish its first history entry.  Policies may
            # abstain on existing tracks only; this preserves ID lifecycle.
            commit_mask = commit_mask | is_newborn
            self.current_track_results["history_commit_mask"] = commit_mask.detach().cpu()
            self._update_trajectory_infos(
                boxes=boxes,
                output_embeds=output_embeds,
                id_labels=id_labels,
                commit_mask=commit_mask,
            )

        # Filter out inactive tracks:
        self._filter_out_inactive_tracks()
        if self.visual_trail is not None:
            # Assignment precedes the update: current detections cannot leak
            # into the context used to predict their own IDs.
            trail_ids, trail_indices = self._prepare_trail_update(
                id_labels=id_labels, boxes=boxes, rois=rois,
                output_embeds=output_embeds, scores=scores,
                gt_frame=gt_frame, released_slots=released_slots,
            )
            if self.trail_diagnostic_mode != "off":
                self.visual_trail.update_runtime(
                    self.trail_state, trail_ids, rois[trail_indices],
                    boxes[trail_indices], output_embeds[trail_indices],
                    scores[trail_indices], frame_index=self._frame_index,
                    identity_confidences=id_pred_confidences[trail_indices],
                )
                self.last_trail_write_statuses = dict(self.visual_trail.last_write_statuses)
            else:
                self.last_trail_write_statuses = {}
            self.current_track_results["slot_id"] = id_labels.detach().cpu()
            if self.trail_diagnostic_mode == "off":
                self.current_track_results["gate"] = scores.new_zeros(scores.shape)
            else:
                self.current_track_results["gate"] = scores.new_tensor(
                    [self.trail_state.gates.get(i, 0.0) for i in id_labels.tolist()]
                )
            self.current_track_results["trail_update_slot"] = id_labels.new_full(id_labels.shape, -1)
            if len(trail_ids):
                self.current_track_results["trail_update_slot"][trail_indices] = trail_ids
            self.current_track_results["trail_write_status"] = [
                self.last_trail_write_statuses.get(int(slot), "not_attempted")
                for slot in id_labels.tolist()
            ]
            self.current_track_results["trail_memory_stats"] = dict(self.visual_trail.memory_stats)
        if getattr(self, "tracklet_confirm_n_init", 1) > 1:
            self._apply_tracklet_confirmation(id_labels, id_pred_labels)
            self._discard_unobserved_tentative_tracklets(id_labels.tolist())
        pass
        return

    @staticmethod
    def _tracklet_output_row(result, index):
        """Copy the public per-detection fields needed to backfill a confirmed track."""
        keys = (
            "score", "category", "bbox", "id", "id_confidence", "slot_id",
            "gate", "trail_update_slot", "history_commit_mask",
        )
        row = {}
        for key in keys:
            value = result.get(key)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > index:
                row[key] = value[index:index + 1].detach().clone()
            elif isinstance(value, list) and len(value) > index:
                row[key] = [value[index]]
        return row

    @staticmethod
    def _append_tracklet_output_row(result, row):
        for key, value in row.items():
            current = result.get(key)
            if torch.is_tensor(current):
                result[key] = torch.cat((current, value.to(device=current.device, dtype=current.dtype)), dim=0)
            elif isinstance(current, list):
                result[key] = current + value

    @staticmethod
    def _filter_tracklet_output_rows(result, keep):
        keep_tensor = torch.as_tensor(keep, dtype=torch.bool)
        for key in (
            "score", "category", "bbox", "id", "id_confidence", "slot_id",
            "gate", "trail_update_slot", "history_commit_mask",
        ):
            value = result.get(key)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == len(keep):
                result[key] = value[keep_tensor.to(device=value.device)]
            elif isinstance(value, list) and len(value) == len(keep):
                result[key] = [item for item, selected in zip(value, keep) if selected]

    def _discard_tentative_tracklet(self, slot):
        slot = int(slot)
        state = self._tentative_tracklets.pop(slot, None)
        if state is None:
            return
        if self.trajectory_id_labels.numel():
            keep = ~(self.trajectory_id_labels == slot).any(dim=0)
            self.trajectory_features = self.trajectory_features[:, keep]
            self.trajectory_boxes = self.trajectory_boxes[:, keep]
            self.trajectory_id_labels = self.trajectory_id_labels[:, keep]
            self.trajectory_times = self.trajectory_times[:, keep]
            self.trajectory_frame_times = self.trajectory_frame_times[:, keep]
            self.trajectory_masks = self.trajectory_masks[:, keep]
        self.id_label_to_id.pop(slot, None)
        if self.trail_state is not None:
            self.trail_state.release([slot])
        # OrderedSet is an LRU-like allocation order in this tracker; moving
        # the released slot to the tail makes it available for later births.
        self.id_queue.add(slot)
        self.tracklet_confirmation_stats["discarded"] += 1

    def _prune_stale_tentative_tracklets(self):
        for slot, state in list(self._tentative_tracklets.items()):
            if int(state["last_frame"]) != self._frame_index - 1:
                self._discard_tentative_tracklet(slot)

    def _apply_tracklet_confirmation(self, id_labels, id_pred_labels):
        """Withhold newborn rows until the same slot is assigned consecutively."""
        result = self.current_track_results
        slots = [int(slot) for slot in id_labels.tolist()]
        newborn_flags = [int(label) == self.num_id_vocabulary for label in id_pred_labels.tolist()]
        keep = [True] * len(slots)

        for index, (slot, is_newborn) in enumerate(zip(slots, newborn_flags)):
            row = self._tracklet_output_row(result, index)
            if is_newborn:
                # The allocator may recycle the slot of an unconfirmed track.
                # Its previous pending rows remain suppressed and are dropped.
                self._tentative_tracklets.pop(slot, None)
                self._tentative_tracklets[slot] = {
                    "count": 1,
                    "last_frame": self._frame_index,
                    "records": [(result, row)],
                }
                self.tracklet_confirmation_stats["created"] += 1
                keep[index] = False
                continue

            state = self._tentative_tracklets.get(slot)
            if state is None:
                continue
            if int(state["last_frame"]) != self._frame_index - 1:
                # Defensive fallback: this should already have been pruned at
                # frame entry, but never count a non-consecutive assignment.
                self._discard_tentative_tracklet(slot)
                continue

            state["count"] += 1
            state["last_frame"] = self._frame_index
            state["records"].append((result, row))
            if state["count"] >= self.tracklet_confirm_n_init:
                for prior_result, prior_row in state["records"][:-1]:
                    self._append_tracklet_output_row(prior_result, prior_row)
                self._tentative_tracklets.pop(slot, None)
                self.tracklet_confirmation_stats["confirmed"] += 1
            else:
                keep[index] = False

        self._filter_tracklet_output_rows(result, keep)
        result["tracklet_confirmation"] = {
            "n_init": self.tracklet_confirm_n_init,
            "pending": len(self._tentative_tracklets),
            **self.tracklet_confirmation_stats,
        }

    def _discard_unobserved_tentative_tracklets(self, observed_slots):
        observed = set(int(slot) for slot in observed_slots)
        for slot in list(self._tentative_tracklets):
            if slot not in observed:
                self._discard_tentative_tracklet(slot)

    def _filter_last_decoder_logits(self, keep_idxs: torch.Tensor):
        """Keep decoder rows aligned with detections after runtime filtering.

        Decoder logits have shape ``[..., N, K+1]``.  RuntimeTracker may drop
        illegal newborns (and, in a full ID bank, excess newborns) before
        assigning slots.  Filtering the cached diagnostics at the same point
        prevents a score row from being associated with the wrong detection.
        """
        if keep_idxs is None:
            return
        indices = keep_idxs.nonzero(as_tuple=False).flatten()
        for name in (
            "last_base_logits", "last_geometry_logits", "last_fused_logits",
            "last_geometry_candidate_ids", "last_geometry_candidate_active",
            "last_geometry_candidate_valid", "last_geometry_candidate_lag_valid",
        ):
            value = getattr(self, name, None)
            if value is not None:
                # Logits and candidate metadata use [..., detection, class/candidate].
                # Exact-lag validity additionally has a trailing lag axis:
                # [..., detection, candidate, lag], so its detection axis is -3.
                detection_axis = -3 if name == "last_geometry_candidate_lag_valid" else -2
                setattr(self, name, value.index_select(detection_axis, indices))

    def get_track_results(self):
        return self.current_track_results

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        self._active_detection_indices = activate_indices
        # Selecting:
        # logits = logits[activate_indices]
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, output_embeds

    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor):
        if self.trajectory_features.shape[0] == 0:
            # There is no ID decoder distribution for the first frame.  A
            # newborn detection is treated as a clean candidate and its
            # detector confidence is used only as the first confirmation
            # observation; subsequent frames use ID-decoder confidence.
            self.last_id_pred_confidences = torch.ones(
                boxes.shape[0], dtype=boxes.dtype, device=boxes.device,
            )
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            # 2. prepare seq_info:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                # ``trajectory_times`` remains the local causal index used by
                # MOTIP's attention masks.  Geometry matching uses actual
                # frame numbers so temporal lags stay correct after gaps.
                "trajectory_frame_times": self.trajectory_frame_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
                "unknown_frame_times": torch.full_like(
                    current_times, int(self._frame_index)
                )[None, None, ...],
            }
            # 3. forward:
            if self.visual_trail is not None:
                context = self.visual_trail.runtime_context(self.trail_state, boxes.device)
                if context is not None:
                    if self.trail_diagnostic_mode == "shuffled":
                        # Keep slot IDs (and their ID embeddings) fixed, but
                        # feed each slot the spatial memory of another active
                        # slot. A cyclic permutation is deterministic.
                        k = context["trail_tokens"].shape[3]
                        if k > 1:
                            perm = torch.roll(torch.arange(k, device=boxes.device), shifts=1)
                            context["trail_tokens"] = context["trail_tokens"][:, :, :, perm]
                    if self.trail_token_limit is not None:
                        p = context["trail_tokens"].shape[4]
                        keep = min(int(self.trail_token_limit), p)
                        context["trail_tokens"] = context["trail_tokens"][:, :, :, :, :keep]
                    if self.trail_diagnostic_mode != "off":
                        seq_info.update(context)
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
            decoder = getattr(get_model(self.model), "id_decoder", None)
            if decoder is not None and getattr(decoder, "use_geometry", False):
                self.last_geometry_logits = decoder.last_geometry_logits
                self.last_base_logits = decoder.last_base_logits
                self.last_fused_logits = decoder.last_fused_logits
                if decoder.compute_geometry_candidate_diagnostics:
                    branch = decoder.geometry_branch
                    self.last_geometry_candidate_ids = branch.last_candidate_ids
                    self.last_geometry_candidate_active = branch.last_candidate_active
                    self.last_geometry_candidate_valid = branch.last_pair_valid
                    self.last_geometry_candidate_lag_valid = branch.last_exact_lag_valid
                self.last_geometry_diagnostics = {
                    "lambda": float(
                        decoder.geometry_lambda.item()
                        if decoder.geometry_lambda is not None
                        else decoder.geometry_fixed_lambda
                    ),
                    "geometry_logit_norm": decoder.geometry_logit_norm,
                    "base_logit_norm": decoder.geometry_base_logit_norm,
                    "effective_geometry_logit_norm": decoder.geometry_effective_logit_norm,
                    "contribution_ratio": decoder.geometry_contribution_ratio,
                    "calibrated_base_logit_norm": decoder.geometry_calibrated_base_logit_norm,
                    "calibrated_geometry_logit_norm": decoder.geometry_calibrated_effective_logit_norm,
                    "calibrated_contribution_ratio": decoder.geometry_calibrated_contribution_ratio,
                    "logit_correlation": decoder.geometry_logit_correlation,
                    "margin": decoder.geometry_margin,
                    **{
                        f"candidate_{name}": value
                        for name, value in decoder.geometry_candidate_diagnostics.items()
                    },
                }
            # 4. get scores:
            id_logits = id_logits[0, 0, 0]
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            # 5. assign id labels:
            # Different assignment protocols:
            match self.assignment_protocol:
                case "hungarian":
                    id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max":
                    id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max":
                    id_labels = self._id_max_assignment(id_scores=id_scores)
                # case "object-priority": id_labels = self._object_priority_assignment(id_scores=id_scores)
                case _:
                    raise NotImplementedError

            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
            self.last_id_pred_confidences = id_scores.max(dim=-1).values
            return id_pred_labels

    @staticmethod
    def _gt_rows(gt_frame):
        """Normalize a DanceTrack annotation frame to (id, xywh) rows."""
        if gt_frame is None:
            return []
        if torch.is_tensor(gt_frame):
            gt_frame = gt_frame.detach().cpu().tolist()
        rows = []
        for row in gt_frame:
            if torch.is_tensor(row):
                row = row.detach().cpu().tolist()
            if len(row) >= 6:
                # Full TrackEval rows are frame,id,x,y,w,h,mark,class,vis.
                if len(row) >= 8 and (int(row[6]) != 1 or int(row[7]) != 1):
                    continue
                if len(row) >= 8:
                    rows.append((int(row[1]), [float(x) for x in row[2:6]]))
                else:
                    rows.append((int(row[0]), [float(x) for x in row[1:5]]))
        return rows

    def _oracle_matches(self, boxes, gt_frame, threshold=0.5):
        """Match current predictions to GT boxes for trail updates only."""
        rows = self._gt_rows(gt_frame)
        if not rows or boxes.numel() == 0:
            return []
        pred_xywh = box_cxcywh_to_xywh(boxes) * self.bbox_unnorm
        pred_xyxy = pred_xywh.clone()
        pred_xyxy[:, 2:] += pred_xyxy[:, :2]
        gt_xyxy = boxes.new_tensor([x for _, b in rows for x in b]).reshape(-1, 4)
        gt_xyxy[:, 2:] += gt_xyxy[:, :2]
        overlaps = box_iou(pred_xyxy.float(), gt_xyxy.float()).cpu()
        rr, cc = linear_sum_assignment((1 - overlaps).numpy())
        return [(int(r), rows[int(c)][0], float(overlaps[int(r), int(c)]))
                for r, c in zip(rr, cc) if overlaps[int(r), int(c)] >= threshold]

    def _prepare_trail_update(self, id_labels, boxes, rois, output_embeds, scores,
                               gt_frame=None, released_slots=None):
        """Choose persistent-memory ownership and append an audit trace.

        In oracle mode GT is consulted only after normal ID prediction and
        only to choose the memory slot receiving a matched observation.
        Predicted IDs in current_track_results and trajectory history remain
        untouched. Unmatched detections keep their predicted slot when that
        slot is not already occupied by an oracle assignment; conflicting
        unmatched observations are skipped to preserve the bank's unique-slot
        invariant.
        """
        predicted = [int(x) for x in id_labels.tolist()]
        released_slots = sorted(set(released_slots or []))
        if self.trail_diagnostic_mode != "oracle_update":
            trail_indices = torch.arange(len(predicted), device=id_labels.device, dtype=torch.long)
            trail_ids = id_labels
            matches = self._oracle_matches(boxes, gt_frame)
            if gt_frame is not None:
                by_det = {det: (gt_id, overlap) for det, gt_id, overlap in matches}
                for det_idx, slot in enumerate(predicted):
                    gt_id, overlap = by_det.get(det_idx, ("", 0.0))
                    self.audit_rows.append({
                        "frame": self._frame_index,
                        "det_index": det_idx,
                        "gt_id": gt_id,
                        "predicted_slot": slot,
                        "trail_update_slot": slot,
                        "memory_slot_previous_owner": "",
                        "matched_iou": overlap,
                        "released_slots": ";".join(map(str, released_slots)),
                        "recycled": 0,
                        "used_oracle": 0,
                        "active_memory_count_before_update": len(self.trail_state.memories),
                        "future_frame_leakage": 0,
                    })
        else:
            active_global = set(self.id_label_to_id) | set(predicted)
            for gt_id, slot in list(self.oracle_gt_to_slot.items()):
                if slot in released_slots or slot not in active_global:
                    self.oracle_gt_to_slot.pop(gt_id, None)
                    self.oracle_slot_to_gt.pop(slot, None)
            matches = self._oracle_matches(boxes, gt_frame)
            chosen = []
            chosen_slots = set()
            for det_idx, gt_id, overlap in matches:
                slot = self.oracle_gt_to_slot.get(gt_id)
                previous_owner = self.oracle_slot_to_gt.get(slot) if slot is not None else None
                if slot is None:
                    slot = predicted[det_idx]
                    if slot in chosen_slots:
                        continue
                    self.oracle_gt_to_slot[gt_id] = slot
                    self.oracle_slot_to_gt[slot] = gt_id
                elif slot in chosen_slots:
                    continue
                chosen_slots.add(slot)
                chosen.append((det_idx, slot, gt_id, overlap, previous_owner))
            matched_indices = {x[0] for x in chosen}
            # Preserve normal updates for unmatched detections where doing so
            # cannot create a duplicate slot. This keeps oracle and normal
            # memory occupancy comparable; only ownership of GT-matched
            # observations is changed.
            for det_idx, slot in enumerate(predicted):
                if det_idx not in matched_indices and slot not in chosen_slots:
                    chosen_slots.add(slot)
                    chosen.append((det_idx, slot, "", 0.0, None))
            trail_indices = torch.tensor([x[0] for x in chosen], device=id_labels.device, dtype=torch.long)
            trail_ids = torch.tensor([x[1] for x in chosen], device=id_labels.device, dtype=torch.int64)
            for det_idx, slot, gt_id, overlap, previous_owner in chosen:
                if gt_id == "":
                    continue
                self.audit_rows.append({
                    "frame": self._frame_index,
                    "det_index": det_idx,
                    "gt_id": gt_id,
                    "predicted_slot": predicted[det_idx],
                    "trail_update_slot": slot,
                    "memory_slot_previous_owner": previous_owner,
                    "matched_iou": overlap,
                    "released_slots": ";".join(map(str, released_slots)),
                    "recycled": int(previous_owner is not None and previous_owner != gt_id),
                    "used_oracle": 1,
                    "active_memory_count_before_update": len(self.trail_state.memories),
                    "future_frame_leakage": 0,
                })
            selected_indices = {x[0] for x in chosen}
            for det_idx, slot in enumerate(predicted):
                if det_idx not in selected_indices:
                    self.audit_rows.append({
                        "frame": self._frame_index,
                        "det_index": det_idx,
                        "gt_id": "",
                        "predicted_slot": slot,
                        "trail_update_slot": "",
                        "memory_slot_previous_owner": self.oracle_slot_to_gt.get(slot, ""),
                        "matched_iou": 0.0,
                        "released_slots": ";".join(map(str, released_slots)),
                        "recycled": 0,
                        "used_oracle": 1,
                        "active_memory_count_before_update": len(self.trail_state.memories),
                        "future_frame_leakage": 0,
                    })
        self.last_trail_update = {
            "frame": self._frame_index,
            "predicted_slots": predicted,
            "updated_slots": [int(x) for x in trail_ids.tolist()],
            "updated_indices": [int(x) for x in trail_indices.tolist()],
            "released_slots": released_slots,
            "matched_gt": len(matches),
            "oracle": self.trail_diagnostic_mode == "oracle_update",
        }
        return trail_ids, trail_indices

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor):
        # 1. how many newborn instances?
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        else:
            # 2. get available id labels from id_queue:
            newborn_id_labels = torch.tensor(
                list(self.id_queue)[:n_newborns], dtype=torch.int64, device=distributed_device(),
            )
            # 3. make sure these id labels are not in trajectory infos:
            if self.trail_state is not None:
                self.trail_state.release(newborn_id_labels.tolist())
            trajectory_remove_idxs = torch.zeros(
                self.trajectory_id_labels.shape[1], dtype=torch.bool, device=distributed_device(),
            )
            for _ in range(len(newborn_id_labels)):
                if self.trajectory_id_labels.shape[0] > 0:
                    trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
                if newborn_id_labels[_].item() in self.id_label_to_id:
                    self.id_label_to_id.pop(newborn_id_labels[_].item())
            # remove from trajectory infos:
            self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
            self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
            self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
            self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
            self.trajectory_frame_times = self.trajectory_frame_times[:, ~trajectory_remove_idxs]
            self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]
            # 4. assign id labels to newborn instances:
            pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
            # 5. update id infos:
            for _ in range(len(newborn_id_labels)):
                self.id_label_to_id[newborn_id_labels[_].item()] = self.next_id
                self.next_id += 1

            return pred_id_labels

    def _update_trajectory_infos(
            self,
            boxes: torch.Tensor,
            output_embeds: torch.Tensor,
            id_labels: torch.Tensor,
            commit_mask: torch.Tensor | None = None,
    ):
        # 1. cut trajectory infos:
        self.trajectory_features = self.trajectory_features[-self.miss_tolerance + 2:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-self.miss_tolerance + 2:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_times = self.trajectory_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_frame_times = self.trajectory_frame_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_masks = self.trajectory_masks[-self.miss_tolerance + 2:, ...]
        # 2. find out all new instances:
        already_id_labels = set(self.trajectory_id_labels[0].tolist() if self.trajectory_id_labels.shape[0] > 0 else [])
        _id_labels = set(id_labels.tolist())
        newborn_id_labels = _id_labels - already_id_labels
        # 3. add newborn instances to trajectory infos:
        if len(newborn_id_labels) > 0:
            newborn_id_labels = torch.tensor(list(newborn_id_labels), dtype=torch.int64, device=distributed_device())
            _T = self.trajectory_id_labels.shape[0]
            _N = len(newborn_id_labels)
            _id_labels = einops.repeat(newborn_id_labels, 'n -> t n', t=_T)
            _boxes = torch.zeros((_T, _N, 4), dtype=self.dtype, device=distributed_device())
            _times = einops.repeat(
                torch.arange(_T, dtype=torch.int64, device=distributed_device()), 't -> t n', n=_N,
            )
            _frame_times = torch.full(
                (_T, _N), -1, dtype=torch.int64, device=distributed_device()
            )
            _features = torch.zeros(
                (_T, _N, 256), dtype=self.dtype, device=distributed_device(),
            )
            _masks = torch.ones((_T, _N), dtype=torch.bool, device=distributed_device())
            # 3.1. padding to trajectory infos:
            self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, _id_labels], dim=1)
            self.trajectory_boxes = torch.cat([self.trajectory_boxes, _boxes], dim=1)
            self.trajectory_times = torch.cat([self.trajectory_times, _times], dim=1)
            self.trajectory_frame_times = torch.cat([self.trajectory_frame_times, _frame_times], dim=1)
            self.trajectory_features = torch.cat([self.trajectory_features, _features], dim=1)
            self.trajectory_masks = torch.cat([self.trajectory_masks, _masks], dim=1)
        # 4. update trajectory infos:
        _N = self.trajectory_id_labels.shape[1]
        current_id_labels = self.trajectory_id_labels[0] if self.trajectory_id_labels.shape[0] > 0 else id_labels
        current_features = torch.zeros((_N, 256), dtype=self.dtype, device=distributed_device())
        current_boxes = torch.zeros((_N, 4), dtype=self.dtype, device=distributed_device())
        current_times = self.trajectory_id_labels.shape[0] * torch.ones((_N,), dtype=torch.int64, device=distributed_device())
        current_frame_times = self._frame_index * torch.ones(
            (_N,), dtype=torch.int64, device=distributed_device()
        )
        current_masks = torch.ones((_N,), dtype=torch.bool, device=distributed_device())
        # 4.1. find out the same id labels (matching):
        indices = torch.eq(current_id_labels[:, None], id_labels[None, :]).nonzero(as_tuple=False)
        current_idxs = indices[:, 0]
        idxs = indices[:, 1]
        # 4.2. fill in the infos:
        current_id_labels[current_idxs] = id_labels[idxs]
        if commit_mask is None:
            # Exact legacy behavior when no history-write intervention is
            # requested.
            current_features[current_idxs] = output_embeds[idxs]
            current_boxes[current_idxs] = boxes[idxs]
            current_masks[current_idxs] = False
        else:
            if commit_mask.ndim != 1 or commit_mask.shape != id_labels.shape or commit_mask.dtype != torch.bool:
                raise ValueError("commit_mask must be a boolean vector aligned with id_labels")
            should_commit = commit_mask[idxs]
            write_current_idxs = current_idxs[should_commit]
            write_detection_idxs = idxs[should_commit]
            current_features[write_current_idxs] = output_embeds[write_detection_idxs]
            current_boxes[write_current_idxs] = boxes[write_detection_idxs]
            current_masks[write_current_idxs] = False
        # 4.3. cat to trajectory infos:
        self.trajectory_features = torch.cat([self.trajectory_features, current_features[None, ...]], dim=0).contiguous()
        self.trajectory_boxes = torch.cat([self.trajectory_boxes, current_boxes[None, ...]], dim=0).contiguous()
        self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, current_id_labels[None, ...]], dim=0).contiguous()
        self.trajectory_times = torch.cat([self.trajectory_times, current_times[None, ...]], dim=0).contiguous()
        self.trajectory_frame_times = torch.cat(
            [self.trajectory_frame_times, current_frame_times[None, ...]], dim=0
        ).contiguous()
        self.trajectory_masks = torch.cat([self.trajectory_masks, current_masks[None, ...]], dim=0).contiguous()
        # 4.4. a hack implementation to fix "times":
        self.trajectory_times = einops.repeat(
            torch.arange(self.trajectory_times.shape[0], dtype=torch.int64, device=distributed_device()),
            't -> t n', n=self.trajectory_times.shape[1],
        ).contiguous().clone()
        return

    def _filter_out_inactive_tracks(self):
        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        if self.trail_state is not None:
            self.trail_state.retain(self.trajectory_id_labels[-1, is_active].tolist())
        self.trajectory_features = self.trajectory_features[:, is_active]
        self.trajectory_boxes = self.trajectory_boxes[:, is_active]
        self.trajectory_id_labels = self.trajectory_id_labels[:, is_active]
        self.trajectory_times = self.trajectory_times[:, is_active]
        # Older callers/tests may construct a tracker state manually before
        # the absolute-frame-time buffer was introduced.  Keep that state
        # backward compatible while preserving the exact timestamp buffer for
        # normal runtime/inference paths.
        if self.trajectory_frame_times.numel() and self.trajectory_frame_times.shape[1] == is_active.shape[0]:
            self.trajectory_frame_times = self.trajectory_frame_times[:, is_active]
        self.trajectory_masks = self.trajectory_masks[:, is_active]
        return

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        if len(id_scores) > 1:
            id_scores_newborn_repeat = id_scores[:, -1:].repeat(1, len(id_scores) - 1)
            id_scores = torch.cat((id_scores, id_scores_newborn_repeat), dim=-1)
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())
        match_rows, match_cols = linear_sum_assignment(1 - id_scores.cpu())
        for _ in range(len(match_rows)):
            _id = match_cols[_]
            if _id not in trajectory_id_labels_set:
                id_labels.append(self.num_id_vocabulary)
            elif _id >= self.num_id_vocabulary:
                id_labels.append(self.num_id_vocabulary)
            elif id_scores[match_rows[_], _id] < self.id_thresh:
                id_labels.append(self.num_id_vocabulary)
            else:
                id_labels.append(_id)
        return id_labels

    def _object_max_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        object_max_confs, object_max_id_labels = torch.max(id_scores, dim=-1)   # get the target ID labels and confs
        # Get the max confs of each ID label:
        id_max_confs = dict()
        for conf, id_label in zip(object_max_confs.tolist(), object_max_id_labels.tolist()):
            if id_label not in id_max_confs:
                id_max_confs[id_label] = conf
            else:
                # if conf == id_max_confs[id_label]:  # a very rare case
                #     conf = conf - 0.0001
                id_max_confs[id_label] = max(id_max_confs[id_label], conf)
        if self.num_id_vocabulary in id_max_confs:
            id_max_confs[self.num_id_vocabulary] = 0.0  # special token

        # Assign ID labels:
        for _ in range(len(object_max_id_labels)):
            if object_max_id_labels[_].item() not in trajectory_id_labels_set:         # not in tracked IDs -> newborn
                id_labels.append(self.num_id_vocabulary)
            else:
                _id_label = object_max_id_labels[_].item()
                _conf = object_max_confs[_].item()
                if _conf < self.id_thresh or _conf < id_max_confs[_id_label]:  # low conf or not the max conf -> newborn
                    id_labels.append(self.num_id_vocabulary)
                elif _id_label in id_labels:
                    id_labels.append(self.num_id_vocabulary)
                else:                                                          # normal case
                    id_labels.append(_id_label)

        return id_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        id_labels = [self.num_id_vocabulary] * len(id_scores)  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        id_max_confs, id_max_obj_idxs = torch.max(id_scores, dim=0)
        # Get the max confs of each object:
        object_max_confs = dict()
        for conf, object_idx in zip(id_max_confs.tolist(), id_max_obj_idxs.tolist()):
            if object_idx not in object_max_confs:
                object_max_confs[object_idx] = conf
            else:
                if conf == object_max_confs[object_idx]:    # a very rare case
                    conf = conf - 0.0001
                object_max_confs[object_idx] = max(object_max_confs[object_idx], conf)

        # Assign ID labels:
        for _ in range(len(id_max_obj_idxs)):
            _obj_idx, _id_label, _conf = id_max_obj_idxs[_].item(), _, id_max_confs[_].item()
            if _conf < self.id_thresh or _conf < object_max_confs[_obj_idx]:
                pass
            elif _id_label not in trajectory_id_labels_set:
                pass
            else:
                id_labels[_obj_idx] = _id_label

        return id_labels
