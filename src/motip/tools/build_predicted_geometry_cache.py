#!/usr/bin/env python3
"""Cache frozen-MOTIP detections and timestamped candidate histories.

This cache is deliberately independent of the visual-trail state.  Candidate
tracks are the same local MOTIP slots used by the normal runtime decoder, while
the owner labels are an audit-only IoU match to DanceTrack ground truth.  The
cache therefore supports the three controlled geometry distributions:

* ``GT_GT``: current and historical ground-truth boxes;
* ``PRED_GT``: detector current box, ground-truth history;
* ``PRED_PRED``: detector current box, predicted tracker history.

Only observations strictly before the current frame are used for history and
all requested lags require an exact frame timestamp.  No nearest-frame or
future substitution is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OPS_BUILD = ROOT / "models" / "ops" / "build" / "lib.linux-x86_64-cpython-312"
if OPS_BUILD.exists() and str(OPS_BUILD) not in sys.path:
    sys.path.insert(0, str(OPS_BUILD))

from data.dancetrack import DanceTrack  # noqa: E402
from data.seq_dataset import SeqDataset  # noqa: E402
from data.sequence_manifest import read_sequence_manifest  # noqa: E402
from models.misc import load_checkpoint  # noqa: E402
from models.motip import build  # noqa: E402
from models.runtime_tracker import RuntimeTracker  # noqa: E402
from tools.run_v2_evaluation import _gt_by_frame, _matches  # noqa: E402
from configs.util import load_super_config  # noqa: E402
from utils.misc import yaml_to_dict  # noqa: E402


LAGS = (1, 2, 4, 8)


def _xywh_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    return torch.stack((box[..., 0] + box[..., 2] / 2,
                        box[..., 1] + box[..., 3] / 2,
                        box[..., 2], box[..., 3]), dim=-1)


def _load_model(config_path: Path, checkpoint: Path):
    cfg = yaml_to_dict(str(config_path))
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    # Cache generation is intentionally deterministic FP32 inference.  The
    # exact same detector checkpoint and association settings are used for all
    # three geometry distributions.
    cfg["INFERENCE_DTYPE"] = "FP32"
    model, _ = build(cfg)
    load_checkpoint(model, str(checkpoint))
    model.cuda().eval()
    return cfg, model


def _slot_columns(tracker: RuntimeTracker):
    """Return stable local slots and their trajectory columns before update."""
    if tracker.trajectory_id_labels.shape[0] == 0:
        return []
    labels = tracker.trajectory_id_labels.detach().cpu()
    masks = tracker.trajectory_masks.detach().cpu()
    columns = []
    for col, slot in enumerate(labels[0].tolist()):
        if bool((~masks[:, col]).any()):
            columns.append((int(slot), col))
    return columns


def _history_for_slots(tracker: RuntimeTracker, slots, frame: int):
    """Extract exact predicted boxes at current-frame minus each lag."""
    k = len(slots)
    boxes = torch.zeros(k, len(LAGS), 4, dtype=torch.float32)
    valid = torch.zeros(k, len(LAGS), dtype=torch.bool)
    ages = torch.zeros(k, dtype=torch.long)
    if not slots or tracker.trajectory_boxes.shape[0] == 0:
        return boxes, valid, ages
    hboxes = tracker.trajectory_boxes.detach().float().cpu()
    htimes = tracker.trajectory_frame_times.detach().cpu().long()
    hmasks = tracker.trajectory_masks.detach().cpu()
    for out_idx, (_slot, col) in enumerate(slots):
        observed = (~hmasks[:, col]).nonzero(as_tuple=False).flatten()
        if len(observed):
            first = int(htimes[int(observed[0]), col].item())
            ages[out_idx] = max(0, int(frame) - first)
        for lag_idx, lag in enumerate(LAGS):
            target = int(frame) - int(lag)
            hits = ((htimes[:, col] == target) & (~hmasks[:, col])).nonzero(
                as_tuple=False
            ).flatten()
            if len(hits):
                row = int(hits[0].item())
                boxes[out_idx, lag_idx] = hboxes[row, col]
                valid[out_idx, lag_idx] = True
    return boxes, valid, ages


def _gt_lookup(gt_frames, width: int, height: int):
    """Map (frame, dataset id) to normalized ``cxcywh`` ground-truth box."""
    lookup = {}
    for frame_idx, rows in enumerate(gt_frames, 1):
        for row in rows:
            row = row.tolist() if hasattr(row, "tolist") else row
            if len(row) >= 8 and (int(row[6]) != 1 or int(row[7]) != 1):
                continue
            if len(row) >= 8:
                obj_id, xywh = int(row[1]), [float(x) for x in row[2:6]]
            elif len(row) >= 5:
                obj_id, xywh = int(row[0]), [float(x) for x in row[1:5]]
            else:
                continue
            norm = [xywh[0] / width, xywh[1] / height,
                    xywh[2] / width, xywh[3] / height]
            lookup[(frame_idx, obj_id)] = _xywh_to_cxcywh(
                torch.tensor(norm, dtype=torch.float32)
            )
    return lookup


def _active_slots_after(tracker: RuntimeTracker):
    if tracker.trajectory_id_labels.shape[0] == 0:
        return set()
    labels = tracker.trajectory_id_labels[-1].detach().cpu()
    masks = tracker.trajectory_masks[-1].detach().cpu()
    return {int(labels[col].item()) for col in range(len(labels)) if not bool(masks[col])}


def _extract_sequence(cfg, model, data_root: Path, split: str, sequence: str, args):
    dataset = DanceTrack(data_root=str(data_root), split=split, load_annotation=False)
    seq_info = dataset.sequence_infos[sequence]
    sequence_root = data_root / "DanceTrack" / split / sequence
    image_paths = [
        str(sequence_root / "img1" / f"{idx + 1:08d}.jpg")
        for idx in range(seq_info["length"])
    ]
    if args.max_frames:
        image_paths = image_paths[: int(args.max_frames)]
    sequence_dataset = SeqDataset(
        seq_info=seq_info,
        image_paths=image_paths,
        max_shorter=cfg.get("INFERENCE_MAX_SHORTER", 800),
        max_longer=cfg.get("INFERENCE_MAX_LONGER", 1440),
        size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0),
        dtype=torch.float32,
    )
    loader = DataLoader(
        sequence_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=lambda x: x[0],
    )
    gt_frames = _gt_by_frame(sequence_root / "gt" / "gt.txt", len(loader))
    gt_lookup = _gt_lookup(gt_frames, seq_info["width"], seq_info["height"])
    tracker = RuntimeTracker(
        model=model, sequence_hw=sequence_dataset.seq_hw(),
        use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
        assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "object-max"),
        miss_tolerance=cfg.get("MISS_TOLERANCE", 30),
        det_thresh=cfg.get("DET_THRESH", 0.3),
        newborn_thresh=cfg.get("NEWBORN_THRESH", 0.6),
        id_thresh=cfg.get("ID_THRESH", 0.2),
        area_thresh=cfg.get("AREA_THRESH", 0),
        only_detr=cfg.get("INFERENCE_ONLY_DETR", False),
        dtype=torch.float32,
        trail_diagnostic_mode="normal",
    )

    rows = {
        "current_box_pred": [], "current_box_gt": [], "current_gt": [],
        "current_feature": [], "current_score": [],
        "sequence": [], "frame": [], "current_slot": [],
        "pair_row": [], "pair_slot": [], "pair_gt": [], "pair_age": [],
        "pair_lag_boxes_pred": [], "pair_lag_valid_pred": [],
        "pair_lag_boxes_gt": [], "pair_lag_valid_gt": [],
    }
    detections = {
        "sequence": [], "frame": [], "slot": [], "score": [],
        "box_pred": [], "matched_gt": [], "matched_iou": [],
    }
    owner_by_slot = {}
    frame_count = 0
    matched_count = 0
    pair_count = 0
    for frame, (image, _path) in enumerate(loader, 1):
        frame_count += 1
        image.tensors = image.tensors.cuda(non_blocking=True)
        image.mask = image.mask.cuda(non_blocking=True)
        slots = _slot_columns(tracker)
        pred_lag, pred_valid, ages = _history_for_slots(tracker, slots, frame)
        gt_lag = torch.zeros_like(pred_lag)
        gt_valid = torch.zeros_like(pred_valid)
        for idx, (slot, _col) in enumerate(slots):
            owner = owner_by_slot.get(slot)
            if owner is None:
                continue
            for lag_idx, lag in enumerate(LAGS):
                value = gt_lookup.get((frame - int(lag), int(owner)))
                if value is not None:
                    gt_lag[idx, lag_idx] = value
                    gt_valid[idx, lag_idx] = True

        tracker.update(image=image, frame_index=frame, gt_frame=gt_frames[frame - 1])
        result = tracker.get_track_results()
        result_slots = result.get("slot_id", torch.empty(0, dtype=torch.long))
        result_slots = result_slots.tolist()
        result_scores = result.get("score", torch.empty(0))
        matches = _matches(gt_frames[frame - 1], result)
        labels = tracker.trajectory_id_labels[-1].detach().cpu()
        masks = tracker.trajectory_masks[-1].detach().cpu()
        boxes_now = tracker.trajectory_boxes[-1].detach().float().cpu()
        feats_now = tracker.trajectory_features[-1].detach().float().cpu()
        slot_to_col = {
            int(labels[col].item()): col
            for col in range(len(labels)) if not bool(masks[col])
        }

        for det_idx, slot in enumerate(result_slots):
            score = float(result_scores[det_idx].item()) if det_idx < len(result_scores) else 0.0
            gt_id, iou = matches.get(det_idx, (-1, 0.0))
            detections["sequence"].append(sequence)
            detections["frame"].append(frame)
            detections["slot"].append(int(slot))
            detections["score"].append(score)
            detections["box_pred"].append(boxes_now[slot_to_col[slot]].clone() if slot in slot_to_col else torch.zeros(4))
            detections["matched_gt"].append(int(gt_id))
            detections["matched_iou"].append(float(iou))
            if det_idx not in matches or not slots:
                continue
            col = slot_to_col.get(int(slot))
            if col is None:
                continue
            known = [
                (idx, cslot, owner_by_slot.get(cslot))
                for idx, (cslot, _ccol) in enumerate(slots)
                if owner_by_slot.get(cslot) is not None
            ]
            if not known:
                continue
            gt_id = int(gt_id)
            cur_row = len(rows["current_gt"])
            current_pred = boxes_now[col].clone()
            current_gt = gt_lookup.get((frame, gt_id), torch.zeros(4))
            rows["current_box_pred"].append(current_pred)
            rows["current_box_gt"].append(current_gt)
            rows["current_gt"].append(gt_id)
            rows["current_feature"].append(feats_now[col].clone())
            rows["current_score"].append(score)
            rows["sequence"].append(sequence)
            rows["frame"].append(frame)
            rows["current_slot"].append(int(slot))
            for idx, cslot, owner in known:
                rows["pair_row"].append(cur_row)
                rows["pair_slot"].append(int(cslot))
                rows["pair_gt"].append(int(owner))
                rows["pair_age"].append(int(ages[idx].item()))
                rows["pair_lag_boxes_pred"].append(pred_lag[idx].clone())
                rows["pair_lag_valid_pred"].append(pred_valid[idx].clone())
                rows["pair_lag_boxes_gt"].append(gt_lag[idx].clone())
                rows["pair_lag_valid_gt"].append(gt_valid[idx].clone())
                pair_count += 1
            matched_count += 1

        # Ownership is updated only after the current prediction has been
        # recorded.  It is an audit label for candidate sets, not a signal
        # fed back into MOTIP.
        active_after = _active_slots_after(tracker)
        owner_by_slot = {
            slot: owner for slot, owner in owner_by_slot.items()
            if slot in active_after
        }
        for det_idx, slot in enumerate(result_slots):
            if det_idx in matches and int(slot) in active_after:
                owner_by_slot[int(slot)] = int(matches[det_idx][0])

    out = {
        "current_box_pred": torch.stack(rows["current_box_pred"]) if rows["current_box_pred"] else torch.zeros(0, 4),
        "current_box_gt": torch.stack(rows["current_box_gt"]) if rows["current_box_gt"] else torch.zeros(0, 4),
        "current_gt": torch.tensor(rows["current_gt"], dtype=torch.long),
        "current_feature": torch.stack(rows["current_feature"]) if rows["current_feature"] else torch.zeros(0, 256),
        "current_score": torch.tensor(rows["current_score"], dtype=torch.float32),
        "sequence": rows["sequence"],
        "frame": torch.tensor(rows["frame"], dtype=torch.long),
        "current_slot": torch.tensor(rows["current_slot"], dtype=torch.long),
        "pair_row": torch.tensor(rows["pair_row"], dtype=torch.long),
        "pair_slot": torch.tensor(rows["pair_slot"], dtype=torch.long),
        "pair_gt": torch.tensor(rows["pair_gt"], dtype=torch.long),
        "pair_age": torch.tensor(rows["pair_age"], dtype=torch.long),
        "pair_lag_boxes_pred": torch.stack(rows["pair_lag_boxes_pred"]) if rows["pair_lag_boxes_pred"] else torch.zeros(0, 4, 4),
        "pair_lag_valid_pred": torch.stack(rows["pair_lag_valid_pred"]) if rows["pair_lag_valid_pred"] else torch.zeros(0, 4, dtype=torch.bool),
        "pair_lag_boxes_gt": torch.stack(rows["pair_lag_boxes_gt"]) if rows["pair_lag_boxes_gt"] else torch.zeros(0, 4, 4),
        "pair_lag_valid_gt": torch.stack(rows["pair_lag_valid_gt"]) if rows["pair_lag_valid_gt"] else torch.zeros(0, 4, dtype=torch.bool),
        "lags": torch.tensor(LAGS, dtype=torch.long),
    }
    det_out = {
        "sequence": detections["sequence"], "frame": torch.tensor(detections["frame"], dtype=torch.long),
        "slot": torch.tensor(detections["slot"], dtype=torch.long),
        "score": torch.tensor(detections["score"], dtype=torch.float32),
        "box_pred": torch.stack(detections["box_pred"]) if detections["box_pred"] else torch.zeros(0, 4),
        "matched_gt": torch.tensor(detections["matched_gt"], dtype=torch.long),
        "matched_iou": torch.tensor(detections["matched_iou"], dtype=torch.float32),
    }
    return out, det_out, {
        "split": split, "sequence": sequence, "frames": frame_count,
        "matched_current_rows": matched_count, "pair_rows": pair_count,
        "detections": len(det_out["frame"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/trail_v3/e0_probe_val_eval.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/research_v1/full_E0_subset_s42/checkpoint_9.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--train-manifest", type=Path, default=Path("configs/trail_v3/probe_train_manifest.txt"))
    parser.add_argument("--val-manifest", type=Path, default=Path("configs/trail_v3/probe_val_manifest.txt"))
    parser.add_argument("--train-dataset-split", default="val",
                        help="Dataset directory containing the fixed probe-train manifest (the representative subset is val-only in this checkout).")
    parser.add_argument("--val-dataset-split", default="val")
    parser.add_argument(
        "--train-only", action="store_true",
        help="Cache only the training manifest; never load validation images or labels.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/research_v3/predicted_geometry_cache"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg, model = _load_model(args.config, args.checkpoint)
    split_manifests = {
        "train": (args.train_dataset_split, read_sequence_manifest(args.train_manifest)),
    }
    if not args.train_only:
        split_manifests["val"] = (
            args.val_dataset_split, read_sequence_manifest(args.val_manifest)
        )
    if args.train_only and any(args.output.glob("*cache.pt")):
        raise FileExistsError(f"Refusing to replace an existing cache in {args.output}")
    all_cache, all_det, metadata = {}, {}, {
        "config": str(args.config), "checkpoint": str(args.checkpoint),
        "seed": args.seed, "lags": list(LAGS),
        "train_only": args.train_only, "sequences": {},
    }
    for split, (dataset_split, names) in split_manifests.items():
        for name in names:
            print(f"[pred-cache] {dataset_split}/{name}", flush=True)
            cache, det, info = _extract_sequence(cfg, model, args.data_root, dataset_split, name, args)
            all_cache.setdefault(split, []).append(cache)
            all_det.setdefault(split, []).append(det)
            metadata["sequences"][name] = info
    # Concatenate sequence-local caches while preserving an explicit sequence
    # list. Pair/current indices are local to each sequence and are offset here.
    merged = {}
    merged_det = {}
    for split, items in all_cache.items():
        keys = [k for k in items[0] if torch.is_tensor(items[0][k])] if items else []
        merged_split = {}
        current_offsets = []
        running = 0
        for item in items:
            current_offsets.append(running)
            running += int(item["current_gt"].shape[0])
        for key in keys:
            vals = [item[key] for item in items]
            if key == "lags":
                # This is sequence-independent metadata, not a per-frame
                # tensor.  Preserve one canonical [1,2,4,8] vector instead
                # of repeating it once per selected sequence.
                merged_split[key] = vals[0]
                continue
            if key == "pair_row":
                adjusted = [value + offset for value, offset in zip(vals, current_offsets)]
                merged_split[key] = torch.cat(adjusted) if adjusted else torch.zeros(0, dtype=torch.long)
            else:
                merged_split[key] = torch.cat(vals) if vals and vals[0].ndim else torch.tensor(vals)
        for key in ("sequence",):
            merged_split[key] = sum((item[key] for item in items), [])
        merged[split] = merged_split
        det_items = all_det[split]
        merged_det[split] = {
            "sequence": sum((item["sequence"] for item in det_items), []),
            **{key: torch.cat([item[key] for item in det_items]) for key in det_items[0] if key != "sequence"},
        }
        torch.save(merged_split, args.output / f"{split}_cache.pt")
        torch.save(merged_det[split], args.output / f"{split}_detections.pt")
    metadata["tensor_shapes"] = {
        split: {key: list(value.shape) for key, value in cache.items() if torch.is_tensor(value)}
        for split, cache in merged.items()
    }
    metadata["notes"] = "Predicted boxes are normalized cxcywh from RuntimeTracker. GT matching is audit-only IoU>=0.5; exact timestamped history uses frame-lag equality and never future frames."
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
