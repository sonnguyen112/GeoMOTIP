#!/usr/bin/env python3
"""Cache detector boxes linked by official-train GT identity, with no ID tracker.

The frozen detector supplies every predicted box. Train GT supplies only the
matching/ownership labels and GT-GT warm-start boxes. Candidate histories are
read before the current frame is appended, so exact lags {1,2,4,8} are causal.
No validation sequence, trained E0 association checkpoint, or MOTIP ID Decoder
is used by this cache builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dancetrack import DanceTrack
from data.seq_dataset import SeqDataset
from data.sequence_manifest import read_sequence_manifest
from models.runtime_tracker import RuntimeTracker
from tools.build_predicted_geometry_cache import LAGS, _gt_lookup, _load_model
from tools.run_v2_evaluation import _gt_by_frame, _matches
from utils.box_ops import box_cxcywh_to_xywh


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_sequence(cfg, model, data_root: Path, sequence: str, workers: int):
    dataset = DanceTrack(data_root=str(data_root), split="train", load_annotation=False)
    info = dataset.sequence_infos[sequence]
    root = data_root / "DanceTrack" / "train" / sequence
    image_paths = [str(root / "img1" / f"{frame:08d}.jpg")
                   for frame in range(1, info["length"] + 1)]
    seq_dataset = SeqDataset(
        seq_info=info, image_paths=image_paths,
        max_shorter=cfg.get("INFERENCE_MAX_SHORTER", 800),
        max_longer=cfg.get("INFERENCE_MAX_LONGER", 1440),
        size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0),
        dtype=torch.float32,
    )
    loader = DataLoader(
        seq_dataset, batch_size=1, shuffle=False, num_workers=workers,
        pin_memory=True, collate_fn=lambda items: items[0],
    )
    gt_frames = _gt_by_frame(root / "gt" / "gt.txt", len(loader))
    gt_lookup = _gt_lookup(gt_frames, info["width"], info["height"])
    # This helper applies exactly the tracker detector activation threshold.
    # Its update() method is never called: no MOTIP association runs here.
    detector_filter = RuntimeTracker(
        model=model, sequence_hw=seq_dataset.seq_hw(),
        use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
        assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "object-max"),
        miss_tolerance=cfg.get("MISS_TOLERANCE", 30),
        det_thresh=cfg.get("DET_THRESH", 0.3),
        newborn_thresh=cfg.get("NEWBORN_THRESH", 0.6),
        id_thresh=cfg.get("ID_THRESH", 0.2),
        area_thresh=cfg.get("AREA_THRESH", 0),
        only_detr=True, dtype=torch.float32,
    )
    output: dict[str, list] = {key: [] for key in (
        "current_box_pred", "current_box_gt", "current_gt", "sequence", "frame",
        "current_score", "pair_row", "pair_gt", "pair_age",
        "pair_lag_boxes_pred", "pair_lag_valid_pred",
        "pair_lag_boxes_gt", "pair_lag_valid_gt",
    )}
    # GT IDs are dictionary keys for *training labels* only; they never enter
    # build_g4_features or the geometry MLP input.
    predicted_history: dict[int, dict[int, torch.Tensor]] = {}
    matched_detections = 0
    total_detections = 0
    target_present_rows = 0
    for frame, (image, _path) in enumerate(loader, 1):
        image.tensors = image.tensors.cuda(non_blocking=True)
        image.mask = image.mask.cuda(non_blocking=True)
        with torch.inference_mode():
            detector_output = model(frames=image, part="detr")
            scores, _categories, boxes, _features = detector_filter._get_activate_detections(
                detr_out=detector_output
            )
        boxes = boxes.detach().float().cpu()
        scores = scores.detach().float().cpu()
        total_detections += len(boxes)
        pixel_xywh = box_cxcywh_to_xywh(boxes) * detector_filter.bbox_unnorm.cpu()
        matches = _matches(gt_frames[frame - 1], {"bbox": pixel_xywh})
        matched_detections += len(matches)
        candidate_ids = sorted(
            identity for identity, history in predicted_history.items()
            if any(frame - lag in history for lag in LAGS)
        )
        # Read histories before adding current observations. This makes every
        # pair strictly past-only, including for identities observed now.
        for det_index, (gt_id, _iou) in sorted(matches.items()):
            if not candidate_ids:
                continue
            gt_id = int(gt_id)
            current_gt = gt_lookup.get((frame, gt_id))
            if current_gt is None:
                raise AssertionError(f"Missing train GT box: {sequence} {frame} {gt_id}")
            row_index = len(output["current_gt"])
            output["current_box_pred"].append(boxes[det_index].clone())
            output["current_box_gt"].append(current_gt.clone())
            output["current_gt"].append(gt_id)
            output["sequence"].append(sequence)
            output["frame"].append(frame)
            output["current_score"].append(float(scores[det_index]))
            target_present_rows += int(gt_id in candidate_ids)
            for candidate_id in candidate_ids:
                history = predicted_history[candidate_id]
                pred_boxes = torch.zeros(len(LAGS), 4, dtype=torch.float32)
                pred_valid = torch.zeros(len(LAGS), dtype=torch.bool)
                gt_boxes = torch.zeros_like(pred_boxes)
                gt_valid = torch.zeros_like(pred_valid)
                for lag_index, lag in enumerate(LAGS):
                    past_frame = frame - lag
                    if past_frame >= frame:
                        raise AssertionError("Future or current frame used as history")
                    if past_frame in history:
                        pred_boxes[lag_index] = history[past_frame]
                        pred_valid[lag_index] = True
                    old_gt = gt_lookup.get((past_frame, candidate_id))
                    if old_gt is not None:
                        gt_boxes[lag_index] = old_gt
                        gt_valid[lag_index] = True
                if not bool(pred_valid.any()):
                    raise AssertionError("Candidate has no exact-lag predicted history")
                output["pair_row"].append(row_index)
                output["pair_gt"].append(candidate_id)
                output["pair_age"].append(frame - min(history))
                output["pair_lag_boxes_pred"].append(pred_boxes)
                output["pair_lag_valid_pred"].append(pred_valid)
                output["pair_lag_boxes_gt"].append(gt_boxes)
                output["pair_lag_valid_gt"].append(gt_valid)
        for det_index, (gt_id, _iou) in matches.items():
            predicted_history.setdefault(int(gt_id), {})[frame] = boxes[det_index].clone()
    tensor_output = {
        "current_box_pred": torch.stack(output["current_box_pred"]),
        "current_box_gt": torch.stack(output["current_box_gt"]),
        "current_gt": torch.tensor(output["current_gt"], dtype=torch.long),
        "sequence": output["sequence"],
        "frame": torch.tensor(output["frame"], dtype=torch.long),
        "current_score": torch.tensor(output["current_score"], dtype=torch.float32),
        "pair_row": torch.tensor(output["pair_row"], dtype=torch.long),
        "pair_gt": torch.tensor(output["pair_gt"], dtype=torch.long),
        "pair_age": torch.tensor(output["pair_age"], dtype=torch.long),
        "pair_lag_boxes_pred": torch.stack(output["pair_lag_boxes_pred"]),
        "pair_lag_valid_pred": torch.stack(output["pair_lag_valid_pred"]),
        "pair_lag_boxes_gt": torch.stack(output["pair_lag_boxes_gt"]),
        "pair_lag_valid_gt": torch.stack(output["pair_lag_valid_gt"]),
        "lags": torch.tensor(LAGS, dtype=torch.long),
    }
    return tensor_output, {
        "split": "train", "sequence": sequence, "frames": len(loader),
        "detections": total_detections, "matched_detections": matched_detections,
        "current_rows": len(output["current_gt"]),
        "target_present_rows": target_present_rows,
        "pair_rows": len(output["pair_row"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing cache: {args.output}")
    names = read_sequence_manifest(args.train_manifest)
    train_root = args.data_root / "DanceTrack" / "train"
    val_root = args.data_root / "DanceTrack" / "val"
    if not names or any(not (train_root / name).is_dir() for name in names):
        raise ValueError("Every manifest sequence must exist in official DanceTrack.train")
    if any((val_root / name).exists() for name in names):
        raise ValueError("Train cache manifest overlaps official DanceTrack.val")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg, model = _load_model(args.config, args.checkpoint)
    if cfg.get("USE_GEOMETRY", False) or cfg.get("USE_TRAIL", False):
        raise ValueError("GT-linked cache must use detector-only E0 architecture")
    cache_parts = []
    metadata = {
        "config": str(args.config), "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "train_manifest": str(args.train_manifest),
        "train_manifest_sha256": sha256(args.train_manifest),
        "seed": args.seed, "lags": list(LAGS), "train_only": True,
        "gt_linked_predicted_history": True,
        "association_model_called": False,
        "validation_used": False,
        "detector_threshold": cfg.get("DET_THRESH", 0.3),
        "gt_match_iou_threshold": 0.5,
        "sequences": {},
    }
    for name in names:
        print(f"[gt-linked-cache] train/{name}", flush=True)
        part, info = _extract_sequence(cfg, model, args.data_root, name, args.num_workers)
        cache_parts.append(part)
        metadata["sequences"][name] = info
    offsets = []
    running = 0
    for part in cache_parts:
        offsets.append(running)
        running += int(part["current_gt"].numel())
    merged = {}
    for key in cache_parts[0]:
        if key == "lags":
            merged[key] = cache_parts[0][key]
        elif key == "sequence":
            merged[key] = sum((part[key] for part in cache_parts), [])
        elif key == "pair_row":
            merged[key] = torch.cat([
                part[key] + offset for part, offset in zip(cache_parts, offsets)
            ])
        else:
            merged[key] = torch.cat([part[key] for part in cache_parts])
    if not bool((merged["pair_lag_valid_pred"].any(-1)).all()):
        raise AssertionError("Every saved pair must have valid past predicted geometry")
    if not bool((merged["current_gt"][merged["pair_row"]] == merged["pair_gt"]).any()):
        raise ValueError("No positive current/candidate association pairs")
    args.output.mkdir(parents=True)
    torch.save(merged, args.output / "train_cache.pt")
    metadata["tensor_shapes"] = {
        key: list(value.shape) for key, value in merged.items() if torch.is_tensor(value)
    }
    metadata["notes"] = (
        "Frozen DETR predicted boxes only; official-train GT links identities. "
        "Candidate histories are read before current-frame update; exact lags only. "
        "No trained E0 association model and no validation data."
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
