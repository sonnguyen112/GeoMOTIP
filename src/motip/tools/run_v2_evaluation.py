#!/usr/bin/env python3
"""Evaluate Trail-MOTIP V2 checkpoints on the fixed representative subset.

This is an inference-only companion to ``run_trail_experiment.py``.  It keeps
the E0/E5 TrackEval protocol, while also recording the V2 memory write
statuses, residual-logit norms, confirmation counters, and a conservative
runtime estimate of persistent-memory ownership precision.  No ground truth is
used to alter predictions or memory updates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dancetrack import DanceTrack  # noqa: E402
from data.seq_dataset import SeqDataset  # noqa: E402
from data.sequence_manifest import read_sequence_manifest, write_trackeval_seqmap  # noqa: E402
from models.misc import get_model, load_checkpoint  # noqa: E402
from models.motip import build  # noqa: E402
from models.runtime_tracker import RuntimeTracker  # noqa: E402
from utils.misc import yaml_to_dict  # noqa: E402


def _gt_by_frame(path: Path, length: int):
    if not path.exists():
        return [np.empty((0, 9), dtype=np.float32) for _ in range(length)]
    data = np.loadtxt(path, delimiter=",", ndmin=2).astype(np.float32)
    return [data[data[:, 0] == frame] for frame in range(1, length + 1)]


def _write_tracker(path: Path, sequence_results):
    rows = []
    for frame_idx, result in enumerate(sequence_results, 1):
        for obj_id, bbox in zip(result["id"], result["bbox"]):
            rows.append(
                f"{frame_idx},{int(obj_id.item())},"
                f"{float(bbox[0].item())},{float(bbox[1].item())},"
                f"{float(bbox[2].item())},{float(bbox[3].item())},1,-1,-1,-1\n"
            )
    path.write_text("".join(rows))


def _summary(path: Path):
    lines = path.read_text().splitlines()
    names, values = lines[:2]
    return {key: float(value) for key, value in zip(names.split(), values.split())}


def _eval_variant(variant_dir: Path, data_root: Path, names: list[str]):
    seqmap = write_trackeval_seqmap(variant_dir / "val_seqmap.txt", names)
    tracker_dir = variant_dir / "tracker"
    cmd = [
        sys.executable, "TrackEval/scripts/run_mot_challenge.py",
        "--SPLIT_TO_EVAL", "val", "--METRICS", "HOTA", "CLEAR", "Identity",
        "--GT_FOLDER", str(data_root / "DanceTrack" / "val"),
        "--SEQMAP_FILE", str(seqmap), "--SKIP_SPLIT_FOL", "True",
        "--TRACKERS_TO_EVAL", "", "--TRACKER_SUB_FOLDER", "",
        "--USE_PARALLEL", "True", "--NUM_PARALLEL_CORES", "8",
        "--PLOT_CURVES", "False", "--TRACKERS_FOLDER", str(tracker_dir),
    ]
    with (variant_dir / "trackeval.log").open("w") as handle:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"TrackEval failed for {variant_dir.name}; see {variant_dir / 'trackeval.log'}")
    summary_path = tracker_dir / "pedestrian_summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return _summary(summary_path), cmd


def _iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _matches(gt_frame, result, threshold=0.5):
    """Return prediction-index -> (gt id, IoU) for an audit-only match."""
    if gt_frame is None or len(gt_frame) == 0 or len(result.get("bbox", [])) == 0:
        return {}
    gt = []
    for row in gt_frame:
        row = row.tolist() if hasattr(row, "tolist") else row
        if len(row) >= 8 and (int(row[6]) != 1 or int(row[7]) != 1):
            continue
        if len(row) >= 8:
            gt.append((int(row[1]), [float(x) for x in row[2:6]]))
        elif len(row) >= 5:
            gt.append((int(row[0]), [float(x) for x in row[1:5]]))
    pred = [x.tolist() if hasattr(x, "tolist") else x for x in result.get("bbox", [])]
    if not gt or not pred:
        return {}
    matrix = np.asarray([[_iou_xywh(gbox, pbox) for pbox in pred] for _, gbox in gt])
    rr, cc = linear_sum_assignment(1.0 - matrix)
    return {int(c): (gt[int(r)][0], float(matrix[int(r), int(c)]))
            for r, c in zip(rr, cc) if matrix[int(r), int(c)] >= threshold}


def _write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_model(config_path: Path, checkpoint: Path):
    cfg = yaml_to_dict(str(config_path))
    cfg["INFERENCE_DTYPE"] = "FP32"
    model, _ = build(cfg)
    load_checkpoint(model, str(checkpoint))
    model.cuda().eval()
    return cfg, model


def evaluate_variant(label: str, config_path: Path, checkpoint: Path, data_root: Path,
                     names: list[str], output_root: Path, max_frames=None,
                     diagnostic_mode: str = "normal"):
    variant_dir = output_root / label
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "tracker").mkdir(parents=True, exist_ok=True)
    cfg, model = _load_model(config_path, checkpoint)
    wrapped = get_model(model)
    decoder = wrapped.id_decoder
    visual_trail = wrapped.visual_trail
    frame_rows, audit_rows, sequence_perf = [], [], []
    owner_by_slot = {}
    write_stats = {"committed": 0, "matched_committed": 0, "correct_committed": 0,
                   "attempted": 0, "pending": 0, "rejected": 0}
    confidence_stats = {"committed": [], "pending": [], "rejected": []}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for sequence in names:
            # RuntimeTracker starts a fresh slot bank per sequence; ownership
            # audit state must follow the same lifecycle and never leak across
            # validation videos.
            owner_by_slot.clear()
            dataset = DanceTrack(data_root=str(data_root), split="val", load_annotation=False)
            seq_info = dataset.sequence_infos[sequence]
            sequence_root = data_root / "DanceTrack" / "val" / sequence
            image_paths = [str(sequence_root / "img1" / f"{index + 1:08d}.jpg")
                           for index in range(seq_info["length"])]
            if max_frames is not None:
                image_paths = image_paths[:max_frames]
            sequence_dataset = SeqDataset(
                seq_info=seq_info, image_paths=image_paths,
                max_shorter=cfg.get("INFERENCE_MAX_SHORTER", 800),
                max_longer=cfg.get("INFERENCE_MAX_LONGER", 1440),
                size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0), dtype=torch.float32,
            )
            loader = DataLoader(sequence_dataset, batch_size=1, shuffle=False, num_workers=2,
                                pin_memory=True, collate_fn=lambda x: x[0])
            gt_frames = _gt_by_frame(sequence_root / "gt" / "gt.txt", len(loader))
            tracker = RuntimeTracker(
                model=model, sequence_hw=sequence_dataset.seq_hw(),
                use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
                assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "object-max"),
                miss_tolerance=cfg.get("MISS_TOLERANCE", 30),
                det_thresh=cfg.get("DET_THRESH", 0.3),
                newborn_thresh=cfg.get("NEWBORN_THRESH", 0.6),
                id_thresh=cfg.get("ID_THRESH", 0.2),
                area_thresh=cfg.get("AREA_THRESH", 0),
                only_detr=cfg.get("INFERENCE_ONLY_DETR", False), dtype=torch.float32,
                trail_diagnostic_mode=diagnostic_mode,
            )
            sequence_results = []
            begin = None
            if len(loader) > 10:
                torch.cuda.synchronize()
            for frame, (image, _image_path) in enumerate(loader, 1):
                image.tensors = image.tensors.cuda(non_blocking=True)
                image.mask = image.mask.cuda(non_blocking=True)
                if frame == 11:
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                tracker.update(image=image, frame_index=frame, gt_frame=gt_frames[frame - 1])
                result = tracker.get_track_results()
                sequence_results.append(result)
                stats = dict(result.get("trail_memory_stats", {}))
                status = result.get("trail_write_status", [])
                confidences = result.get("id_confidence", torch.empty(0))
                slot_ids = result.get("slot_id", torch.empty(0, dtype=torch.long))
                matches = _matches(gt_frames[frame - 1], result)
                released = tracker.last_trail_update.get("released_slots", [])
                for slot in released:
                    owner_by_slot.pop(int(slot), None)
                for det_idx, slot in enumerate(slot_ids.tolist() if hasattr(slot_ids, "tolist") else []):
                    state = status[det_idx] if det_idx < len(status) else "not_attempted"
                    conf = float(confidences[det_idx].item()) if det_idx < len(confidences) else 0.0
                    if state == "committed":
                        write_stats["committed"] += 1
                        confidence_stats["committed"].append(conf)
                        if det_idx in matches:
                            gt_id, _iou = matches[det_idx]
                            write_stats["matched_committed"] += 1
                            previous_owner = owner_by_slot.get(int(slot))
                            if previous_owner is None or previous_owner == gt_id:
                                write_stats["correct_committed"] += 1
                            if previous_owner is None:
                                owner_by_slot[int(slot)] = gt_id
                    elif state == "pending":
                        write_stats["pending"] += 1
                        confidence_stats["pending"].append(conf)
                    elif state.startswith("rejected"):
                        write_stats["rejected"] += 1
                        confidence_stats["rejected"].append(conf)
                    if state != "not_attempted":
                        write_stats["attempted"] += 1
                d = {
                    "variant": label, "sequence": sequence, "frame": frame,
                    "trail_magnitude": float(getattr(visual_trail, "stats", {}).get("trail_magnitude", 0.0)),
                    "trail_token_norm": float(getattr(visual_trail, "stats", {}).get("trail_token_norm", 0.0)),
                    "trail_active": int(stats.get("active_persistent_trails", 0)),
                    "gate_mean": float(getattr(visual_trail, "stats", {}).get("trail_gate_mean", 0.0)),
                    "gate_std": float(getattr(visual_trail, "stats", {}).get("trail_gate_std", 0.0)),
                    "gate_min": float(getattr(visual_trail, "stats", {}).get("trail_gate_min", 0.0)),
                    "gate_max": float(getattr(visual_trail, "stats", {}).get("trail_gate_max", 0.0)),
                    "trail_attention_mass": float(getattr(decoder, "trail_attention_mass", 0.0)),
                    "baseline_logit_norm": float(getattr(decoder, "baseline_logit_norm", 0.0)),
                    "trail_logit_norm": float(getattr(decoder, "trail_logit_norm", 0.0)),
                    "effective_trail_logit_norm": float(getattr(decoder, "effective_trail_logit_norm", 0.0)),
                    "residual_scale": float(getattr(getattr(decoder, "residual_scale", None), "item", lambda: 0.0)()),
                    "finite": int(all(np.isfinite(float(x)) for x in (
                        getattr(visual_trail, "stats", {}).get("trail_magnitude", 0.0),
                        getattr(visual_trail, "stats", {}).get("trail_token_norm", 0.0),
                        getattr(decoder, "baseline_logit_norm", 0.0),
                        getattr(decoder, "trail_logit_norm", 0.0),
                        getattr(decoder, "effective_trail_logit_norm", 0.0)))),
                    **{f"memory_{k}": int(v) for k, v in stats.items()},
                }
                frame_rows.append(d)
                for item in tracker.audit_rows:
                    audit_rows.append(dict(variant=label, sequence=sequence, **item))
                tracker.audit_rows = []
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin if begin is not None else 0.0
            timed = max(0, len(loader) - 10)
            sequence_perf.append({"sequence": sequence, "frames": len(loader),
                                  "timed_frames": timed, "fps": timed / elapsed if elapsed else 0.0})
            _write_tracker(variant_dir / "tracker" / f"{sequence}.txt", sequence_results)
    finally:
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        del model
        torch.cuda.empty_cache()
    metrics, eval_cmd = _eval_variant(variant_dir, data_root, names)
    timed = sum(x["timed_frames"] for x in sequence_perf)
    seconds = sum(x["timed_frames"] / x["fps"] for x in sequence_perf if x["fps"])
    precision = (write_stats["correct_committed"] / write_stats["matched_committed"]
                 if write_stats["matched_committed"] else 0.0)
    confidence_summary = {
        key: {"count": len(values), "mean": float(np.mean(values)) if values else 0.0,
              "std": float(np.std(values)) if values else 0.0,
              "min": float(np.min(values)) if values else 0.0,
              "max": float(np.max(values)) if values else 0.0}
        for key, values in confidence_stats.items()
    }
    summary = {
        "variant": label, "checkpoint": str(checkpoint), "config": str(config_path),
        "metrics": {k: metrics.get(k) for k in ("HOTA", "DetA", "AssA", "IDF1", "IDSW")},
        "FPS": timed / seconds if seconds else 0.0, "peak_vram_mib": peak,
        "sequences": sequence_perf, "frames": sum(x["frames"] for x in sequence_perf),
        "write_stats": write_stats, "persistent_memory_ownership_precision": precision,
        "confidence_summary": confidence_summary, "eval_command": eval_cmd,
    }
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (variant_dir / "frame_diagnostics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in frame_rows))
    _write_csv(variant_dir / "identity_alignment_audit.csv", audit_rows)
    return summary, frame_rows, audit_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--val-manifest", type=Path,
                        default=Path("configs/trail_v1/subsets/decision_2day_val.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/research_v2/evaluation"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--diagnostic-mode", choices=["normal", "off", "oracle_update", "shuffled"],
                        default="normal")
    parser.add_argument("--variant", action="append", nargs=3, metavar=("LABEL", "CONFIG", "CHECKPOINT"),
                        help="Repeat: LABEL CONFIG CHECKPOINT")
    args = parser.parse_args()
    if not args.variant:
        parser.error("at least one --variant LABEL CONFIG CHECKPOINT is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    names = read_sequence_manifest(args.val_manifest)
    summaries = []
    audits = []
    all_frames = []
    for label, config, checkpoint in args.variant:
        summary, frames, audit = evaluate_variant(
            label, Path(config), Path(checkpoint), args.data_root, names, args.output_root,
            args.max_frames, args.diagnostic_mode)
        summaries.append(summary)
        audits.extend(audit)
        all_frames.extend(frames)
        print(json.dumps(summary, sort_keys=True), flush=True)
    _write_csv(args.output_root / "diagnostic_results.csv", [
        {"variant": s["variant"], **s["metrics"], "FPS": s["FPS"],
         "VRAM": s["peak_vram_mib"],
         "ownership_precision": s["persistent_memory_ownership_precision"]}
        for s in summaries])
    _write_csv(args.output_root / "identity_alignment_audit.csv", audits)
    (args.output_root / "diagnostic_summaries.json").write_text(
        json.dumps(summaries, indent=2) + "\n")
    print(f"Saved V2 evaluation under {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
