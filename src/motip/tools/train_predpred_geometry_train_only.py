#!/usr/bin/env python3
"""Train a fixed 66-D PRED-PRED geometry scorer using official-train data only.

This reproduces the earlier GT-GT initialization followed by PRED-PRED
adaptation, but fits both stages and normalization statistics only on the
selected official-train sequences.  It deliberately has no validation split
or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sequence_manifest import read_sequence_manifest  # noqa: E402
from models.motip.geometry.geometry_features import build_g4_features  # noqa: E402


class GeometryProbe(nn.Module):
    def __init__(self, dim: int = 66, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _train_stage(
    model: GeometryProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    stats: torch.Tensor,
    args,
    device: torch.device,
    stage: str,
) -> list[dict]:
    normalized = ((features - stats[0]) / stats[1]).clamp(-8.0, 8.0)
    x_device, y_device = normalized.to(device), labels.to(device)
    positive = float(labels.sum().item())
    negative = float(len(labels) - positive)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / max(positive, 1.0), device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    stage_epochs = args.gtgt_epochs if stage == "GT_GT" else args.predpred_epochs
    for epoch in range(stage_epochs):
        model.train()
        order = torch.randperm(len(labels), generator=generator)
        losses = []
        for start in range(0, len(order), args.batch_size):
            take = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_device[take.to(device)]), y_device[take.to(device)])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {
            "stage": stage,
            "epoch": epoch + 1,
            "train_bce": float(np.mean(losses)),
            "train_pairs": int(len(labels)),
            "train_positive_pairs": int(positive),
            "train_negative_pairs": int(negative),
        }
        history.append(row)
        print(f"[train-only-geometry] {stage} epoch={epoch + 1}/{stage_epochs} "
              f"bce={row['train_bce']:.6f}", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=Path("outputs/research_v3/larger_dancetrack_subset/predicted_geometry_cache/train_cache.pt"))
    parser.add_argument("--cache-metadata", type=Path,
                        default=Path("outputs/research_v3/larger_dancetrack_subset/predicted_geometry_cache/metadata.json"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gtgt-epochs", type=int, default=4)
    parser.add_argument("--predpred-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-exact-train-only-cache", action="store_true",
        help="Reject caches that contain any sequence beyond this train manifest.",
    )
    parser.add_argument(
        "--require-gt-linked-cache", action="store_true",
        help="Reject caches that used trained MOTIP association to construct histories.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.gtgt_epochs <= 0 or args.predpred_epochs <= 0 or args.batch_size <= 0:
        raise ValueError("stage epochs and batch size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    allowed_sequences = set(read_sequence_manifest(args.train_manifest))
    cache_meta = json.loads(args.cache_metadata.read_text())
    cached_train_sequences = {
        name for name, info in cache_meta["sequences"].items()
        if info.get("split") == "train"
    }
    if args.require_exact_train_only_cache:
        if not cache_meta.get("train_only", False):
            raise ValueError("Strict pretraining requires a train-only cache")
        if set(cache_meta["sequences"]) != allowed_sequences:
            raise ValueError("Cache sequences must exactly match the train manifest")
    if args.require_gt_linked_cache:
        if not cache_meta.get("gt_linked_predicted_history", False):
            raise ValueError("Strict detector-only study requires GT-linked predicted boxes")
        if cache_meta.get("association_model_called") is not False:
            raise ValueError("Association model must not be called while caching")
    if not allowed_sequences or not allowed_sequences.issubset(cached_train_sequences):
        raise ValueError(
            f"Manifest sequences not contained in official-train cache: "
            f"{sorted(allowed_sequences - cached_train_sequences)}"
        )
    if any(cache_meta["sequences"][name].get("split") != "train" for name in allowed_sequences):
        raise AssertionError("Geometry pretraining must use official-train sequences only")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    sequences = np.asarray(cache["sequence"], dtype=object)
    selected_rows_np = np.isin(sequences, np.asarray(sorted(allowed_sequences), dtype=object))
    selected_rows = torch.from_numpy(selected_rows_np)
    pair_rows_all = cache["pair_row"].long()
    pair_keep = selected_rows[pair_rows_all]
    pair_keep &= cache["pair_lag_valid_pred"].bool().any(dim=-1)
    pair_indices = torch.where(pair_keep)[0]
    if pair_indices.numel() == 0:
        raise ValueError("No valid PRED-PRED candidate pairs in selected training sequences")

    pair_rows = pair_rows_all[pair_indices]
    current_ids = cache["current_gt"].long()[pair_rows]
    pair_ids = cache["pair_gt"].long()[pair_indices]
    labels = (pair_ids == current_ids).float()
    positive_count = torch.zeros(len(cache["current_gt"]), dtype=torch.long)
    positive_count.index_add_(0, pair_rows, labels.long())
    has_positive = positive_count[pair_rows] > 0
    pair_indices = pair_indices[has_positive]
    pair_rows = pair_rows[has_positive]
    labels = labels[has_positive]
    if pair_indices.numel() == 0 or not bool((labels > 0.5).any()):
        raise ValueError("No target-present candidate sets remain for train-only pretraining")

    pred_features = build_g4_features(
        cache["current_box_pred"].float()[pair_rows],
        cache["pair_lag_boxes_pred"].float()[pair_indices],
        cache["pair_lag_valid_pred"].bool()[pair_indices],
    ).float()
    gt_features = build_g4_features(
        cache["current_box_gt"].float()[pair_rows],
        cache["pair_lag_boxes_gt"].float()[pair_indices],
        cache["pair_lag_valid_gt"].bool()[pair_indices],
    ).float()
    if pred_features.ndim != 2 or pred_features.shape[1] != 66:
        raise AssertionError(f"Expected [pairs,66] G4 features, got {tuple(pred_features.shape)}")
    if not torch.isfinite(pred_features).all() or not torch.isfinite(gt_features).all():
        raise FloatingPointError("Nonfinite G4 training features")
    stats = torch.stack((gt_features.mean(0), gt_features.std(0).clamp_min(1e-3)))

    device = torch.device(args.device)
    model = GeometryProbe().to(device)
    gt_history = _train_stage(model, gt_features, labels, stats, args, device, "GT_GT")
    pred_history = _train_stage(model, pred_features, labels, stats, args, device, "PRED_PRED")
    history = gt_history + pred_history
    positive = float(labels.sum().item())
    negative = float(len(labels) - positive)

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output / "predpred_geometry_train_only.pt"
    probe_blob = {
        "state_dict": {f"net.{key}": value.detach().cpu() for key, value in model.net.state_dict().items()},
        "stats": stats.cpu(),
        "feature_dim": 66,
        "hidden_dim": 128,
        "source": "GT_GT_then_PRED_PRED_official_train_only_fixed_epochs",
        "train_sequences": sorted(allowed_sequences),
        "stage_epochs": {"GT_GT": args.gtgt_epochs, "PRED_PRED": args.predpred_epochs},
        "seed": args.seed,
    }
    torch.save(probe_blob, checkpoint_path)
    # MOTIP construction expects this small stats bundle.  It contains no
    # validation-derived values and will be overwritten by the combined state
    # dict when the per-budget V3 initializer is loaded.
    torch.save({"G4_full_geometry": {"stats": stats.cpu()}},
               args.output / "geometry_probe_models.pt")
    (args.output / "protocol.json").write_text(json.dumps({
        "cache": str(args.cache),
        "cache_metadata": str(args.cache_metadata),
        "train_manifest": str(args.train_manifest),
        "train_sequences": sorted(allowed_sequences),
        "train_frames": sum(cache_meta["sequences"][name]["frames"] for name in allowed_sequences),
        "cache_split_assertion": "all selected sequence metadata split == train",
        "feature_dim": 66,
        "feature_normalization": "per-dimension mean/std fitted on selected train GT-GT pairs only; std floor 1e-3; online-compatible clamp [-8,8]; same statistics used for PRED-PRED adaptation",
        "objective": "target-present candidate-pair BCEWithLogits with positive weight = negative_count / positive_count in each fixed training stage",
        "architecture": "66-128-GELU-128-GELU-1",
        "stage_epochs": {"GT_GT": args.gtgt_epochs, "PRED_PRED": args.predpred_epochs},
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "seed": args.seed,
        "validation_used": False,
        "selected_pairs": int(len(labels)),
        "positive_pairs": int(positive),
        "negative_pairs": int(negative),
        "target_present_rows": int(torch.unique(pair_rows).numel()),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }, indent=2) + "\n")
    _write_csv(args.output / "training_curve.csv", history)
    print(json.dumps({"output": str(args.output), **history[-1]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
