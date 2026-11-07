#!/usr/bin/env python3
"""Merge the frozen E0 checkpoint with a pretrained PRED-PRED geometry MLP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.util import load_super_config  # noqa: E402
from models.motip import build  # noqa: E402
from utils.misc import yaml_to_dict  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/trail_v3/geometry_v3_a.yaml"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("outputs/research_v1/full_E0_subset_s42/checkpoint_9.pth"))
    parser.add_argument("--geometry-probe", type=Path, default=Path("outputs/research_v3/predicted_geometry/pred_pred_mlp_offline_finetuned.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research_v3/short_tracker_fusion/t1_predpred_calibrated.pth"))
    args = parser.parse_args()
    cfg = yaml_to_dict(str(args.config))
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    cfg["USE_GEOMETRY"] = True
    cfg["USE_TRAIL"] = False
    cfg["GEOMETRY_FIXED_LAMBDA"] = 1.0
    cfg["GEOMETRY_LEARNED_LAMBDA"] = False
    model, _ = build(cfg)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(base, strict=False)
    allowed = tuple(k for k in missing if k.startswith("id_decoder.geometry_branch."))
    if unexpected or any(not key.startswith("id_decoder.geometry_branch.") for key in missing):
        raise RuntimeError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    probe = torch.load(args.geometry_probe, map_location="cpu", weights_only=True)
    decoder = model.id_decoder
    decoder.geometry_branch.mlp.load_state_dict({
        key.replace("net.", ""): value for key, value in probe["state_dict"].items()
    }, strict=True)
    decoder.geometry_branch.input_mean.copy_(probe["stats"][0])
    decoder.geometry_branch.input_std.copy_(probe["stats"][1].clamp_min(1e-3))
    combined = model.state_dict()
    changed_base_keys = [
        key for key, value in base.items()
        if key not in combined or not torch.equal(combined[key], value)
    ]
    if changed_base_keys:
        raise RuntimeError(
            "Geometry checkpoint merge changed pretrained MOTIP/detector weights: "
            f"{changed_base_keys[:8]}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": combined, "optimizer": None, "scheduler": None,
                "states": {"source_base": str(args.base_checkpoint), "source_geometry": str(args.geometry_probe),
                           "fixed_lambda": 1.0, "base_weights_exactly_preserved": True}}, args.output)
    print({"output": str(args.output), "missing_geometry_keys": len(allowed),
           "unexpected": len(unexpected), "base_weights_exactly_preserved": True})


if __name__ == "__main__":
    main()
