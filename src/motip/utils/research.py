"""Self-contained provenance for research runs; never substitutes for metrics."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

import torch


def _source_provenance(config):
    """Collect real Git metadata, or explicit locked snapshot provenance.

    Deployment snapshots need not contain .git. Never invent a commit or
    initialize a fake repository to make logging succeed: retain the source
    commit from the verified protocol and label how it was obtained.
    """
    git = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    if git.returncode == 0:
        candidates = subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard", "--", "*.py", "*.yaml", "*.md"],
            text=True,
        ).splitlines()
        candidates = sorted(set(x for x in candidates if not x.startswith(("outputs/", "datasets", ".trail_motip"))))
        return {
            "git_commit": git.stdout.strip(),
            "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
            "git_metadata_source": "live_repository", "git_available": True,
        }, candidates

    protocol_name = config.get("SOURCE_PROVENANCE_PATH") or os.environ.get("MOTIP_SOURCE_PROTOCOL")
    if not protocol_name:
        raise RuntimeError(
            "No Git repository at the deployed source. Set SOURCE_PROVENANCE_PATH "
            "or MOTIP_SOURCE_PROTOCOL to a locked protocol containing git_commit and source_hashes."
        )
    protocol_path = Path(protocol_name).resolve(strict=True)
    protocol = json.loads(protocol_path.read_text())
    commit = protocol.get("git_commit")
    expected = protocol.get("source_hashes")
    if not isinstance(commit, str) or not commit or not isinstance(expected, dict) or not expected:
        raise ValueError("Snapshot protocol must contain git_commit and nonempty source_hashes")
    source_root = Path.cwd().resolve()
    for name, digest in expected.items():
        candidate = (source_root / name).resolve(strict=True)
        if not candidate.is_relative_to(source_root):
            raise ValueError(f"Locked source path escapes deployment snapshot: {name}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Locked deployed source hash mismatch: {name}")

    # Explicit source-only roots avoid following shared outputs/dataset links
    # or copying credentials into the source archive.
    files = list(source_root.glob("*.py"))
    for folder in ("models", "data", "configs", "structures", "utils", "log", "tools", "tests", "TrackEval"):
        directory = source_root / folder
        if directory.is_dir() and not directory.is_symlink():
            files.extend(p for p in directory.rglob("*") if p.suffix in {".py", ".yaml", ".yml"})
    candidates = sorted({str(p.relative_to(source_root)) for p in files
                         if p.is_file() and not p.is_symlink()
                         and "__pycache__" not in p.parts
                         and p.resolve().is_relative_to(source_root)})
    return {
        "git_commit": commit, "dirty_worktree": protocol.get("dirty_worktree"),
        "git_metadata_source": "locked_snapshot_protocol", "git_available": False,
        "source_protocol": str(protocol_path),
        "source_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "verified_locked_source_files": len(expected),
    }, candidates


def record_manifest(output_dir, config, model, accelerator, states=None, complete=False):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        provenance, candidates = _source_provenance(config)
        hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in candidates if Path(p).is_file()}
        with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
            for p in hashes:
                archive.add(p, arcname=p)
        (root / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(), **provenance,
            "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "training_command": shlex.join(sys.orig_argv), "config": config,
            "seed": config["SEED"], "data_root": str(Path(config["DATA_ROOT"]).resolve()),
            "dataset_split": list(zip(config["DATASETS"], config["DATASET_SPLITS"])),
            "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(accelerator.device) if torch.cuda.is_available() else "CPU",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "mixed_precision": accelerator.mixed_precision,
            "evaluation_command": f"python submit_and_evaluate.py --config-path {root}/train/config.yaml --inference-model {root}/checkpoint_{config['EPOCHS']-1}.pth --outputs-dir {root} --inference-mode evaluate",
        }
        manifest["training_started_utc"] = manifest["created_utc"]
        protocol_path = Path("outputs/research_v1/full_decision_protocol.json")
        if os.environ.get("TRAIL_DECISION_RECORD") and protocol_path.exists():
            protocol = json.loads(protocol_path.read_text())
            manifest.update(verified_dataset=protocol["dataset"], protocol_path=str(protocol_path),
                            primary_checkpoint=protocol["primary_checkpoint"])
    manifest.update(updated_utc=datetime.now(timezone.utc).isoformat(), states=states,
                    status="complete" if complete else "running",
                    peak_vram_mib=torch.cuda.max_memory_allocated(accelerator.device) / 2**20 if torch.cuda.is_available() else 0,
                    checkpoint_path=str(root / f"checkpoint_{(states or {}).get('start_epoch', 0)-1}.pth") if states else None)
    if complete:
        manifest["training_ended_utc"] = manifest["updated_utc"]
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    temporary.replace(path)
