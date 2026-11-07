"""Measurement only: no model, RNG, optimizer or scheduler changes."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record_epoch(root, epoch, train_metrics, eval_metrics, peak_vram, started, iterations):
    """Exact epoch means from Metrics; missing validation stays blank."""
    path = Path(root) / "metrics_by_epoch.csv"
    train = {k: v.global_average for k, v in train_metrics.metrics.items() if v.total_count}
    validation = {k: v.global_average for k, v in eval_metrics.metrics.items()} if eval_metrics else {}
    geometry_fields = (
        "geometry_aux_loss", "geometry_logit_norm", "geometry_base_logit_norm",
        "geometry_effective_logit_norm", "geometry_contribution_ratio",
        "geometry_logit_correlation", "geometry_margin", "geometry_lambda",
        "geometry_feature_abs_max", "geometry_raw_feature_abs_max",
        "geometry_target_candidate_recall", "geometry_existing_query_fraction",
        "geometry_newborn_query_fraction",
    )
    perf_path = Path(root) / f"train/eval_during_train/epoch_{epoch}/performance.json"
    perf = json.loads(perf_path.read_text()) if perf_path.exists() else {}
    row = dict(epoch=epoch, train_total_loss=train.get("loss"), train_id_loss=train.get("id_loss"),
               **{k: validation.get(k) for k in ("HOTA", "DetA", "AssA", "IDF1", "IDSW")},
               FPS=perf.get("FPS"), peak_VRAM=peak_vram,
               **{name: train.get(name) for name in geometry_fields},
               **{dest: train.get(src) for dest, src in {
                   "gate_mean": "trail_gate_mean", "gate_std": "trail_gate_std",
                   "gate_min": "trail_gate_min", "gate_max": "trail_gate_max",
                   "trail_magnitude": "trail_magnitude", "trail_token_norm": "trail_token_norm",
                   "trail_attention_mass": "trail_attention_mass", "number_active_trails": "trail_active",
               }.items()}, started_utc=started, ended_utc=utc_now(), iterations=iterations)
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    rows = [x for x in rows if int(x["epoch"]) != epoch] + [row]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: int(x["epoch"])))
    temporary.replace(path)


class IterationRecorder:
    def __init__(self, root, epoch):
        path = Path(root) / "iteration_timing.jsonl"
        self.handle = path.open("a", buffering=1)
        self.epoch = epoch

    def update(self, step, seconds, data_wait_seconds, metrics):
        row = dict(epoch=self.epoch, step=step, utc=utc_now(), seconds=seconds,
                   data_wait_seconds=data_wait_seconds,
                   loss=metrics["loss"].value_deque[-1], id_loss=metrics["id_loss"].value_deque[-1])
        self.handle.write(json.dumps(row, allow_nan=False) + "\n")

    def close(self):
        self.handle.close()
