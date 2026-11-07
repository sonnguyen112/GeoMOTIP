#!/usr/bin/env python3
"""Verify existing clean DanceTrack artifacts; never launch training."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent / 'MOTIP'
STUDY = RESEARCH / 'outputs/research_data_efficiency/percent_5_10_15_s42'
MULTI = RESEARCH / 'outputs/research_data_efficiency/percent_5_10_15_multiseed'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    cancelled = ROOT / '_cancelled_mot17'
    cancelled.mkdir(exist_ok=True)
    old = ROOT / 'experiments/protocol.json'
    if old.exists() and not (cancelled/'protocol.json').exists():
        old.rename(cancelled/'protocol.json')
        for name in ['e0.yaml', 'v3.yaml', 'train.txt', 'val.txt']:
            path = ROOT/'configs'/name
            if path.exists():
                path.rename(cancelled/name)
    for name in ['report', 'presentation', 'demo', 'figures', 'sources/artifacts', 'experiments/manifests']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    (ROOT/'data/DanceTrack').symlink_to(RESEARCH/'datasets/DanceTrack', target_is_directory=True) if not (ROOT/'data/DanceTrack').exists() else None
    rows = list(csv.DictReader((MULTI/'selected_best_fullval_metrics_all_seeds.csv').open()))
    assert len(rows) == 18
    verified = []
    for row in rows:
        source = RESEARCH/row['source']
        lines = source.read_text().splitlines()
        summary = dict(zip(lines[0].split(), map(float, lines[1].split())))
        for metric in ['HOTA','DetA','AssA','IDF1','IDSW']:
            assert abs(summary[metric]-float(row[metric])) < .00051, (source, metric)
        target = ROOT/'sources/artifacts'/f"p{row['budget_pct']}_s{row['seed']}_{row['model']}_epoch{row['epoch']}_summary.txt"
        shutil.copy2(source, target)
        verified.append(dict(source=str(source), sha256=sha(source), local_copy=str(target.relative_to(ROOT))))
    shutil.copy2(MULTI/'selected_best_fullval_metrics_all_seeds.csv', ROOT/'experiments/metrics_all_seeds.csv')
    shutil.copy2(MULTI/'mean_std_summary.csv', ROOT/'experiments/mean_std_summary.csv')
    shutil.copy2(STUDY/'comparison_best/best_checkpoint_metrics.csv', ROOT/'experiments/exploratory_best_fullval_seed42.csv')
    for budget in [5, 10, 15]:
        manifest = STUDY/f'train_{budget}pct.txt'
        shutil.copy2(manifest, ROOT/'experiments/manifests'/manifest.name)
        pretrain = STUDY/f'geometry_pretrain_gt_linked/p{budget}'
        protocol = json.loads((pretrain/'probe/protocol.json').read_text())
        cache = json.loads((pretrain/'predicted_cache/metadata.json').read_text())
        assert protocol['validation_used'] is False and cache['validation_used'] is False
        assert cache['association_model_called'] is False and cache['gt_linked_predicted_history'] is True
        assert set(protocol['train_sequences']) == set(manifest.read_text().split())
        assert set(cache['sequences']) == set(manifest.read_text().split())
        assert cache['train_manifest_sha256'] == sha(manifest)
        shutil.copy2(pretrain/'probe/protocol.json', ROOT/'sources/artifacts'/f'g4_pretrain_p{budget}.json')
        shutil.copy2(pretrain/'predicted_cache/metadata.json', ROOT/'sources/artifacts'/f'g4_cache_p{budget}.json')
    shutil.copy2(STUDY/'full_dancetrack_val.txt', ROOT/'experiments/manifests/val_full.txt')
    all_val = set((ROOT/'experiments/manifests/val_full.txt').read_text().split())
    assert len(all_val) == 25
    for budget in [5, 10, 15]:
        assert not all_val & set((ROOT/f'experiments/manifests/train_{budget}pct.txt').read_text().split())
    demo_checkpoints = {}
    for model, epoch, subdir in [('e0',6,'e0'), ('v3',9,'v3_gtlinked')]:
        run = STUDY/'runs/p15'/subdir
        source = run/f'checkpoint_{epoch}.pth'
        target = ROOT/'checkpoints'/f'{model}_15pct_epoch{epoch}.pth'
        if not target.exists():
            if model == 'e0' and (ROOT/'checkpoints/motip_dancetrack_init.pth').exists():
                (ROOT/'checkpoints/motip_dancetrack_init.pth').rename(target)
            else:
                shutil.copy2(source,target)
        assert sha(source) == sha(target)
        cfg = yaml.safe_load((run/'train/config.yaml').read_text())
        cfg.update(SUPER_CONFIG_PATH=None, DATA_ROOT=str(ROOT/'data'),
            DETR_PRETRAIN=str(ROOT/'checkpoints/detector_pretrained.pth'),
            INIT_CHECKPOINT=None, RESUME_MODEL=None,
            TRAIN_SEQUENCE_MANIFEST=str(ROOT/'experiments/manifests/train_15pct.txt'),
            INFERENCE_SEQUENCE_MANIFEST=str(ROOT/'experiments/manifests/val_full.txt'),
            INFERENCE_MODEL=str(target), OUTPUTS_DIR=str(ROOT/f'experiments/{model}'),
            SAVE_BRANCH_LOGIT_CACHE=False, TRACKLET_CONFIRM_N_INIT=1,
            HISTORY_COMMIT_GATE_MODEL=None, USE_WANDB=False)
        if model == 'v3':
            stats = ROOT/'checkpoints/g4_train_only_stats.pt'
            shutil.copy2(STUDY/'geometry_pretrain_gt_linked/p15/probe/geometry_probe_models.pt',stats)
            cfg['GEOMETRY_INPUT_STATS_PATH'] = str(stats)
        else:
            cfg['GEOMETRY_INPUT_STATS_PATH'] = None
        (ROOT/f'configs/{model}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=True))
        demo_checkpoints[model] = dict(path=str(target), sha256=sha(target), original=str(source), epoch=epoch,
            selection='Highest AssA on the locked 3,894-frame small validation; NOT highest full-val metric')
        shutil.copy2(run/'train/log.txt', ROOT/'sources/artifacts'/f'{model}_15pct_training_log.txt')
    shutil.copy2(MULTI/'protocol.json', ROOT/'sources/artifacts/original_multiseed_protocol.json')
    shutil.copy2(STUDY/'subset_protocol.json', ROOT/'sources/artifacts/original_subset_protocol.json')
    protocol = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        change='User cancelled MOT17 training; use verified existing DanceTrack experiments only. No new training started.',
        dataset='DanceTrack', train_total_frames=41796, train_budgets={'5':2090,'10':4179,'15':6265},
        evaluation_frames=25508, evaluation_sequences=25,
        primary_result='Three-seed selected-best by small-validation AssA, then evaluated on full validation',
        selection_frames=3894, selection_caveat='Small checkpoint-selection validation is contained in full validation; the latter is NOT an untouched test set.',
        seeds=[42,123,3407], epochs=10, std_ddof=1,
        source_metrics=str(MULTI/'selected_best_fullval_metrics_all_seeds.csv'), verified_summaries=verified,
        demo_checkpoints=demo_checkpoints,
        demo_sequence='dancetrack0004', demo_selection='First sequence lexicographically in validation, not chosen for performance',
        improvements='Candidate-wise G4 geometry, 66D exact lags 1/2/4/8, 25,217-parameter MLP, train-only GT-linked predicted-box pretraining; Tb=2,Tg=0.5,lambda=1',
        excluded='Old 2,088-frame and 9,547-frame legacy geometry studies and NEW-omitting candidate probes are not evidence for this course report',
        cancelled_mot17='Unused downloaded material retained locally, excluded from submission package',
        no_new_training=True)
    (ROOT/'experiments/protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(verified_summaries=len(verified), checkpoints=demo_checkpoints, dataset='DanceTrack'),indent=2))


if __name__ == '__main__':
    main()
