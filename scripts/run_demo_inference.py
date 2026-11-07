#!/usr/bin/env python3
"""Actual frozen-checkpoint inference for the demo; annotations never loaded."""
import argparse
from pathlib import Path
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT/'src/motip'
sys.path.insert(0,str(VENDOR))
sys.path.insert(0,str(VENDOR/'models/ops/build/lib.linux-x86_64-cpython-312'))

import torch
import yaml
from torch.utils.data import DataLoader
from data.dancetrack import DanceTrack
from data.seq_dataset import SeqDataset
from models.motip import build
from models.runtime_tracker import RuntimeTracker


def run(name, limit, sequence='dancetrack0004', output=None):
    destination=(output or ROOT/'demo/inference')/name
    destination.mkdir(parents=True,exist_ok=True)
    marker=destination/'complete.json'
    if marker.exists():
        return
    cfg=yaml.safe_load((ROOT/f'configs/{name}.yaml').read_text())
    # Relocate the exported project without retaining the author's abs paths.
    cfg['DATA_ROOT']=str(ROOT/'data')
    cfg['INFERENCE_MODEL']=str(ROOT/'checkpoints'/Path(cfg['INFERENCE_MODEL']).name)
    if cfg.get('GEOMETRY_INPUT_STATS_PATH'):
        cfg['GEOMETRY_INPUT_STATS_PATH']=str(ROOT/'checkpoints'/Path(cfg['GEOMETRY_INPUT_STATS_PATH']).name)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
    model,_=build(cfg)
    checkpoint=torch.load(cfg['INFERENCE_MODEL'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    model.cuda().eval().requires_grad_(False)
    dataset=DanceTrack(data_root=str(ROOT/'data'),split='val',load_annotation=False)
    paths=dataset.image_paths[sequence][:limit] if limit else dataset.image_paths[sequence]
    seq=SeqDataset(dataset.sequence_infos[sequence],paths,max_shorter=cfg['INFERENCE_MAX_SHORTER'],max_longer=cfg['INFERENCE_MAX_LONGER'])
    loader=DataLoader(seq,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,collate_fn=lambda x:x[0])
    tracker=RuntimeTracker(model=model,sequence_hw=seq.seq_hw(),use_sigmoid=cfg['USE_FOCAL_LOSS'],
        assignment_protocol=cfg['ASSIGNMENT_PROTOCOL'],miss_tolerance=cfg['MISS_TOLERANCE'],
        det_thresh=cfg['DET_THRESH'],newborn_thresh=cfg['NEWBORN_THRESH'],id_thresh=cfg['ID_THRESH'],area_thresh=cfg['AREA_THRESH'])
    records=[]
    start=time.monotonic()
    steady=None
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for t,(image,_) in enumerate(loader,1):
            image.tensors=image.tensors.cuda(non_blocking=True)
            image.mask=image.mask.cuda(non_blocking=True)
            tracker.update(image=image,frame_index=t)
            result=tracker.get_track_results()
            records.append(dict(frame=t,ids=result['id'].cpu().tolist(),boxes=result['bbox'].cpu().tolist(),scores=result['score'].cpu().tolist()))
            if t==10:
                torch.cuda.synchronize()
                steady=time.monotonic()
            if t==1 or t%100==0:
                print(f'{name} {t}/{len(loader)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    torch.cuda.synchronize()
    end=time.monotonic()
    lines=[f"{r['frame']},{identity},{box[0]},{box[1]},{box[2]},{box[3]},1,-1,-1,-1\n"
           for r in records for identity,box in zip(r['ids'],r['boxes'])]
    (destination/f'{sequence}.txt').write_text(''.join(lines))
    (destination/'predictions.json').write_text(json.dumps(records))
    meta=dict(model=name,sequence=sequence,frames=len(records),checkpoint=cfg['INFERENCE_MODEL'],
        fps_steady=(len(records)-10)/(end-steady) if steady else len(records)/(end-start),
        wall_seconds=end-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        gt_read=False,precision='FP32',visual_trace='Display-only predicted box centers, not model input')
    marker.write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)
    del tracker,model
    torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['e0','v3','both'],default='both')
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--sequence',default='dancetrack0004')
    parser.add_argument('--output',type=Path,default=ROOT/'demo/inference')
    args=parser.parse_args()
    for name in (['e0','v3'] if args.model=='both' else [args.model]):
        run(name,args.max_frames,args.sequence,args.output)
