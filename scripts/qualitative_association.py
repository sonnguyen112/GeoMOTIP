#!/usr/bin/env python3
"""GT-audited, explicitly selected positive association examples; no inference.

Analyze with the existing MOTIP environment (SciPy), render with .venv-docs.
The per-frame assignment follows TrackEval CLEAR: continue the previous match
when its IoU is valid, then maximize IoU; validate aggregate IDSW parity.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'experiments/qualitative_association.json'
SOURCES=[('dancetrack0004','inference'),('dancetrack0034','inference_crowded')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ious(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2])
    hi=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(-1)
    return inter/np.maximum(a[:,None,2:].prod(-1)+b[None,:,2:].prod(-1)-inter,1e-10)


def gt_frames(sequence,length):
    rows=np.loadtxt(ROOT/f'data/DanceTrack/val/{sequence}/gt/gt.txt',delimiter=',')
    result={t:{} for t in range(1,length+1)}
    for row in rows:
        if int(row[6])==1 and int(row[7])==1:
            result[int(row[0])][int(row[1])]=row[2:6].tolist()
    return result


def assign(records,gt):
    from scipy.optimize import linear_sum_assignment
    result={};prev={};last={};switches=[];ids=set();pred_ids=set();similarities=[]
    for record in records:
        t=record['frame'];g=list(gt[t]);p=record['ids'];ids.update(g);pred_ids.update(p)
        sim=ious(list(gt[t].values()),record['boxes']);similarities.append(sim)
        result[t]={}
        if not g or not p:continue
        score=np.array([[1000*(prev.get(gid)==pid) for pid in p] for gid in g])+sim
        score[sim<.5-np.finfo(float).eps]=0
        rr,cc=linear_sum_assignment(-score);next_prev={}
        for r,c in zip(rr,cc):
            if score[r,c]<=np.finfo(float).eps:continue
            gid,pid=g[r],p[c]
            if gid in last and last[gid]!=pid:switches.append(dict(frame=t,gt_id=gid,old=last[gid],new=pid))
            result[t][gid]=dict(id=int(pid),box=record['boxes'][c],iou=float(sim[r,c]))
            next_prev[gid]=pid;last[gid]=pid
        prev=next_prev
    # Check that assignment/event extraction agrees with the actual metric code.
    sys.path.insert(0,str(ROOT/'src/motip/TrackEval'))
    from trackeval.metrics import CLEAR
    gm={v:i for i,v in enumerate(sorted(ids))};pm={v:i for i,v in enumerate(sorted(pred_ids))}
    data=dict(num_gt_ids=len(gm),num_tracker_ids=len(pm),num_timesteps=len(records),
        num_gt_dets=sum(len(gt[t]) for t in gt),num_tracker_dets=sum(len(r['ids']) for r in records),
        gt_ids=[np.array([gm[g] for g in gt[r['frame']]],dtype=int) for r in records],
        tracker_ids=[np.array([pm[v] for v in r['ids']],dtype=int) for r in records],similarity_scores=similarities)
    metric=CLEAR(dict(THRESHOLD=.5,PRINT_CONFIG=False)).eval_sequence(data)
    assert int(metric['IDSW'])==len(switches),(metric['IDSW'],len(switches))
    return result,switches


def analyze():
    audits=[];selected=[]
    for sequence,folder in SOURCES:
        files={m:ROOT/f'demo/{folder}/{m}/predictions.json' for m in ['e0','v3']}
        records={m:json.loads(path.read_text()) for m,path in files.items()}
        length=len(records['e0']);assert length==len(records['v3'])
        gt=gt_frames(sequence,length)
        mappings={};events={}
        for m in ['e0','v3']:mappings[m],events[m]=assign(records[m],gt)
        pool=[]
        for event in events['e0']:
            t,gid=event['frame'],event['gt_id'];window=list(range(t-4,t+5))
            if t-4<1 or t+4>length:continue
            if any(gid not in mappings[m][f] for m in mappings for f in window):continue
            b=[mappings['e0'][f][gid]['id'] for f in window]
            g=[mappings['v3'][f][gid]['id'] for f in window]
            # Exactly one persistent baseline switch, stable geometry, no missing
            # target in either output; not merely a birth or unmatched detection.
            if len(set(b[:4]))!=1 or len(set(b[4:]))!=1 or b[3]==b[4] or len(set(g))!=1:continue
            quality=min(mappings[m][f][gid]['iou'] for m in mappings for f in window)
            if quality<.65:continue
            overlap=0;peer=None
            for f in window:
                others=[(k,v) for k,v in gt[f].items() if k!=gid]
                if not others:continue
                vals=ious([gt[f][gid]],[v for _,v in others])[0];j=int(vals.argmax())
                if float(vals[j])>overlap:overlap=float(vals[j]);peer=int(others[j][0])
            if overlap<.1:continue
            frames=[t-4,t-1,t,t+4]
            entry=dict(sequence=sequence,folder=folder,event_frame=t,gt_id=gid,peer_gt_id=peer,
                frames=frames,window=window,overlap_iou=overlap,min_match_iou=quality,
                baseline_ids=b,geometry_ids=g,
                observations=[dict(frame=f,gt_box=gt[f][gid],peer_box=gt[f].get(peer),
                    base=mappings['e0'][f][gid],geometry=mappings['v3'][f][gid]) for f in frames])
            pool.append(entry)
        pool.sort(key=lambda c:(-c['overlap_iou'],-c['min_match_iou'],c['event_frame'],c['gt_id']))
        audits.append(dict(sequence=sequence,frames=length,eligible_positive_events=len(pool),
            base_idsw=len(events['e0']),geometry_idsw=len(events['v3']),clear_parity=True,
            prediction_sha256={m:sha(path) for m,path in files.items()},
            gt_sha256=sha(ROOT/f'data/DanceTrack/val/{sequence}/gt/gt.txt')))
        if pool:selected.append(pool[0])
    blob=dict(purpose='Purposively selected positive illustrations, NOT a representative error study',
        selection='One per existing demo sequence: baseline persistent switch, geometry stable across 9 consecutive frames; target matched in both with IoU>=0.65; GT overlap>=0.1. Rank by overlap, match IoU, time, GT ID.',
        matching='TrackEval CLEAR continuity-prioritized Hungarian at IoU 0.5; aggregate IDSW parity asserted on identical raw GT/prediction cohort.',
        gt_usage='Post-hoc analysis only. Existing saved predictions are never modified.',
        sources=audits,cases=selected,new_training=False,new_inference=False)
    DEST.write_text(json.dumps(blob,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(sources=audits,cases=[{k:c[k] for k in ['sequence','event_frame','gt_id','frames','baseline_ids','geometry_ids','overlap_iou','min_match_iou']} for c in selected]),indent=2))


def render(slide_view=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    data=json.loads(DEST.read_text());assert data['cases'],'No verified examples'
    for idx,case in enumerate(data['cases'],1):
        obs=case['observations'][:3] if slide_view else case['observations'];bounds=[]
        if slide_view:case['slide_frames']=[o['frame'] for o in obs]
        for item in obs:
            for box in [item['gt_box'],item['peer_box'],item['base']['box'],item['geometry']['box']]:
                if box is not None:
                    x,y,w,h=box;bounds.extend([(x,y),(x+w,y+h)])
        points=np.asarray(bounds);lo=points.min(0)-[65,50];hi=points.max(0)+[65,50]
        sample=Image.open(ROOT/f'data/DanceTrack/val/{case["sequence"]}/img1/{case["frames"][0]:08d}.jpg')
        lo=np.maximum(lo,0);hi=np.minimum(hi,sample.size)
        if slide_view:
            # Upper-body crop lets three temporal snapshots stay readable on a
            # widescreen slide. The crop is identical across frames/methods.
            cy=float(np.median([o['gt_box'][1]+.30*o['gt_box'][3] for o in obs]))
            half=(hi[0]-lo[0])/4
            lo[1]=max(0,cy-half);hi[1]=min(sample.height,cy+half)
        crop=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
        case['slide_crop_xyxy' if slide_view else 'crop_xyxy']=crop
        ratio=(hi[0]-lo[0])/(hi[1]-lo[1]);width=14;panel_w=4.2 if slide_view else 3.05;panel_h=panel_w/ratio
        height=2*panel_h+1.45
        fig,axes=plt.subplots(2,len(obs),figsize=(width,height),layout='constrained')
        row_specs=[('MOTIP nền','base','#526C91'),('MOTIP + G4','geometry','#007E73')]
        for r,(name,key,color) in enumerate(row_specs):
            for c,item in enumerate(obs):
                ax=axes[r,c];frame=item['frame']
                path=ROOT/f'data/DanceTrack/val/{case["sequence"]}/img1/{frame:08d}.jpg'
                ax.imshow(Image.open(path));ax.set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]));ax.set_aspect('equal');ax.axis('off')
                box=item['gt_box'] if key is None else item[key]['box'];x,y,w,h=box
                ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#00E0C0',lw=2.1))
                gx,gy,gw,gh=item['gt_box']
                ax.add_patch(Rectangle((gx,gy),gw,gh,fill=False,edgecolor='white',lw=1.0,ls='--'))
                if item['peer_box'] is not None:
                    x2,y2,w2,h2=item['peer_box'];ax.add_patch(Rectangle((x2,y2),w2,h2,fill=False,edgecolor='#F8C56C',lw=1.0,ls='--'))
                ident=case['gt_id'] if key is None else item[key]['id']
                label=f'GT {ident}' if key is None else f'ID {ident}'
                ax.text(.035,.97,label,transform=ax.transAxes,ha='left',va='top',fontsize=26 if slide_view else 20,fontweight='bold',color='white',bbox=dict(facecolor='#18324A',edgecolor='none',alpha=.92,pad=3))
                if r==0:ax.set_title(f'Frame {frame}',fontsize=22 if slide_view else 17,color='#18324A',pad=9)
                if c==0:ax.text(-.05,.5,name,transform=ax.transAxes,rotation=90,ha='right',va='center',fontsize=20 if slide_view else 16,fontweight='bold',color=color)
        fig.suptitle(f'{case["sequence"]} · cùng đối tượng GT {case["gt_id"]}',fontsize=18,fontweight='bold',color='#18324A')
        stem=f'association_slide_case_{idx}' if slide_view else f'association_case_{idx}'
        for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'figures/{stem}.{ext}',dpi=260,bbox_inches='tight',pad_inches=.08)
        plt.close(fig)
    DEST.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print('Rendered',len(data['cases']),'verified positive examples')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analyze',action='store_true');parser.add_argument('--render',action='store_true');parser.add_argument('--render-slides',action='store_true');args=parser.parse_args()
    if args.analyze:analyze()
    if args.render:render()
    if args.render_slides:render(slide_view=True)
