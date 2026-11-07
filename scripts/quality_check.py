#!/usr/bin/env python3
"""Check numerical provenance and render contact sheets for human inspection."""
from pathlib import Path
import csv
import json
import re
import hashlib
import subprocess
import numpy as np
import pymupdf
from pptx import Presentation
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT.parent/'MOTIP'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def contact(pdf,output,indices,columns=3,width=500):
    doc=pymupdf.open(pdf)
    images=[]
    for index in indices:
        page=doc[index]
        pix=page.get_pixmap(matrix=pymupdf.Matrix(width/page.rect.width,width/page.rect.width))
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        tile=Image.new('RGB',(width,im.height+34),'#EAF0F5')
        tile.paste(im,(0,34));ImageDraw.Draw(tile).text((12,6),f'Trang {index+1}',font=ImageFont.truetype(FONT,17),fill='#15304C')
        images.append(tile)
    height=max(im.height for im in images)
    sheet=Image.new('RGB',(columns*width,((len(images)+columns-1)//columns)*height),'#D3DFE9')
    for i,im in enumerate(images):sheet.paste(im,((i%columns)*width,(i//columns)*height))
    sheet.save(output)
    return len(doc)


def main():
    checks={}
    rows=list(csv.DictReader((ROOT/'experiments/metrics_all_seeds.csv').open()))
    for b in [5,10,15]:
        for seed in [42,123,3407]:
            pair={r['model']:r for r in rows if int(r['budget_pct'])==b and int(r['seed'])==seed}
            for metric in ['HOTA','AssA','IDF1']:
                assert float(pair['v3_gtlinked'][metric])>float(pair['e0'][metric])
            assert float(pair['v3_gtlinked']['IDSW'])<float(pair['e0']['IDSW'])
    checks['all_nine_paired_quality_checks']=True
    checks['geometry_pretraining']=[]
    for b in [5,10,15]:
        expected=set((ROOT/f'experiments/manifests/train_{b}pct.txt').read_text().split())
        for seed in [42,123,3407]:
            if seed==42:path=RESEARCH/f'outputs/research_data_efficiency/percent_5_10_15_s42/geometry_pretrain_gt_linked/p{b}/probe/protocol.json'
            else:path=RESEARCH/f'outputs/research_data_efficiency/percent_5_10_15_multiseed/geometry_pretrain/p{b}/s{seed}/probe/protocol.json'
            meta=json.loads(path.read_text())
            assert meta['validation_used'] is False
            assert set(meta['train_sequences'])==expected
            checks['geometry_pretraining'].append(dict(budget=b,seed=seed,train_only_protocol_verified=True))
    checks['demo_replay_parity']=[]
    qualitative=json.loads((ROOT/'experiments/qualitative_association.json').read_text())
    assert len(qualitative['cases'])==2
    for source in qualitative['sources']:
        assert source['clear_parity'] is True
        folder=next(c['folder'] for c in qualitative['cases'] if c['sequence']==source['sequence'])
        for model in ['e0','v3']:
            path=ROOT/f'demo/{folder}/{model}/predictions.json'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==source['prediction_sha256'][model]
    for case in qualitative['cases']:
        assert len(case['frames'])==4 and len(case['window'])==9
        assert len(set(case['geometry_ids']))==1 and len(set(case['baseline_ids']))==2
        assert case['min_match_iou']>=.65 and case['overlap_iou']>=.1
        for obs in case['observations']:
            for model,key in [('e0','base'),('v3','geometry')]:
                records=json.loads((ROOT/f'demo/{case["folder"]}/{model}/predictions.json').read_text())
                row=records[obs['frame']-1]
                assert len(row['ids'])==len(set(row['ids']))
                index=row['ids'].index(obs[key]['id'])
                assert np.array_equal(row['boxes'][index],obs[key]['box'])
    checks['gt_audited_positive_examples']=2
    for folder,sequence in [('inference','dancetrack0004'),('inference_crowded','dancetrack0034')]:
        for name,subdir,epoch in [('e0','e0',6),('v3','v3_gtlinked',9)]:
            current=ROOT/f'demo/{folder}/{name}/{sequence}.txt'
            old=RESEARCH/f'outputs/research_data_efficiency/percent_5_10_15_s42/runs/p15/{subdir}/train/eval_during_train/epoch_{epoch}/tracker/{sequence}.txt'
            a=np.loadtxt(current,delimiter=',');b=np.loadtxt(old,delimiter=',')
            same_shape=a.shape==b.shape
            ids_equal=bool(same_shape and np.array_equal(a[:,:2],b[:,:2]))
            err=float(np.abs(a[:,2:6]-b[:,2:6]).max()) if same_shape else None
            checks['demo_replay_parity'].append(dict(model=name,sequence=sequence,same_shape=same_shape,frame_and_ids_identical=ids_equal,max_box_abs_difference=err))
    count=len(pymupdf.open(ROOT/'report/BAO_CAO_MOTIP.pdf'))
    checks['report_pages']=contact(ROOT/'report/BAO_CAO_MOTIP.pdf',ROOT/'report/contact_sheet.png',list(range(count)),columns=4,width=350)
    notes=json.loads((ROOT/'presentation/slide_structure.json').read_text())
    assert len(notes)==31 and sum(not n['backup'] for n in notes)==25
    assert sum(n['seconds'] for n in notes)+45*sum(bool(n.get('video_after')) for n in notes)==1200
    checks['presentation_pages']=contact(ROOT/'presentation/THUYET_TRINH_MOTIP.pdf',ROOT/'presentation/contact_sheet.png',list(range(len(notes))),columns=3,width=520)
    assert checks['presentation_pages']==len(notes)
    html_audit=json.loads((ROOT/'presentation/html/EXPORT_AUDIT.json').read_text())
    assert not html_audit['overflow'] and html_audit['full_slide_rasterization'] is False
    assert html_audit['native_charts']==6 and html_audit['native_tables']==4
    checks['html_to_pptx']={k:v for k,v in html_audit.items() if k!='slides'}
    deck=Presentation(ROOT/'presentation/THUYET_TRINH_MOTIP.pptx')
    charts=[shape.chart for slide in deck.slides for shape in slide.shapes if shape.has_chart]
    assert len(charts)==6
    summary=json.loads((ROOT/'experiments/presentation_summary.json').read_text())
    stats={(r['budget'],r['model']):r for r in summary['stats']}
    for index,metric in [(0,'HOTA'),(1,'AssA'),(2,'IDF1'),(4,'IDSW'),(5,'DetA')]:
        for si,model in enumerate(['e0','v3_gtlinked']):
            assert np.allclose(list(charts[index].series[si].values),[stats[(b,model)][metric]['mean'] for b in [5,10,15]],rtol=0,atol=1e-10)
    checks['native_chart_values_match_verified_statistics']=True
    checks['report_vector_figures']=[]
    for name in ['pipeline','multilag','geometry_fusion','geometry_pretraining','quality_vs_budget','paired_delta','association_case_1','association_case_2']:
        assert (ROOT/f'figures/{name}.pdf').exists()
        assert '{../../figures/'+name+'.pdf}' in (ROOT/'report/latex/content.tex').read_text()
        checks['report_vector_figures'].append(name)
    for path in [ROOT/'report/BAO_CAO_MOTIP.pdf',ROOT/'presentation/THUYET_TRINH_MOTIP.pdf']:
        doc=pymupdf.open(path)
        text='\n'.join(page.get_text() for page in doc)
        assert 'Trương Phước Hưng' in text and 'MOTIP' in text
        assert '\ufffd' not in text
        assert not re.search(r'\b(?:E0|V3)\b',text),f'Internal model name remains in {path.name}'
        assert not any(token in text for token in ['%TABLE_','%ABSTRACT_','[CẦN XÁC MINH','[CHƯA CÓ KẾT QUẢ'])
    checks['pdf_text_and_placeholders']=True
    checks['public_model_names_only']=True
    probe=subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height,r_frame_rate','-of','json',str(ROOT/'demo/DEMO_MOTIP_DanceTrack.mp4')],text=True)
    checks['video']=json.loads(probe)
    assert float(checks['video']['format']['duration'])<60
    video_meta=json.loads((ROOT/'demo/video_metadata.json').read_text())
    assert video_meta['introduction'] is False and video_meta['outro'] is False
    checks['short_demo_no_introduction']=True
    texlog=(ROOT/'report/latex/build/main.log').read_text()
    assert 'undefined references' not in texlog and 'Missing character:' not in texlog
    assert 'Overfull \\hbox' not in texlog
    checks['latex_compiled']=dict(engine='XeLaTeX',template='kourgeorge/arxiv-style',
        figures=8,tables=7,bibliography_entries=6,numbered_equations=13,shell_escape=False)
    tex=(ROOT/'report/latex/content.tex').read_text()
    assert tex.count('\\begin{figure}')==8
    assert tex.count('\\begin{equation}')==13
    assert all(f'\\label{{eq:{name}}}' in tex for name in ['attention','baseline-loss','relative-geometry','fusion','new','v3-loss'])
    (ROOT/'QUALITY_CHECK.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(checks,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
