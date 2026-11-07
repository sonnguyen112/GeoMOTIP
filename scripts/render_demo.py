#!/usr/bin/env python3
"""Render a captioned demo from fresh predictions, not GT-assisted tracking."""
import json
from pathlib import Path
from collections import defaultdict, deque
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
W,H,FPS=1920,1080,20
INK='#15304C';TEAL='#008D91';BLUE='#3C6A96';LIGHT='#EDF3F7'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)


def title_card(title,lines):
    out=Image.new('RGB',(W,H),INK);d=ImageDraw.Draw(out)
    d.rectangle((90,155,105,895),fill=TEAL)
    d.text((145,175),'THỊ GIÁC MÁY TÍNH · ĐỒ ÁN CUỐI KỲ',font=font(29,True),fill='#72D9D0')
    for i,line in enumerate(title.split('\n')):d.text((145,275+i*85),line,font=font(62,True),fill='white')
    for i,line in enumerate(lines):d.text((150,590+i*65),line,font=font(30),fill='#E4EDF7')
    return out


def color(identity):
    # Stable colors within each tracker, not GT-person correspondence.
    palette=[(0,180,160),(247,145,70),(78,149,225),(214,104,190),(177,178,55),(105,190,230),(231,91,83),(136,112,215)]
    return palette[identity%len(palette)]


def panel(original,record,history):
    im=original.resize((944,531));d=ImageDraw.Draw(im)
    sx,sy=944/original.width,531/original.height
    for identity,box in zip(record['ids'],record['boxes']):
        x,y,w,h=box;x*=sx;y*=sy;w*=sx;h*=sy
        rgb=color(identity);center=(x+w/2,y+h/2)
        history[identity].append(center)
        if len(history[identity])>=2:d.line(list(history[identity]),fill=rgb,width=2)
        d.rectangle((x,y,x+w,y+h),outline=rgb,width=3)
        label=f'ID {identity}';tw=d.textlength(label,font=font(18,True))
        lx=max(0,min(x,944-tw-12));ly=max(0,y-26)
        d.rectangle((lx,ly,lx+tw+12,ly+25),fill=rgb)
        d.text((lx+6,ly+1),label,font=font(18,True),fill='white')
    return im


def comparison(original,frame,records,meta,histories,caption,sequence='dancetrack0004'):
    out=Image.new('RGB',(W,H),'#F6F8FB');d=ImageDraw.Draw(out)
    d.rectangle((0,0,W,122),fill=INK)
    d.text((40,25),'MOTIP nền  ↔  MOTIP + G4',font=font(42,True),fill='white')
    d.text((44,84),f'DanceTrack validation · {sequence} · train 15% · seed 42',font=font(24),fill='#C9DCEB')
    for i,(name,label) in enumerate([('e0','MOTIP nền'),('v3','MOTIP + G4')]):
        x=8+i*960
        d.text((x+20,147),label,font=font(31,True),fill=BLUE if i==0 else TEAL)
        d.text((x+20,194),f"Suy luận: {meta[name]['fps_steady']:.2f} FPS · không dùng GT",font=font(22),fill=INK)
        out.paste(panel(original,records[name][frame-1],histories[name]),(x,245))
        d.text((x+20,795),f"Frame {frame:04d}/{len(records[name])}   |   {len(records[name][frame-1]['ids'])} track hiện tại",font=font(23),fill=INK)
    d.rounded_rectangle((28,855,1892,1008),radius=18,fill=LIGHT)
    for j,line in enumerate(caption):d.text((55,876+j*45),line,font=font(28 if j==0 else 22,j==0),fill=INK)
    d.text((45,1032),'Phát lại 20 FPS · màu/ID độc lập ở mỗi tracker · đường nối tâm chỉ để minh họa quỹ đạo',font=font(20),fill=BLUE)
    return out


def main():
    destination=ROOT/'demo/DEMO_MOTIP_DanceTrack.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    written=0
    def emit(image,count=1):
        nonlocal written
        payload=image.tobytes()
        for _ in range(count):proc.stdin.write(payload);written+=1
    # Fixed windows, not selected by tracking success/failure. No introduction,
    # outro or freeze frames: 20 + 25 seconds = 45 seconds of actual tracking.
    segments=[('inference','dancetrack0004',201,600),
              ('inference_crowded','dancetrack0034',201,700)]
    all_meta={}
    for folder,sequence,start,end in segments:
        records={name:json.loads((ROOT/f'demo/{folder}/{name}/predictions.json').read_text()) for name in ['e0','v3']}
        meta={name:json.loads((ROOT/f'demo/{folder}/{name}/complete.json').read_text()) for name in records}
        assert len(records['e0'])==len(records['v3'])>=end
        assert all(not m['gt_read'] for m in meta.values())
        all_meta[sequence]=meta
        histories={name:defaultdict(lambda:deque(maxlen=40)) for name in records}
        # Warm up display trails using only preceding predictions.
        for name in records:
            for record in records[name][max(0,start-41):start-1]:
                for identity,(x,y,w,h) in zip(record['ids'],record['boxes']):
                    # Raw DanceTrack frames in these scenes are 1920 x 1080.
                    histories[name][identity].append(((x+w/2)*944/1920,(y+h/2)*531/1080))
        caption=['Cùng video, cùng detector; MOTIP nền bên trái — MOTIP + G4 bên phải.',
                 'Nhãn ID độc lập giữa hai tracker. Không dùng GT để liên kết.']
        for frame in range(start,end+1):
            path=ROOT/'data/DanceTrack/val'/sequence/'img1'/f'{frame:08d}.jpg'
            original=Image.open(path).convert('RGB')
            out=comparison(original,frame,records,meta,histories,caption,sequence=sequence)
            emit(out)
            if sequence=='dancetrack0034' and frame==461:out.save(ROOT/'demo/crowded_preview.png')
            if written%200==0:print(f'Rendered {written}/900',flush=True)
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('Video encoding failed')
    (ROOT/'demo/video_metadata.json').write_text(json.dumps(dict(
        file=destination.name,render_fps=FPS,frames=written,duration_seconds=written/FPS,
        source='Fresh frozen-checkpoint predictions from scripts/run_demo_inference.py',
        screen_capture=False,description='Rendered tracking visualization from actual inference, not GT annotations or screen recording',
        audio='No audio; Vietnamese captions included',inference=all_meta,
        segments=[dict(sequence=s,start_frame=a,end_frame=b) for _,s,a,b in segments],
        introduction=False,outro=False),indent=2,ensure_ascii=False)+'\n')
    assert written/FPS<60
    print(f'Video complete: {written/FPS:.2f} seconds',flush=True)


if __name__=='__main__':main()
