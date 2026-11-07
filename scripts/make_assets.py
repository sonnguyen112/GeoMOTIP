#!/usr/bin/env python3
"""Build figures and manuscript tables from verified, machine-readable runs."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'
INK='#15304C'; BLUE='#3C6A96'; TEAL='#008D91'; CORAL='#E48054'; LIGHT='#EDF3F7'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size,bold=False):
    return ImageFont.truetype(BOLD if bold else FONT,size)


def canvas(title,subtitle='',size=(1800,850)):
    image=Image.new('RGB',size,'white');d=ImageDraw.Draw(image)
    d.text((55,35),title,font=font(39,True),fill=INK)
    if subtitle:d.text((55,94),subtitle,font=font(24),fill=BLUE)
    return image,d


def box(d,xy,text,fill=LIGHT,outline=None,size=27):
    d.rounded_rectangle(xy,radius=18,fill=fill,outline=outline,width=3)
    lines=text.split('\n');lineheight=int(size*1.45)
    y=(xy[1]+xy[3])/2-len(lines)*lineheight/2
    for line in lines:
        w=d.textlength(line,font=font(size,True))
        d.text(((xy[0]+xy[2]-w)/2,y),line,font=font(size,True),fill=INK)
        y+=lineheight


def arrow(d,start,end,color=TEAL):
    d.line([start,end],fill=color,width=6)
    x,y=end
    if end[0]>start[0]:points=[(x,y),(x-16,y-10),(x-16,y+10)]
    elif end[0]<start[0]:points=[(x,y),(x+16,y-10),(x+16,y+10)]
    elif end[1]>start[1]:points=[(x,y),(x-10,y-16),(x+10,y-16)]
    else:points=[(x,y),(x-10,y+16),(x+10,y+16)]
    d.polygon(points,fill=color)


def math_label(image,name,center,max_width,max_height):
    """Composite an actual XeLaTeX equation, not Unicode pseudo-math."""
    item=Image.open(FIG/f'math/{name}.png').convert('RGBA')
    scale=min(max_width/item.width,max_height/item.height)
    item=item.resize((round(item.width*scale),round(item.height*scale)),Image.Resampling.LANCZOS)
    image.paste(item,(round(center[0]-item.width/2),round(center[1]-item.height/2)),item)


def diagrams():
    image,d=canvas('MOTIP: dự đoán ID theo ngữ cảnh','Detector tạo phát hiện; lịch sử có ID là ngữ cảnh cho liên kết.')
    nodes=[(50,240,290,420),(360,240,670,420),(740,240,1050,420),(1120,240,1420,420),(1490,240,1750,420)]
    labels=['','Deformable DETR\nHộp + đặc trưng','ID Decoder\nAttention lịch sử','Logit ID\nSlot 0…49 + NEW','Gán ID\nHộp + ID toàn cục']
    for xy,label in zip(nodes,labels):box(d,xy,label,size=25)
    math_label(image,'image',(170,330),130,85)
    for a,b in zip(nodes,nodes[1:]):arrow(d,(a[2]+8,330),(b[0]-8,330),BLUE)
    box(d,(600,550,1050,720),'Lịch sử trước t\nĐặc trưng + ID + timestamp',fill='#DFF2EE',size=25)
    arrow(d,(895,550),(895,435))
    d.line([(1620,435),(1620,790),(825,790),(825,735)],fill=TEAL,width=6)
    arrow(d,(825,780),(825,735))
    d.text((1110,740),'Ghi nhớ sau khi dự đoán',font=font(24),fill=TEAL)
    image.save(FIG/'pipeline.png')
    image,d=canvas('Lịch sử đa độ trễ: giữ đúng thứ tự thời gian','Ví dụ minh họa: lag thiếu có mask 0, không thay bằng quan sát gần nhất.')
    xs=[200,550,900,1250,1600];labels=['t−8','t−4','t−2','t−1','t']
    d.line([(180,390),(1630,390)],fill=BLUE,width=7)
    for i,(x,label) in enumerate(zip(xs,labels)):
        d.ellipse((x-16,374,x+16,406),fill=CORAL if i==1 else TEAL)
        d.text((x-46,430),label,font=font(40,True),fill=INK)
        box(d,(x-130,180,x+130,310),'Thiếu quan sát' if i==1 else ('Phát hiện\nhiện tại' if i==4 else 'Hộp quá khứ'),fill='#FAE9DF' if i==1 else LIGHT,size=25)
        d.text((x-90,510),'mask = 0' if i==1 else ('query' if i==4 else 'mask = 1'),font=font(27),fill=CORAL if i==1 else TEAL)
    box(d,(340,650,1460,780),'Chỉ dùng timestamp < t · Độ trễ chính xác {1, 2, 4, 8}',fill='#DFF2EE',size=29)
    image.save(FIG/'multilag.png')
    image,d=canvas('Nhánh geometry theo từng cặp phát hiện–track','Không thay ID Decoder; không dựng visual canvas; không chấm NEW bằng geometry.')
    box(d,(70,200,440,365),'Phát hiện hiện tại\n+ hộp lịch sử ứng viên',size=25)
    box(d,(520,200,790,365),'G4 · 66 chiều\nLag + mask',fill='#DFF2EE',size=27)
    box(d,(880,200,1220,365),'MLP\n66 → 128 → 128 → 1',fill='#DFF2EE',size=25)
    box(d,(1330,200,1720,365),'',fill='#DFF2EE',size=28)
    label='Điểm tương thích'
    d.text((1525-d.textlength(label,font=font(28,True))/2,230),label,font=font(28,True),fill=INK)
    math_label(image,'pair_score',(1525,315),230,65)
    for a,b in [(440,520),(790,880),(1220,1330)]:arrow(d,(a+10,280),(b-10,280))
    box(d,(120,530,640,690),'',fill=LIGHT,size=32)
    label='MOTIP nền'
    d.text((380-d.textlength(label,font=font(32,True))/2,550),label,font=font(32,True),fill=INK)
    math_label(image,'base_temperature',(380,640),270,75)
    box(d,(960,515,1680,720),'',fill='#E4EFF8',size=28)
    math_label(image,'fusion',(1320,590),650,130)
    math_label(image,'new_neutral',(1320,680),560,45)
    arrow(d,(660,610),(940,610),BLUE)
    arrow(d,(1520,385),(1520,495))
    image.save(FIG/'geometry_fusion.png')


def tracking_image(frame=600,sequence='dancetrack0004',inference='inference'):
    source=ROOT/f'data/DanceTrack/val/{sequence}/img1'/f'{frame:08d}.jpg'
    original=Image.open(source).convert('RGB')
    parts=[]
    for name,label in [('e0','MOTIP nền'),('v3','MOTIP + G4')]:
        records=json.loads((ROOT/f'demo/{inference}/{name}/predictions.json').read_text())
        record=records[frame-1]
        im=original.copy();d=ImageDraw.Draw(im)
        for identity,bbox in zip(record['ids'],record['boxes']):
            x,y,w,h=bbox
            color=tuple(int(v) for v in plt.get_cmap('tab20')((identity%20)/20)[:3]*np.array([255,255,255]))
            d.rectangle((x,y,x+w,y+h),outline=color,width=5)
            d.rectangle((x,y-36,x+130,y),fill=color)
            d.text((x+5,y-34),f'ID {identity}',font=font(24,True),fill='white')
        im=im.resize((1000,563))
        tile=Image.new('RGB',(1000,653),'white');tile.paste(im,(0,75))
        ImageDraw.Draw(tile).text((20,16),label,font=font(30,True),fill=INK)
        parts.append(tile)
    out=Image.new('RGB',(2040,710),'white');out.paste(parts[0],(0,0));out.paste(parts[1],(1040,0))
    ImageDraw.Draw(out).text((20,665),f'DanceTrack validation · {sequence} · frame {frame} · ID là dự đoán, không phải GT',font=font(24),fill=BLUE)
    out.save(FIG/'demo_comparison.png')
    original.resize((1280,720)).save(FIG/'dataset_sample.jpg')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    FIG.mkdir(exist_ok=True)
    rows=list(csv.DictReader((ROOT/'experiments/metrics_all_seeds.csv').open()))
    stats={}
    for budget in [5,10,15]:
        for model in ['e0','v3_gtlinked']:
            selected=[r for r in rows if int(r['budget_pct'])==budget and r['model']==model]
            stats[(budget,model)]={key:(float(np.mean([float(r[key]) for r in selected])),float(np.std([float(r[key]) for r in selected],ddof=1))) for key in ['HOTA','DetA','AssA','IDF1','IDSW']}
    existing={(int(r['budget_pct']),r['model'],r['metric']):float(r['mean']) for r in csv.DictReader((ROOT/'experiments/mean_std_summary.csv').open())}
    for (budget,model),values in stats.items():
        for key,(mean,std) in values.items():assert abs(existing[(budget,model,key)]-mean)<1e-8
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'font.size':11})
    for filename,metrics in [('quality_vs_budget',['HOTA','AssA','IDF1']),('idsw_detection',['IDSW','DetA'])]:
        fig,axes=plt.subplots(1,len(metrics),figsize=(5.1*len(metrics),3.8))
        for ax,key in zip(axes,metrics):
            for model,label,color,mark in [('e0','MOTIP nền',BLUE,'o'),('v3_gtlinked','MOTIP + G4',TEAL,'s')]:
                ax.errorbar([5,10,15],[stats[(b,model)][key][0] for b in [5,10,15]],yerr=[stats[(b,model)][key][1] for b in [5,10,15]],fmt=mark+'-',color=color,label=label,lw=2.3,capsize=4,ms=7)
            ax.set_title(key+' ↓' if key=='IDSW' else key+' ↑',weight='bold')
            ax.set_xticks([5,10,15],['5%','10%','15%']);ax.set_xlabel('Ngân sách dữ liệu train');ax.grid(alpha=.18)
        axes[0].legend(frameon=False,fontsize=9)
        fig.suptitle('DanceTrack full validation · trung bình ± độ lệch chuẩn mẫu, n = 3',fontsize=12)
        fig.tight_layout();fig.savefig(FIG/f'{filename}.png',dpi=200,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.2,3.8))
    delta_rows=[]
    for j,seed in enumerate([42,123,3407]):
        values=[]
        for budget in [5,10,15]:
            pair={r['model']:r for r in rows if int(r['budget_pct'])==budget and int(r['seed'])==seed}
            delta=float(pair['v3_gtlinked']['AssA'])-float(pair['e0']['AssA']);values.append(delta)
            delta_rows.append(dict(budget_pct=budget,seed=seed,delta_AssA=delta))
        ax.bar(np.arange(3)+(j-1)*.22,values,width=.22,label=f'Seed {seed}',color=[BLUE,TEAL,CORAL][j])
    ax.set_xticks(np.arange(3),['5% · 2.090 frame','10% · 4.179 frame','15% · 6.265 frame'])
    ax.set_ylabel('ΔAssA (đề xuất − nền), điểm');ax.set_title('Cải thiện liên kết xuất hiện ở cả 9 cặp seed/ngân sách',weight='bold');ax.legend(frameon=False,ncol=3);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'paired_delta.png',dpi=200);plt.close(fig)
    assert all(r['delta_AssA']>0 for r in delta_rows)
    tracking_image()
    if (ROOT/'demo/inference_crowded/v3/complete.json').exists():
        tracking_image(frame=461,sequence='dancetrack0034',inference='inference_crowded')
    primary=[];idsw=[];runtime=[]
    gains=[]
    for budget in [5,10,15]:
        for model,label in [('e0','MOTIP nền'),('v3_gtlinked','MOTIP + G4')]:
            cells=[f'{budget}%',label]
            for key in ['HOTA','DetA','AssA','IDF1']:
                mean,std=stats[(budget,model)][key];value=f'{mean:.2f} ± {std:.2f}'
                cells.append(f'**{value}**' if model!='e0' else value)
            primary.append(cells)
        e0=stats[(budget,'e0')];v3=stats[(budget,'v3_gtlinked')]
        gain=v3['AssA'][0]-e0['AssA'][0];gains.append(gain)
        idsw.append([f'{budget}%',f"{e0['IDSW'][0]:,.0f} ± {e0['IDSW'][1]:,.0f}",f"{v3['IDSW'][0]:,.0f} ± {v3['IDSW'][1]:,.0f}",f"{(1-v3['IDSW'][0]/e0['IDSW'][0])*100:.1f}%",f'{gain:+.3f}'])
        pair={r['model']:r for r in rows if int(r['budget_pct'])==budget and int(r['seed'])==42}
        a,b=float(pair['e0']['FPS']),float(pair['v3_gtlinked']['FPS'])
        runtime.append([f'{budget}%',f'{a:.2f}',f'{b:.2f}',f'{(b/a-1)*100:.1f}%',f"{float(pair['v3_gtlinked']['VRAM_MiB'])-float(pair['e0']['VRAM_MiB']):+.1f}"])
    feature_rows=[['Hộp hiện tại','4'],['Bốn hộp lịch sử','16'],['Validity từng lag','4'],['3 vận tốc + phần dư ngoại suy (2D)','8'],['Validity vận tốc/ngoại suy','4'],['2 vector biến thiên vận tốc (2D)','4'],['Validity biến thiên vận tốc','2'],['Độ dời tương đối + log-scale mỗi lag','16'],['Thay đổi log-diện tích mỗi lag','4'],['Mã lag: 0, 1/3, 2/3, 1','4'],['Tổng','66']]
    dataset_rows=[]
    for budget in [5,10,15]:
        manifest=(ROOT/f'experiments/manifests/train_{budget}pct.txt').read_text().split()
        frame={5:2090,10:4179,15:6265}[budget]
        dataset_rows.append([f'{budget}%',len(manifest),f'{frame:,}',3,'25.508 / 25 sequence'])
    protocol_rows=[['Detector','Deformable DETR R50, pretrained, đóng băng'],['Nhánh được học','Trajectory modeling, ID Decoder, ID embedding; thêm G4 MLP ở MOTIP + G4'],['Thời lượng / seed','10 epoch; 42, 123, 3407'],['Batch / tích lũy / precision','1 / 4 / FP32'],['Optimizer / LR / weight decay','AdamW / 0,0001 / 0,0005'],['Lịch LR','Milestone 6 và 9; hệ số 0,1'],['Clip / history khi suy luận','30 frame / miss tolerance 30'],['Geometry','66 → 128 → 128 → 1; 25.217 tham số'],['Tiền huấn luyện geometry','4 GT–GT + 4 GT-linked Pred–Pred; chỉ train'],['Fusion','T_b=2; T_g=0,5; λ=1 cố định'],['Chọn checkpoint','AssA trên small validation cố định 3.894 frame']]
    allseed=[[r['budget_pct']+'%',r['seed'],'MOTIP nền' if r['model']=='e0' else 'MOTIP + G4',r['epoch'],r['HOTA'],r['AssA'],r['IDF1'],f"{int(float(r['IDSW'])):,}"] for r in rows]
    manuscript=(ROOT/'report/manuscript.md').read_text()
    replacements={
        '%ABSTRACT_RESULTS%':f'Nhánh hình học cải thiện AssA trung bình lần lượt {gains[0]:.3f}, {gains[1]:.3f} và {gains[2]:.3f} điểm, với ID switch giảm trong cả chín cặp seed/ngân sách.',
        '%TABLE_FEATURES%':table(['Nhóm đặc trưng','Số chiều'],feature_rows),
        '%TABLE_DATASET%':table(['Ngân sách','Sequence train','Frame train','Số seed','Đánh giá'],dataset_rows),
        '%TABLE_PROTOCOL%':table(['Thành phần','Cấu hình thực nghiệm'],protocol_rows),
        '%TABLE_PRIMARY%':table(['Train','Mô hình','HOTA ↑','DetA ↑','AssA ↑','IDF1 ↑'],primary),
        '%TABLE_IDSW%':table(['Train','MOTIP nền IDSW ↓','MOTIP + G4 IDSW ↓','Giảm IDSW','ΔAssA'],idsw),
        '%TABLE_RUNTIME%':table(['Train','MOTIP nền FPS','MOTIP + G4 FPS','ΔFPS tương đối','ΔVRAM (MiB)'],runtime),
        '%TABLE_ALL_SEEDS%':table(['Train','Seed','Mô hình','Epoch chọn','HOTA','AssA','IDF1','IDSW'],allseed),
        '%RESULTS_DISCUSSION%':f'Ở ngân sách 5%, AssA tăng từ {stats[(5,"e0")]["AssA"][0]:.3f} lên {stats[(5,"v3_gtlinked")]["AssA"][0]:.3f}. Ở 10% và 15%, chênh lệch trung bình tương ứng là +{gains[1]:.3f} và +{gains[2]:.3f} điểm. Kết quả ở cả chín cặp seed/ngân sách đều có AssA của MOTIP + G4 cao hơn MOTIP nền. Các metric HOTA và IDF1 cũng tăng trong từng cặp của nhóm số liệu này. Mức tăng DetA nhỏ hơn rõ rệt so với AssA, phù hợp với mục tiêu cải thiện liên kết; tuy vậy, DetA không bất biến tuyệt đối do tác động của quyết định NEW/lifecycle. Không diễn giải các giá trị này thành kết quả test của paper gốc.'}
    for key,value in replacements.items():manuscript=manuscript.replace(key,value)
    assert '%TABLE_' not in manuscript and '%ABSTRACT_' not in manuscript
    (ROOT/'report/BAO_CAO_MOTIP.md').write_text(manuscript)
    summary=dict(stats=[dict(budget=b,model=m,**{k:dict(mean=x,std=s) for k,(x,s) in values.items()}) for (b,m),values in stats.items()],gains=gains,runtime=runtime,verified_pairs=9)
    (ROOT/'experiments/presentation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    from paper_figures import main as publication_figures
    publication_figures()
    print('Figures and manuscript generated from verified runs.',flush=True)


if __name__=='__main__':main()
