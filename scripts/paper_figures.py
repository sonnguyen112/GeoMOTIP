#!/usr/bin/env python3
"""Publication-style vector figures from the unchanged, verified course data.

All decorative token/feature/lag schematics are explicitly illustrative. Photos,
boxes and IDs come from the existing model rollouts; no new tracker inference.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,FancyArrowPatch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
INK='#18324A';BASE='#526C91';GEOM='#007E73';ORANGE='#B76336';GRAY='#65778A'
PALE='#EDF2F7';MINT='#E5F4F0';WHITE='white';BORDER='#C9D4DF'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,
 'axes.labelcolor':INK,'axes.edgecolor':BORDER,'xtick.color':GRAY,'ytick.color':GRAY,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix',
 'axes.spines.top':False,'axes.spines.right':False})


def text(ax,x,y,s,size=10,color=INK,weight='normal',ha='center',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,**kw)


def module(ax,xy,w,h,title,sub='',color=BASE,fill=PALE,size=11):
    x,y=xy
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.10',
                 facecolor=fill,edgecolor=color,linewidth=.85))
    title_size=min(size,(w-.18)*27.5/(max(map(len,title.split('\n')))*.61))
    text(ax,x+w/2,y+h*.74 if sub else y+h/2,title,title_size,color,'bold')
    if sub:
        body_size=8.6 if '$' in sub else min(8.6,(w-.18)*27.5/(max(map(len,sub.split('\n')))*.56))
        text(ax,x+w/2,y+h*.24,sub,body_size,INK)


def arrow(ax,pts,color=INK,dashed=False,lw=1.35):
    if len(pts)>2:
        ax.plot(*zip(*pts[:-1]),color=color,lw=lw,ls=(0,(4,3)) if dashed else '-',zorder=2)
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=11,
        color=color,lw=lw,linestyle=(0,(4,3)) if dashed else '-',zorder=3))


def figure(width=20,height=8.4):
    fig=plt.figure(figsize=(width/2.54,height/2.54))
    ax=fig.add_axes([.015,.015,.97,.97]);ax.set(xlim=(0,width),ylim=(0,height));ax.axis('off')
    return fig,ax


def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=350,bbox_inches='tight',pad_inches=.05,facecolor='white')
    plt.close(fig)


def photo(ax,path,extent):
    ax.imshow(Image.open(path),extent=extent,aspect='auto',zorder=0)


def record(model='e0',frame=461):
    return json.loads((ROOT/f'demo/inference_crowded/{model}/predictions.json').read_text())[frame-1]


def draw_tracks(ax,model,extent,frame=461,roi=None,labels=True,labelsize=5.9,label_filter=None,id_prefix=False):
    im=np.asarray(Image.open(ROOT/f'data/DanceTrack/val/dancetrack0034/img1/{frame:08d}.jpg'))
    ax.imshow(im,extent=extent,aspect='auto')
    x0,x1,y0,y1=extent
    for identity,(x,y,w,h) in zip(record(model,frame)['ids'],record(model,frame)['boxes']):
        c=plt.get_cmap('tab10')(identity%10)
        xx=x0+x/1920*(x1-x0);yy=y1-y/1080*(y1-y0)
        ww=w/1920*(x1-x0);hh=h/1080*(y1-y0)
        ax.add_patch(Rectangle((xx,yy-hh),ww,hh,fill=False,lw=.9,edgecolor=c))
        if labels and (label_filter is None or identity in label_filter):
            label=f'ID {identity}' if id_prefix else f'{identity}'
            text(ax,xx,yy,label,labelsize,INK,weight='bold',ha='left',va='bottom',clip_on=True,bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=.3))


def pipeline():
    fig,ax=figure(20.2,8.5)
    text(ax,.05,8.2,'(a) Quan sát hiện tại',10.5,weight='bold',ha='left')
    src=OUT/'dataset_sample.jpg'
    photo(ax,src,(.05,2.65,5.65,7.15));text(ax,1.35,5.25,r'Ảnh $I_t$',10)
    module(ax,(3.25,5.55),2.35,1.72,'Deformable\nDETR R50','Detector\nđóng băng',GRAY,'#F3F5F7',10)
    # Small pyramid conveys the detector's multiscale feature representation.
    for i in range(3):ax.add_patch(Rectangle((3.65+i*.16,4.86+i*.13),1.55-i*.32,.25,facecolor='#CED8E4',edgecolor=GRAY,lw=.6))
    arrow(ax,[(2.73,6.45),(3.2,6.45)])
    arrow(ax,[(5.65,6.45),(6.3,6.45)])
    text(ax,7.55,7.6,'Query hiện tại',10.5,weight='bold')
    for i in range(3):
        y=5.35+i*.52
        ax.add_patch(Rectangle((6.55,y),1.10,.38,facecolor='#C9D9EE',edgecolor=BASE,lw=.6))
        ax.add_patch(Rectangle((7.65,y),1.10,.38,facecolor='#E5E9EE',edgecolor=GRAY,lw=.6))
        text(ax,7.1,y+.19,r'$\mathbf{f}_n$',8.5)
        text(ax,8.2,y+.19,r'$\mathbf{e}_{\varnothing}$',8.5)
    text(ax,7.65,4.9,'Đặc trưng + ID rỗng\n256 + 256 chiều',8.4)
    arrow(ax,[(8.9,6.45),(9.5,6.45)],BASE)
    # The repeated decoder is the centerpiece, not a generic chain of boxes.
    for i in [2,1,0]:ax.add_patch(FancyBboxPatch((9.65+i*.09,4.65+i*.09),3.35,3.05,
         boxstyle='round,pad=.02,rounding_size=.10',facecolor='#F7FAFE' if i else '#EDF3FC',edgecolor=BASE,lw=.8))
    text(ax,11.32,7.35,'ID Decoder',11,BASE,'bold')
    for y,label in [(6.72,'Self-attention'),(6.02,'Cross-attention'),(5.32,'FFN + Add & Norm')]:
        ax.add_patch(Rectangle((9.97,y-.24),2.7,.48,facecolor=WHITE,edgecolor='#C4D1E1',lw=.6))
        text(ax,11.32,y,label,7.3)
    text(ax,11.32,8.10,'6 lớp; lớp đầu bỏ self-attention',7.4,GRAY)
    arrow(ax,[(13.25,6.45),(13.85,6.45)],BASE)
    text(ax,14.8,7.35,'Dự đoán ID',10.5,weight='bold')
    text(ax,14.8,6.35,r'$\boldsymbol{\ell}_B\in\mathbb{R}^{51}$',12)
    text(ax,14.8,5.92,'50 slot + NEW',8)
    arrow(ax,[(14.8,5.68),(14.8,5.48)])
    module(ax,(13.65,4.85),2.55,.62,'object-max','',BASE,PALE,9)
    text(ax,14.9,4.42,'Gán slot / xử lý NEW',8.5)
    arrow(ax,[(16.27,5.16),(16.65,5.16),(16.65,6.45),(16.99,6.45)])
    draw_tracks(ax,'e0',(17.05,20.15,5.55,7.30),labels=True,labelsize=7,label_filter={0},id_prefix=True)
    text(ax,18.6,5.0,'Hộp dự đoán\n+ ID toàn cục',8.4,weight='bold')
    text(ax,.05,3.85,'(b) Ngữ cảnh đã có ID',10.5,weight='bold',ha='left')
    # Abstract memory bank, not a measured attention or confidence matrix.
    for r,cid in enumerate(['ID a','ID b','ID c']):
        text(ax,.08,2.96-r*.52,cid,8.2,ha='left')
        for col in range(4):
            ax.add_patch(Rectangle((1.0+col*.88,2.78-r*.52),.72,.34,
                facecolor=['#C5D6EC','#D5EAE0','#EEDBCB'][r],edgecolor='white',lw=1))
    for i,lab in enumerate([r'$t-T$',r'$\cdots$',r'$t-2$',r'$t-1$']):text(ax,1.36+i*.88,3.44,lab,8)
    text(ax,2.4,1.40,'Đặc trưng / hộp / slot / timestamp',7.0)
    arrow(ax,[(4.6,2.43),(5.25,2.43)],GEOM)
    module(ax,(5.35,1.58),3.0,1.72,'Trajectory\nmodeling','FFN + residual\nLayerNorm',GEOM,MINT,9.5)
    arrow(ax,[(8.42,2.43),(9.1,2.43)],GEOM)
    module(ax,(9.2,1.58),3.55,1.72,'Ghép ID embedding',r'$[\widetilde{\mathbf{f}};\mathbf{e}_{\mathrm{ID}}]\in\mathbb{R}^{512}$',GEOM,MINT,9.5)
    arrow(ax,[(11.0,3.37),(11.0,4.59)],GEOM)
    text(ax,12.75,3.90,'Key / Value\nMask + thời gian',6.9,GEOM)
    arrow(ax,[(18.55,5.03),(18.55,.55),(4.8,.55),(4.8,1.96),(4.55,1.96)],GEOM,True,1.05)
    text(ax,10.6,.20,'Cập nhật bộ nhớ sau quyết định ID; chỉ dùng lịch sử trước t',8.4,GEOM)
    save(fig,'pipeline')


def multilag():
    fig,ax=figure(18.5,7.4)
    text(ax,.15,7.04,'(a) Truy xuất theo timestamp, không theo vị trí trong mảng',10.5,weight='bold',ha='left')
    xs=[1.35,5.1,8.85,12.6,16.5];ys=[4.8]*5
    ax.plot([.7,17.5],[4.8,4.8],color=BORDER,lw=1.2)
    for i,(x,lag) in enumerate(zip(xs,[8,4,2,1,0])):
        invalid=i==1;color=ORANGE if invalid else BASE if i<4 else GEOM
        ax.scatter([x],[4.8],s=85,facecolors='white' if invalid else color,edgecolors=color,lw=1.6,zorder=4)
        text(ax,x,4.26,f'$t-{lag}$' if lag else '$t$',12,color,'bold')
        if invalid:
            ax.plot([x-.20,x+.20],[6.0,6.4],color=ORANGE,lw=1.6)
            ax.plot([x-.20,x+.20],[6.4,6.0],color=ORANGE,lw=1.6)
            text(ax,x,5.46,'Thiếu quan sát',8.5,ORANGE)
        else:
            # Local bbox glyphs illustrate the storage rule; no empirical motion claim.
            for j in range(2):
                ax.add_patch(Rectangle((x-.58+j*.19,5.65+j*.17),.78,.80,fill=False,edgecolor=color,lw=1.0,alpha=.5+.5*j))
        text(ax,x,3.72,'mask = 0' if invalid else ('Phát hiện hiện tại' if i==4 else 'mask = 1'),8.8,color)
    text(ax,.15,2.77,'(b) Một mô tả cho mỗi cặp phát hiện–ứng viên',10.5,weight='bold',ha='left')
    module(ax,(.5,.8),4.4,1.1,'Hộp hiện tại',r'$\mathbf{b}_t^n=(c_x,c_y,w,h)$',GEOM,MINT,10)
    module(ax,(6.1,.8),5.3,1.1,'Bốn lag lịch sử + validity',r'$\{(\mathbf{b}_{t-\Delta}^{k},m_\Delta^k)\}_{\Delta\in\{1,2,4,8\}}$',BASE,PALE,10)
    arrow(ax,[(4.95,1.35),(5.65,1.35)])
    text(ax,5.5,2.08,'đối chiếu',8.2,GRAY)
    arrow(ax,[(11.5,1.35),(12.3,1.35)])
    module(ax,(12.5,.8),5.25,1.1,'G4: hình học có thứ tự',r'$\mathbf{g}_{nk}\in\mathbb{R}^{66}$',GEOM,MINT,10)
    text(ax,9.25,.16,'Lag thiếu được điền 0 và đánh dấu; không sao chép quan sát lân cận.',8.7,GRAY)
    save(fig,'multilag')


def geometry_pretraining():
    """Schematic of the audited two-stage training and predicted-ID inference."""
    fig,ax=figure(20,10.8)
    def observations(y,predicted=False):
        for j,lab in enumerate([r'$t-2$',r'$t-1$',r'$t$']):
            x=.35+j*2.0
            ax.add_patch(Rectangle((x,y),1.62,1.30,facecolor='#F5F7FA',edgecolor=BORDER,lw=.65))
            text(ax,x+.81,y+1.57,lab,9)
            for k,(col,identity) in enumerate([(BASE,'A'),(ORANGE,'B')]):
                bx=x+.18+k*.70+j*.04;by=y+.24+k*.10
                if predicted:
                    ax.add_patch(Rectangle((bx,by),.43,.69,fill=False,edgecolor=GRAY,lw=.55,ls='--'))
                    bx+=[.07,-.05,.04][j];by+=[-.05,.03,-.02][j]
                ax.add_patch(Rectangle((bx,by),.43,.69,fill=False,edgecolor=col,lw=1.25))
                text(ax,bx+.215,by+.87,identity,8,col,'bold')
        text(ax,3.05,y-.29,'Hộp GT + ID GT' if not predicted else 'Hộp detector; ID nối lịch sử lấy từ GT train',8.4,GRAY)
    text(ax,.1,10.46,'(a) GT–GT: học từ hộp có nhãn chính xác · 4 epoch',10.4,BASE,'bold',ha='left')
    observations(8.20)
    text(ax,.1,7.16,'(b) GT-linked Pred–Pred: học tiếp với hộp detector · 4 epoch',10.4,GEOM,'bold',ha='left')
    observations(4.90,True)
    for y,col in [(8.20,BASE),(4.90,GEOM)]:
        arrow(ax,[(6.15,y+.65),(6.85,y+.65)],col)
        module(ax,(7.0,y+.04),3.8,1.22,'G4 theo cặp','Hộp hiện tại + lịch sử\nứng viên A hoặc B',col,PALE if col==BASE else MINT,10)
        arrow(ax,[(10.9,y+.65),(11.65,y+.65)],col)
        module(ax,(11.8,y+.04),3.0,1.22,'Geometry MLP','66 → 128 → 128 → 1',col,PALE if col==BASE else MINT,10)
        arrow(ax,[(14.9,y+.65),(15.55,y+.65)],col)
        text(ax,17.6,y+.97,'Nhãn giám sát BCE',9.4,col,'bold')
        text(ax,17.6,y+.47,'A–A: cùng ID → 1\nA–B: khác ID → 0',9)
    arrow(ax,[(13.3,8.15),(13.3,6.25)],GEOM,True)
    text(ax,15.3,7.12,'Giữ trọng số MLP\nvà tiếp tục học',8.5,GEOM)
    text(ax,10.0,4.23,'Nhãn GT chỉ nối lịch sử và tạo mục tiêu; mã ID không nằm trong vector G4.',9,GEOM)
    ax.plot([.1,19.8],[3.85,3.85],color=BORDER,lw=.8)
    text(ax,.1,3.44,'(c) Khi chạy tracker: lịch sử được nối bằng ID dự đoán, không dùng nhãn GT',10.4,weight='bold',ha='left')
    module(ax,(.25,1.68),3.0,1.10,'Detector','Hộp dự đoán',BASE,PALE,10)
    module(ax,(4.15,1.68),4.0,1.10,'MOTIP + G4','Chấm và liên kết',GEOM,MINT,10)
    module(ax,(9.05,1.68),3.6,1.10,'ID dự đoán','Cho quan sát hiện tại',BASE,PALE,10)
    module(ax,(13.55,1.68),6.0,1.10,'Cập nhật lịch sử','Hộp được lưu theo ID vừa dự đoán',BASE,PALE,10)
    for x1,x2 in [(3.35,4.0),(8.25,8.90),(12.75,13.40)]:arrow(ax,[(x1,2.23),(x2,2.23)])
    arrow(ax,[(16.55,1.60),(16.55,.65),(6.15,.65),(6.15,1.60)],GEOM,True)
    text(ax,10.0,.27,'Dùng ở frame tiếp theo; lịch sử chỉ chứa các quan sát trước thời điểm hiện tại.',8.6,GRAY)
    save(fig,'geometry_pretraining')


def geometry():
    fig,ax=figure(19.6,8.7)
    text(ax,.05,8.35,'(a) Tín hiệu hình học theo cặp ứng viên',10.5,weight='bold',ha='left')
    # Ordered trajectory, marked as a conceptual illustration rather than model output.
    positions=np.array([[.65,6.3],[1.35,6.15],[2.2,6.55],[3.10,7.18]])
    ax.plot(positions[:,0],positions[:,1],'-o',color=BASE,lw=1.5,ms=4)
    for (x,y),lag in zip(positions,[8,4,2,1]):text(ax,x,y-.42,f'$t-{lag}$',8.2)
    ax.scatter([3.70],[7.42],s=65,color=GEOM,marker='s');text(ax,3.72,6.94,'$t$',9)
    text(ax,2.1,5.30,'Hộp hiện tại + lịch sử ứng viên',8.3)
    arrow(ax,[(4.1,6.65),(4.75,6.65)],GEOM)
    module(ax,(4.9,5.75),4.1,1.8,'G4 · 66 chiều','Độ dời / log-scale / vận tốc\nLag và validity tường minh',GEOM,MINT,10.5)
    arrow(ax,[(9.1,6.65),(9.75,6.65)],GEOM)
    module(ax,(9.9,5.75),4.3,1.8,'Geometry MLP',r'$66\to128\to128\to1$'+'\nGELU; 25.217 tham số',GEOM,MINT,10.5)
    arrow(ax,[(14.3,6.65),(14.95,6.65)],GEOM)
    text(ax,16.95,6.95,r'$\ell_{G,nk}$',20,GEOM)
    text(ax,16.95,6.07,'Điểm tương thích',9.1,GEOM)
    text(ax,.05,4.50,'(b) Giữ nguyên nhánh MOTIP và bổ sung logit geometry',10.5,weight='bold',ha='left')
    module(ax,(.55,2.05),4.7,1.48,'MOTIP nền',r'$\boldsymbol{\ell}_B/T_b$',BASE,PALE,11)
    arrow(ax,[(5.4,2.78),(8.30,2.78)],BASE)
    ax.add_patch(plt.Circle((8.72,2.78),.34,facecolor='white',edgecolor=INK,lw=1.2))
    text(ax,8.72,2.79,'+',18)
    arrow(ax,[(16.95,5.65),(16.95,3.93),(8.72,3.93),(8.72,3.2)],GEOM)
    text(ax,12.2,4.10,r'$\lambda\,\boldsymbol{\ell}_G/T_g$',10.5,GEOM)
    arrow(ax,[(9.15,2.78),(10.05,2.78)],INK)
    text(ax,14.20,2.95,r'$\boldsymbol{\ell}_{\mathrm{final}}=\frac{\boldsymbol{\ell}_B}{2}+\frac{\boldsymbol{\ell}_G}{0.5}$',19)
    text(ax,14.2,1.84,r'$T_b=2,\quad T_g=0.5,\quad\lambda=1,\quad\ell_{G,\mathrm{NEW}}=0$',10.5)
    text(ax,9.8,.65,'Huấn luyện: '+r'$\mathcal{L}=\mathcal{L}_{\mathrm{fused}}+\mathcal{L}_{\mathrm{geom}}$',13)
    text(ax,9.8,.12,'Geometry chỉ chấm track hiện hữu; mô tả đường đi là minh họa cấu trúc.',8.4,GRAY)
    save(fig,'geometry_fusion')


def plots():
    rows=list(csv.DictReader((ROOT/'experiments/metrics_all_seeds.csv').open()))
    def values(b,m,k):return np.array([float(r[k]) for r in rows if int(r['budget_pct'])==b and r['model']==m])
    xs=np.array([5,10,15])
    for name,metrics in [('quality_vs_budget',['HOTA','AssA','IDF1']),('idsw_detection',['IDSW','DetA'])]:
        fig,axs=plt.subplots(1,len(metrics),figsize=(3.15*len(metrics),3.25),layout='constrained')
        for j,(ax,k) in enumerate(zip(axs,metrics)):
            for m,label,c,marker in [('e0','MOTIP nền',BASE,'o'),('v3_gtlinked','MOTIP + G4',GEOM,'s')]:
                y=[values(b,m,k).mean() for b in xs];err=[values(b,m,k).std(ddof=1) for b in xs]
                ax.errorbar(xs,y,yerr=err,fmt=marker+'-',color=c,lw=1.8,ms=6,capsize=3,elinewidth=1.0,label=label)
            ax.set_title(f'({chr(97+j)}) {k}',loc='left',fontsize=13,fontweight='bold',pad=13)
            ax.set_xticks(xs,['5%\n2.090 frame','10%\n4.179 frame','15%\n6.265 frame'],fontsize=10)
            ax.set_xlabel('Ngân sách huấn luyện',fontsize=11,labelpad=10)
            ax.set_ylabel('Số sự kiện (thấp hơn tốt hơn)' if k=='IDSW' else 'Điểm (cao hơn tốt hơn)',fontsize=10)
            ax.grid(axis='y',color='#E7ECF1',lw=.7);ax.set_axisbelow(True);ax.margins(x=.12)
        axs[0].legend(frameon=False,fontsize=10,loc='best')
        save(fig,name)
    fig,ax=plt.subplots(figsize=(7.5,3.4),layout='constrained')
    colors=[BASE,GEOM,ORANGE]
    for j,s in enumerate([42,123,3407]):
        ds=[]
        for b in xs:
            p={r['model']:float(r['AssA']) for r in rows if int(r['budget_pct'])==b and int(r['seed'])==s}
            ds.append(p['v3_gtlinked']-p['e0'])
        centers=np.arange(3)+(j-1)*.23
        ax.bar(centers,ds,.205,color=colors[j],label=f'Seed {s}',zorder=3)
        for x,y in zip(centers,ds):ax.text(x,y+.13,f'+{y:.2f}',ha='center',va='bottom',fontsize=9,color=colors[j])
    ax.axhline(0,color=GRAY,lw=.8);ax.set_ylim(0,11.5)
    ax.set_xticks(range(3),['5% · 2.090 frame','10% · 4.179 frame','15% · 6.265 frame'])
    ax.set_ylabel(r'$\Delta$AssA (đề xuất − nền), điểm');ax.set_xlabel('Ngân sách huấn luyện',labelpad=10)
    ax.legend(frameon=False,ncol=3,loc='upper center');ax.grid(axis='y',color='#E7ECF1',zorder=0)
    save(fig,'paired_delta')


def qualitative():
    fig,axs=plt.subplots(2,2,figsize=(12,6.15),gridspec_kw={'height_ratios':[1.28,1]},layout='constrained')
    for col,(m,title,c) in enumerate([('e0','MOTIP nền',BASE),('v3','MOTIP + G4',GEOM)]):
        ax=axs[0,col];draw_tracks(ax,m,(0,1920,0,1080),labels=False);ax.set(xlim=(0,1920),ylim=(0,1080));ax.axis('off')
        ax.set_title(title,loc='left',fontsize=13,fontweight='bold',color=c,pad=10)
        # Fixed central ROI, same coordinates for both models, not a selected failure case.
        ax.add_patch(Rectangle((400,400),1100,530,fill=False,edgecolor='white',lw=1,ls='--'))
        ax.set_aspect('equal',adjustable='box')
        ax=axs[1,col];draw_tracks(ax,m,(0,1920,0,1080),labelsize=11.0);ax.set(xlim=(400,1500),ylim=(400,930));ax.axis('off')
        ax.set_aspect('equal',adjustable='box')
        ax.text(.01,-.02,'Cùng vùng trung tâm, phóng to',transform=ax.transAxes,ha='left',va='top',fontsize=10,color=c)
    save(fig,'demo_comparison')


def observation_assets():
    # Readable slide illustrations from exactly the existing saved rollout.
    fig,ax=plt.subplots(figsize=(10,5.625),layout='constrained')
    im=np.asarray(Image.open(ROOT/'data/DanceTrack/val/dancetrack0034/img1/00000461.jpg'))
    ax.imshow(im)
    for (x,y,w,h),score in zip(record()['boxes'],record()['scores']):
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#51D6B4',lw=1.4))
        if score>=.6:ax.text(x,y-7,f'{score:.2f}',fontsize=9,color='white',bbox=dict(facecolor=GEOM,edgecolor='none',pad=1),clip_on=True)
    ax.axis('off');save(fig,'detector_observation')
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.6),layout='constrained')
    for ax,frame,lab in zip(axes,[453,457,461],['t − 8','t − 4','t']):
        im=np.asarray(Image.open(ROOT/f'data/DanceTrack/val/dancetrack0034/img1/{frame:08d}.jpg'))
        ax.imshow(im)
        for ident,(x,y,w,h) in zip(record(frame=frame)['ids'],record(frame=frame)['boxes']):
            c=plt.get_cmap('tab10')(ident%10);ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=c,lw=1.2))
            if ident == 0:
                ax.text(x,y-10,f'ID {ident}',fontsize=9,color='white',ha='left',va='bottom',
                        bbox=dict(facecolor=c,edgecolor='white',linewidth=.35,pad=1.1),clip_on=True,zorder=5)
        ax.set_title(lab,fontsize=15,color=INK);ax.axis('off')
    save(fig,'association_filmstrip')


def main():
    OUT.mkdir(exist_ok=True)
    pipeline();multilag();geometry();geometry_pretraining();plots();qualitative();observation_assets()
    if (ROOT/'experiments/qualitative_association.json').exists():
        from qualitative_association import render as render_association_cases
        render_association_cases()
    (OUT/'FIGURE_PROVENANCE.json').write_text(json.dumps(dict(
        architecture_source='src/motip/models/motip/{trajectory_modeling,id_decoder}.py',
        geometry_source='src/motip/models/motip/geometry/{geometry_features,geometry_branch}.py',
        metrics_source='experiments/metrics_all_seeds.csv',new_training=False,
        association_examples='experiments/qualitative_association.json; purposively selected positive events, GT used only for post-hoc analysis',
        schematic_not_measurement=['token/memory drawings','trajectory drawing in geometry figure','geometry pretraining box/ID examples'],
        qualitative_sequence='dancetrack0034',qualitative_frame=461,
        qualitative_roi=dict(xmin=400,xmax=1500,ymin=150,ymax=680),
        inference_inputs='existing MOTIP nền/MOTIP + G4 predictions; no GT used to draw predicted boxes',
        formats=['PDF vector + embedded photographs','SVG','PNG 350 DPI']),indent=2,ensure_ascii=False)+'\n')
    print('Seven publication-style figures exported as PDF/SVG/PNG; metrics unchanged.')


if __name__=='__main__':main()
