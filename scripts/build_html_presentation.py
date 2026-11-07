#!/usr/bin/env python3
"""HTML-first scientific presentation, rendered in Chromium then exported to PPTX.

Text, simple shapes, tables, and result charts remain native PowerPoint objects.
Only photographs, typeset equations and complex diagrams are image assets.
"""
from pathlib import Path
import csv
import html
import json
import math

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'presentation/html'
INK='#18324A';TEAL='#007E73';BLUE='#526C91';CORAL='#B76336';MUTED='#64768A';LIGHT='#EAF1F6'
PARTS=[]


def style(x,y,w,h):return f'left:{x}px;top:{y}px;width:{w}px;height:{h}px;'


def t(x,y,w,h,s,size=34,color=INK,bold=False,align='left',middle=False):
    display_size=round(size*1.12) if 23 <= size <= 46 else size
    css=style(x,y,w,h)+f'font-size:{display_size}px;color:{color};font-weight:{700 if bold else 400};text-align:{align};'
    if middle:css+='display:flex;align-items:center;justify-content:'+('center' if align=='center' else 'flex-start')+';'
    return '<div data-kind="text" class="text" style="'+css+'">'+html.escape(s)+'</div>'


def shape(x,y,w,h,color=LIGHT,radius=0,kind='rect'):
    return f'<div data-kind="{kind}" style="{style(x,y,w,h)}background:{color};border-radius:{radius}px"></div>'


def img(file,x,y,w,h):
    return f'<img data-kind="image" src="../../{file}" style="{style(x,y,w,h)}object-fit:contain"/>'


def formula(name,x,y,w,h):return img(f'figures/math/{name}.png',x,y,w,h)


def graphic(svg,x,y,w,h):
    return f'<div data-kind="graphic" style="{style(x,y,w,h)}">{svg}</div>'


def table(headers,rows,x=80,y=255,w=1440,h=470,size=29):
    data=html.escape(json.dumps(dict(headers=headers,rows=rows),ensure_ascii=False),quote=True)
    inner='<table><thead><tr>'+''.join(f'<th>{html.escape(str(c))}</th>' for c in headers)+'</tr></thead><tbody>'
    inner+=''.join('<tr>'+''.join(f'<td>{html.escape(str(c))}</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table>'
    return f'<div data-kind="table" data-table="{data}" class="table" style="{style(x,y,w,h)}font-size:{size*1.08}px">{inner}</div>'


def svg_base(body,w=1400,h=500):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" font-family="Arial" fill="{INK}">{body}</svg>'


def timeline(labels,subtitles,w=1400,h=420,colors=None):
    colors=colors or [BLUE,TEAL,CORAL,BLUE]
    body=f'<path d="M130 170H{w-130}" stroke="#CBD6DF" stroke-width="3"/>'
    for i,(name,sub) in enumerate(zip(labels,subtitles)):
        xx=130+i*(w-260)/max(1,len(labels)-1);c=colors[i%len(colors)]
        body+=f'<circle cx="{xx}" cy="170" r="31" fill="{c}"/>'
        body+=f'<text x="{xx}" y="181" font-size="31" fill="white" text-anchor="middle">{i+1}</text>'
        for j,line in enumerate(name.split('\n')):body+=f'<text x="{xx}" y="{245+j*42}" font-size="35" font-weight="700" text-anchor="middle">{html.escape(line)}</text>'
        for j,line in enumerate(sub.split('\n')):body+=f'<text x="{xx}" y="{335+j*38}" font-size="29" fill="{MUTED}" text-anchor="middle">{html.escape(line)}</text>'
    return svg_base(body,w,h)


def token_bank():
    b=''
    for j,lab in enumerate(['t − T','…','t − 2','t − 1']):b+=f'<text x="{260+j*190}" y="60" font-size="30" text-anchor="middle">{lab}</text>'
    for i,col in enumerate([BLUE,TEAL,CORAL]):
        b+=f'<text x="55" y="{150+i*100}" font-size="31" fill="{col}">Track {i+1}</text>'
        for j in range(4):
            if (i,j)==(1,1):
                b+=f'<rect x="{185+j*190}" y="{105+i*100}" width="150" height="66" fill="none" stroke="{CORAL}" stroke-dasharray="7 5"/><text x="{260+j*190}" y="{149+i*100}" font-size="26" fill="{CORAL}" text-anchor="middle">mask</text>'
            else:
                for k in range(5):b+=f'<rect x="{185+j*190+k*29}" y="{105+i*100}" width="25" height="66" fill="{col}" opacity="{.35+.12*k}"/>'
    b+='<text x="500" y="465" font-size="26" fill="#64768A" text-anchor="middle">Sơ đồ cấu trúc tensor; không phải attention map đo được.</text>'
    return svg_base(b,1000,500)


def embeddings():
    b=''
    for i,(name,c) in enumerate([('Đặc trưng đối tượng',BLUE),('ID embedding',TEAL)]):
        x=50+i*490
        b+=f'<text x="{x+180}" y="65" font-size="32" text-anchor="middle" fill="{c}">{name}</text>'
        for j in range(9):b+=f'<rect x="{x+j*40}" y="115" width="34" height="120" fill="{c}" opacity="{.35+.07*j}"/>'
        b+=f'<text x="{x+180}" y="285" font-size="30" text-anchor="middle">256 chiều</text>'
    b+='<text x="475" y="193" font-size="66" text-anchor="middle">+</text><path d="M1000 180H1120" stroke="#18324A" stroke-width="4"/><path d="M1104 170L1120 180L1104 190" fill="none" stroke="#18324A" stroke-width="4"/>'
    for j in range(9):b+=f'<rect x="{1155+j*22}" y="115" width="18" height="120" fill="{BLUE if j<5 else TEAL}"/>'
    b+='<text x="1250" y="285" font-size="30" text-anchor="middle">512 chiều</text>'
    return svg_base(b,1400,345)


def pretraining_process():
    b=''
    for row,(name,sub,col,jitter) in enumerate([
        ('GT–GT · 4 epoch','Tọa độ GT + danh tính GT',BLUE,False),
        ('GT-linked Pred–Pred · 4 epoch','Tọa độ detector; GT train nối danh tính',TEAL,True)]):
        y=65+row*240
        for j,lag in enumerate(['t − 2','t − 1','t']):
            x=25+j*150
            b+=f'<text x="{x+65}" y="{y-20}" font-size="27" text-anchor="middle">{lag}</text>'
            b+=f'<rect x="{x}" y="{y}" width="130" height="120" fill="#F2F5F8"/>'
            for k,(c,ident) in enumerate([(BLUE,'A'),(CORAL,'B')]):
                xx=x+15+k*57;yy=y+35
                if jitter:b+=f'<rect x="{xx}" y="{yy}" width="37" height="66" fill="none" stroke="#8291A0" stroke-dasharray="4 3"/>'
                dx=(6 if j!=1 else -4) if jitter else 0
                b+=f'<rect x="{xx+dx}" y="{yy}" width="37" height="66" fill="none" stroke="{c}" stroke-width="3"/><text x="{xx+18}" y="{y+24}" font-size="25" fill="{c}" text-anchor="middle">{ident}</text>'
        b+=f'<text x="540" y="{y+20}" font-size="34" fill="{col}" font-weight="700">{name}</text><text x="540" y="{y+76}" font-size="28">{sub}</text>'
        b+=f'<text x="1100" y="{y+20}" font-size="32" fill="{col}" font-weight="700">G4 → MLP</text><text x="1100" y="{y+76}" font-size="27">A–A: nhãn 1</text><text x="1100" y="{y+116}" font-size="27">A–B: nhãn 0</text>'
    b+='<path d="M1220 200V274" stroke="#007E73" stroke-width="3" stroke-dasharray="7 5"/><path d="M1211 262L1220 274L1229 262" fill="none" stroke="#007E73" stroke-width="3"/><text x="900" y="244" font-size="26" fill="#007E73" text-anchor="middle">Giữ trọng số, học tiếp cùng MLP</text>'
    return svg_base(b,1400,485)


def chart(spec,x,y,w,h):
    values=np_array=spec['values'];colors=spec['colors'];names=spec['names'];cats=spec['categories']
    ww=round(w);hh=round(h);L=74;R=18;T=64;B=90
    flat=[v for vs in values for v in vs]
    if spec.get('kind')=='bar':lo=0;hi=math.ceil((max(flat)+1)/2)*2
    else:
        errs=spec.get('std',[[0]*len(cats)]*len(values))
        lo=spec.get('axis_min',math.floor(min(v-e for vs,es in zip(values,errs) for v,e in zip(vs,es))-2))
        hi=math.ceil(max(v+e for vs,es in zip(values,errs) for v,e in zip(vs,es))+2)
    def yy(v):return T+(hi-v)/(hi-lo)*(hh-T-B)
    def xx(i):return L+(i+.5)/len(cats)*(ww-L-R)
    b=f'<text x="{L}" y="34" font-size="35" font-weight="700">{html.escape(spec["title"])}</text>'
    for v in np_linspace(lo,hi,5):
        yv=yy(v)
        b+=f'<line x1="{L}" y1="{yv}" x2="{ww-R}" y2="{yv}" stroke="#E2E8EE"/><text x="{L-12}" y="{yv+9}" font-size="26" text-anchor="end" fill="{MUTED}">{v:.0f}</text>'
    for j,(vs,c) in enumerate(zip(values,colors)):
        if spec.get('kind')=='bar':
            bw=(ww-L-R)/len(cats)/4.6
            for i,v in enumerate(vs):
                cx=xx(i)+(j-(len(values)-1)/2)*bw*1.1
                b+=f'<rect x="{cx-bw/2}" y="{yy(v)}" width="{bw}" height="{yy(0)-yy(v)}" fill="{c}"/>'
        else:
            pts=' '.join(f'{xx(i)},{yy(v)}' for i,v in enumerate(vs))
            b+=f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="4"/>'
            for i,v in enumerate(vs):
                e=spec.get('std',[[0]*len(cats)]*len(values))[j][i]
                b+=f'<path d="M{xx(i)} {yy(v-e)}V{yy(v+e)} M{xx(i)-6} {yy(v-e)}H{xx(i)+6} M{xx(i)-6} {yy(v+e)}H{xx(i)+6}" fill="none" stroke="{c}" stroke-width="2"/>'
                if j==0:b+=f'<circle cx="{xx(i)}" cy="{yy(v)}" r="7" fill="{c}"/>'
                else:b+=f'<rect x="{xx(i)-6}" y="{yy(v)-6}" width="12" height="12" fill="{c}"/>'
    for i,cat in enumerate(cats):b+=f'<text x="{xx(i)}" y="{hh-47}" font-size="29" text-anchor="middle">{html.escape(cat)}</text>'
    for j,(name,c) in enumerate(zip(names,colors)):
        x0=L+j*(ww-L-R)/len(names)
        b+=f'<circle cx="{x0}" cy="{hh-11}" r="6" fill="{c}"/><text x="{x0+14}" y="{hh-4}" font-size="24">{html.escape(name)}</text>'
    attr=html.escape(json.dumps(spec,ensure_ascii=False),quote=True)
    return f'<div data-kind="chart" data-chart="{attr}" style="{style(x,y,w,h)}">'+svg_base(b,ww,hh)+'</div>'


def np_linspace(lo,hi,n):return [lo+(hi-lo)*i/(n-1) for i in range(n)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    notes=json.loads((ROOT/'presentation/slide_structure.json').read_text())
    rows=list(csv.DictReader((ROOT/'experiments/metrics_all_seeds.csv').open()))
    summary=json.loads((ROOT/'experiments/presentation_summary.json').read_text())
    stats={(r['budget'],r['model']):r for r in summary['stats']}
    def result_spec(metric):return dict(title=metric,categories=['5%','10%','15%'],names=['MOTIP nền','MOTIP + G4'],colors=[BLUE,TEAL],values=[[stats[(b,m)][metric]['mean'] for b in [5,10,15]] for m in ['e0','v3_gtlinked']],std=[[stats[(b,m)][metric]['std'] for b in [5,10,15]] for m in ['e0','v3_gtlinked']],**({'axis_min':0} if metric=='IDSW' else {}))
    scenes=[]
    def slide(title,sub,body,dark=False):
        n=len(scenes)+1
        notes[n-1]['title']=title.replace('\n',' ')
        longest=max(len(line) for line in title.split('\n'))
        title_size=round(60*min(1,47/longest))
        head=t(80,45,1440,150,title,title_size,'white' if dark else INK,True)
        if sub:head+=t(82,205 if '\n' in title else 155,1436,54,sub,30,'#CCDDEB' if dark else MUTED)
        foot=t(82,850,1340,28,'Nguyễn Hồ Trường Sơn  |  25C01020  |  Thị giác máy tính',20,'#B8CCDC' if dark else MUTED)+t(1450,846,65,35,f'{n:02}',22,'#B8CCDC' if dark else MUTED,align='right')
        scenes.append(f'<section class="slide {"dark" if dark else ""}" id="slide-{n}" data-notes="{html.escape(notes[n-1]["notes"],quote=True)}">{head}{body}{foot}</section>')
    slide('MOTIP: từ phát hiện\nđến duy trì danh tính','Tái hiện paper CVPR 2025 và bổ sung hình học quỹ đạo đa độ trễ',
      img('figures/dataset_sample.jpg',640,315,920,518)+t(82,335,540,200,'Vị trí đúng\nchưa đủ.',66,'white',True)+t(82,594,510,114,'Cần giữ đúng người\nqua thời gian.',38,'#CFEDE8')+t(82,754,540,65,'Giảng viên: TS. Trương Phước Hưng',24,'#CCDDEB'),True)
    slide('Phát hiện đúng người chưa đảm bảo đúng ID','Ngoại hình tương tự, giao cắt và che khuất làm liên kết trở nên khó.',
      img('figures/association_filmstrip.png',55,255,1490,415)+t(120,695,640,75,'Phát hiện: người ở đâu?',36,BLUE,True)+t(870,695,640,75,'Liên kết: ai là ai?',36,TEAL,True))
    slide('DanceTrack tách độ khó liên kết khỏi ngoại hình','Dữ liệu video người múa/nhảy; trang phục giống nhau và chuyển động đa dạng.',
      img('figures/dataset_sample.jpg',80,280,1000,530)+t(1160,300,360,84,'41.796',66,BLUE,True)+t(1160,385,360,80,'frame train\n40 sequence',30)+t(1160,550,360,84,'25.508',66,TEAL,True)+t(1160,635,360,100,'frame validation\n25 sequence',30))
    slide('MOTIP dự đoán ID từ ngữ cảnh đã có nhãn','Lịch sử được đọc trước; chỉ ghi quan sát mới sau quyết định ID.',img('figures/pipeline.png',60,258,1480,565))
    slide('Detector tạo hộp, confidence và đặc trưng','Cùng một detector tiền huấn luyện đóng băng trong các cấu hình được đối chiếu.',
      img('figures/detector_observation.png',80,280,865,490)+t(1010,320,510,128,'Deformable DETR\nResNet-50',43,INK,True)+t(1010,505,510,130,'Tọa độ hộp chuẩn hóa\nĐặc trưng query 256 chiều',31)+formula('detection',955,690,585,95))
    slide('Lịch sử được tổ chức theo track và thời gian','Padding và quan sát thiếu phải bị loại khỏi attention và loss.',
      graphic(token_bank(),80,278,950,490)+formula('history',1060,340,460,230)+t(1060,650,460,120,'B: batch · G: nhóm\nT: thời gian · K: track',29,MUTED))
    slide('ID embedding cho biết quan sát thuộc slot nào','ID slot là ký hiệu cục bộ; không phải lớp người cố định trên toàn dataset.',
      graphic(embeddings(),80,310,1440,355)+t(120,720,1360,74,'Lịch sử: embedding slot đã biết  |  Hiện tại: embedding ID rỗng',30,TEAL,align='center'))
    slide('Attention liên kết phát hiện với lịch sử','Self-attention trong frame; cross-attention truy vấn lịch sử mang ID.',
      formula('attention',120,290,1360,155)+graphic(timeline(['Query hiện tại','Lịch sử có ID','FFN + ID head'],['Query','Key / Value','50 slot + NEW'],h=390),120,440,1360,365)+t(470,480,270,45,'Mask thời gian:',27,MUTED,middle=True)+formula('causal',760,475,350,55))
    slide('Huấn luyện ID theo ngữ cảnh, không nhớ ID toàn cục','Hoán vị slot và tăng cường quỹ đạo giúp học quy tắc liên kết.',
      graphic(timeline(['Clip có nhãn','GT → ID slot','Hoán vị + mask','CE hợp lệ'],['Lịch sử / hiện tại','Nhãn cục bộ','Mô phỏng lịch sử lỗi','Bỏ padding'],h=410),80,245,1440,410)+formula('base_loss',200,690,1200,125))
    inference_cards=''
    for i,(label,value,desc,col) in enumerate([
        ('LỌC PHÁT HIỆN','> 0,3','Điểm detector để đưa\nvào bước liên kết.',BLUE),
        ('TRACK MỚI','> 0,6','Điểm detector khi ID\nđược dự đoán là NEW.',TEAL),
        ('ID HIỆN CÓ','≥ 0,2','Điểm ID tối thiểu;\nslot vẫn phải hợp lệ.',CORAL),
        ('TRACK VẮNG','30 frame','Giới hạn lưu lịch sử\nkhi thiếu quan sát.',BLUE)]):
        x=80+i*360
        inference_cards+=shape(x,595,340,225,'#F2F5F8',18)
        inference_cards+=t(x+20,610,300,34,label,23,MUTED,True,'center')
        inference_cards+=t(x+20,647,300,62,value,50,col,True,'center')
        inference_cards+=t(x+20,715,300,90,desc,24,INK,False,'center')
    slide('Suy luận là một vòng đọc → quyết định → ghi','Dùng object-max; NEW và slot cũ được xử lý trong cùng vòng đời tracker.',
      graphic(timeline(['Đọc lịch sử','Dự đoán ID','Slot cũ / NEW','Ghi bộ nhớ'],['Chỉ thời điểm < t','Xử lý xung đột','Sinh / duy trì track','Sau quyết định'],h=410),80,225,1440,350)+inference_cards)
    slide('Bổ sung geometry như một tín hiệu tương thích riêng','Không thay detector hay ID Decoder; chấm trực tiếp từng cặp phát hiện–track.',
      shape(85,275,660,380,'#F2F5F8',22)+shape(855,275,660,380,'#EEF7F5',22)+
      t(120,300,590,62,'MOTIP nền',43,BLUE,True,'center')+
      t(130,390,570,105,'Đặc trưng đối tượng\n+ lịch sử mang ID',34,INK,False,'center')+
      t(280,490,280,64,'↓',52,BLUE,True,'center')+
      t(130,565,570,62,'Logit liên kết nền',37,BLUE,True,'center')+
      t(890,300,590,62,'Hình học G4',43,TEAL,True,'center')+
      t(900,390,570,105,'Hộp hiện tại\n+ quỹ đạo của track ứng viên',34,INK,False,'center')+
      t(1050,490,280,64,'↓',52,TEAL,True,'center')+
      t(900,565,570,62,'Điểm tương thích hình học',37,TEAL,True,'center')+
      t(752,425,96,100,'+',82,TEAL,True,'center')+
      t(100,710,1400,75,'Hai nhánh chấm cùng các ID ứng viên; logit được căn chỉnh rồi hợp nhất.',30,MUTED,False,'center'))
    slide('G4 mã hóa lịch sử hộp thành mô tả hình học 66 chiều','Mỗi cặp phát hiện–track có một vector; lag lấy theo timestamp và đi kèm validity mask.',
      img('figures/multilag.png',45,258,900,465)+
      t(975,278,545,42,'Hộp và vị trí',30,BLUE,True)+
      t(975,318,545,62,'Hộp hiện tại + bốn hộp quá khứ; dx, dy chuẩn hóa theo hộp hiện tại.',23)+
      t(975,395,545,42,'Kích thước',30,TEAL,True)+
      t(975,435,545,58,'Log thay đổi chiều rộng, chiều cao và diện tích.',23)+
      t(975,510,545,42,'Chuyển động',30,CORAL,True)+
      t(975,550,545,58,'Vận tốc, dư ngoại suy và biến thiên vận tốc.',23)+
      t(975,625,545,42,'Thời gian và dữ liệu thiếu',30,BLUE,True)+
      t(975,665,545,62,'Mã lag cùng validity mask; không thay lag thiếu bằng quan sát gần nhất.',23))
    slide('G4–MLP bổ sung điểm hình học vào logit ID','25.217 tham số; nhiệt độ và trọng số fusion giữ cố định.',
      img('figures/geometry_fusion.png',80,255,1440,575))
    slide('Tiền huấn luyện: đổi nguồn hộp, giữ cùng một MLP','Chỉ dữ liệu train; mã ID dùng để tạo nhãn, không nằm trong vector G4.',
      graphic(pretraining_process(),60,255,1480,510)+t(100,780,1400,42,'Pred–Pred giữ nguyên tọa độ detector; GT chỉ giúp nối lịch sử và tạo nhãn cùng/khác ID.',25,TEAL,align='center'))
    slide('Hàm mất mát tổng giám sát cả hai đầu ra','Khi học liên kết, ID của track mục tiêu giám sát điểm hợp nhất và điểm hình học.',
      formula('total_loss',140,300,1320,115)+formula('loss_parts',130,475,790,210)+
      t(1010,475,510,65,'Điểm hợp nhất',34,BLUE,True)+t(1010,540,510,85,'Học quyết định ID cuối cùng\ntrên các query hợp lệ.',28)+
      t(1010,650,510,60,'Điểm hình học',34,TEAL,True)+t(1010,710,510,90,'Chỉ giám sát track hiện hữu\ncó lag hợp lệ; loại NEW.',28))
    bars=''
    for j,(b,n) in enumerate([(5,2090),(10,4179),(15,6265)]):
        width=900*b/15
        bars+=f'<rect x="200" y="{35+j*115}" width="{width}" height="65" fill="{TEAL}" opacity="{.50+.23*j}"/><text x="150" y="{80+j*115}" font-size="37" text-anchor="end">{b}%</text><text x="{230+width}" y="{79+j*115}" font-size="32">{n:,} frame</text>'
    slide('Ba ngân sách lồng nhau, một tập đánh giá chung','Subset nguyên sequence; không lấy frame rời rạc và không trùng train/validation.',
      graphic(svg_base(bars,1400,390),80,270,1440,390)+formula('nested',430,680,740,75)+t(120,786,1360,45,'Full validation: 25.508 frame  |  chọn checkpoint trên small validation 3.894 frame',25,MUTED,align='center'))
    slide('Đối chiếu MOTIP nền với MOTIP + G4','Cùng detector, dữ liệu, augmentation, lifecycle và quy tắc đánh giá.',
      table(['Thiết lập chung','Cấu hình'],[['Seed / epoch','42, 123, 3407 / 10'],['Batch / accumulation / precision','1 / 4 / FP32'],['Optimizer / LR / weight decay','AdamW / 0,0001 / 0,0005'],['LR milestone','6 và 9, hệ số 0,1'],['Phần được học','Trajectory modeling + ID Decoder + ID embedding']],h=475,size=29)+t(120,770,1360,52,'MOTIP + G4 có thêm G4–MLP và bước tiền huấn luyện train-only.',29,TEAL,align='center'))
    body=''
    for i,m in enumerate(['HOTA','AssA','IDF1']):body+=chart(result_spec(m),80+i*485,285,470,475)
    body+=t(120,795,1360,35,'Trung bình ± độ lệch chuẩn mẫu, n = 3; checkpoint chọn bằng small-validation AssA.',24,MUTED,align='center')
    slide('MOTIP + G4 cải thiện HOTA, AssA và IDF1','Cùng toàn bộ DanceTrack validation; không đối chiếu với bảng SOTA test.',body)
    body=''
    for i,b in enumerate([5,10,15]):
        gain=stats[(b,'v3_gtlinked')]['AssA']['mean']-stats[(b,'e0')]['AssA']['mean'];x=100+i*500
        body+=t(x,300,430,70,f'Train {b}%',35,MUTED,align='center')+t(x,405,430,135,f'+{gain:.2f}',96,TEAL,True,'center')+t(x,565,430,70,'điểm AssA',34,INK,align='center')
        body+=t(x,690,430,90,f"MOTIP nền {stats[(b,'e0')]['AssA']['mean']:.2f}\nMOTIP + G4 {stats[(b,'v3_gtlinked')]['AssA']['mean']:.2f}",30,INK,align='center')
    slide('Mức tăng AssA đạt +7,36 đến +8,82 điểm','Chênh lệch trên thang 0–100, không phải phần trăm tương đối.',body)
    ds=[]
    for seed in [42,123,3407]:
        vals=[]
        for b in [5,10,15]:
            p={r['model']:float(r['AssA']) for r in rows if int(r['budget_pct'])==b and int(r['seed'])==seed};vals.append(p['v3_gtlinked']-p['e0'])
        ds.append(vals)
    spec=dict(kind='bar',title='ΔAssA (MOTIP + G4 − MOTIP nền)',categories=['5%','10%','15%'],names=['Seed 42','Seed 123','Seed 3407'],colors=[BLUE,TEAL,CORAL],values=ds)
    slide('Cả chín cặp seed/ngân sách đều cải thiện AssA','Ba seed hỗ trợ nhận xét nhất quán, chưa phải bằng chứng thống kê dứt khoát.',chart(spec,140,265,1320,530))
    slide('ID switch giảm, chi phí tính toán vẫn có','Đọc IDSW cùng AssA/IDF1; không đồng nhất FPS phát video với FPS suy luận.',chart(result_spec('IDSW'),80,280,700,480)+chart(result_spec('DetA'),830,280,690,480)+t(100,790,1400,35,'Thêm 25.217 tham số; FPS seed 42 trên cùng máy giảm khoảng 5–6%.',25,MUTED,align='center'))
    slide('G4 giữ ID khi các đối tượng giao cắt','dancetrack0004 · đối tượng GT 2 · checkpoint seed 42, train 15%.',
      img('figures/association_slide_case_1.png',75,260,1450,495)+t(100,770,1400,50,'MOTIP nền: ID 0 → 2     |     MOTIP + G4: giữ ID 0',32,TEAL,True,'center')+t(100,820,1400,25,'Ví dụ chọn có chủ đích; GT chỉ dùng kiểm chứng hậu nghiệm. Đường trắng: GT, đường xanh: dự đoán.',20,MUTED,align='center'))
    slide('G4 giữ ID trong vùng chồng lấn','dancetrack0034 · đối tượng GT 8 · cùng trang phục, vị trí gần nhau.',
      img('figures/association_slide_case_2.png',75,260,1450,495)+t(100,770,1400,50,'MOTIP nền: ID 387 → 442     |     MOTIP + G4: giữ ID 6',32,TEAL,True,'center')+t(100,820,1400,25,'Ví dụ chọn có chủ đích, không đại diện mọi trường hợp. Video demo tổng quát: 45 giây.',20,MUTED,align='center'))
    slide('Bằng chứng tích cực nhưng kết luận có phạm vi','Không đồng nhất validation trên subset với ưu thế phổ quát ở full-data/test.',
      shape(80,275,690,450,'#EAF5F3',24)+shape(830,275,690,450,'#FBF0EE',24)+
      t(125,315,600,65,'Đã quan sát',44,TEAL,True)+
      t(130,410,170,75,'9/9',48,TEAL,True)+t(310,420,410,65,'cặp tăng AssA',31,INK)+
      t(130,510,170,75,'9/9',48,TEAL,True)+t(310,520,410,65,'cặp giảm IDSW',31,INK)+
      t(130,615,550,55,'Detector được giữ cố định',28,INK,True)+
      t(875,315,600,65,'Chưa chứng minh',44,CORAL,True)+
      t(880,410,600,90,'Ưu thế khi huấn luyện end-to-end\ntrên toàn bộ dữ liệu',29,INK)+
      t(880,510,600,65,'Khả năng tổng quát hóa trên test độc lập',29,INK)+
      t(880,610,600,90,'Hiệu quả khi cân bằng đầy đủ chi phí tính toán',29,INK)+
      t(120,755,1360,52,'MOTIP + G4 có chi phí pretrain riêng; validation dùng chọn checkpoint nằm trong full validation.',25,MUTED,align='center'))
    slide('Liên kết học được\n+ hình học tường minh','Một cải tiến nhỏ, kiểm chứng bằng dữ liệu và đối chứng matched.',
      img('figures/dataset_sample.jpg',850,330,690,390)+t(90,340,700,190,'Tái hiện MOTIP.\nThêm G4 theo ứng viên.\nĐánh giá trên ba ngân sách.',40,'white',True)+t(90,645,680,145,'Cải thiện nhất quán trong\nphạm vi đã khảo sát.',33,'#CFEDE8'),True)
    slide('Phụ lục: G4 gồm đúng 66 chiều','Độ dời chuẩn hóa theo hộp hiện tại; lag và các đặc trưng động học thiếu có mask riêng.',table(['Nhóm đặc trưng','Số chiều'],[['Hộp hiện tại + bốn hộp lịch sử','4 + 16'],['Độ dời tương đối + log tỷ lệ w/h','16'],['Vận tốc, dư ngoại suy + biến thiên vận tốc','8 + 4'],['Validity lag / vận tốc / gia tốc','4 + 4 + 2'],['Log thay đổi diện tích + mã lag','4 + 4'],['Tổng','66']],h=510,size=29))
    slide('IDF1 và IDSW đo hai khía cạnh khác nhau','IDF1 xem xét liên kết danh tính trên toàn chuỗi; IDSW đếm các lần đổi ID.',
      '<div style="position:absolute;left:80px;top:275px;width:690px;height:485px;background:#F2F5F8;border:2px solid #DCE5EC;border-radius:24px"></div>'+
      '<div style="position:absolute;left:830px;top:275px;width:690px;height:485px;background:#F2F5F8;border:2px solid #DCE5EC;border-radius:24px"></div>'+
      t(125,305,600,85,'IDF1',62,BLUE,True)+t(125,390,595,70,'Độ nhất quán danh tính trên toàn quỹ đạo.',27,INK)+
      t(125,495,115,50,'GT: A',25,MUTED,True)+t(125,595,115,50,'Pred',25,MUTED,True)+
      ''.join(shape(260+i*86,490,58,48,'#DDE6EF',6)+t(260+i*86,491,58,46,'A',22,BLUE,True,'center',True) for i in range(5))+
      ''.join(shape(260+i*86,590,58,48,'#DDEFEA',6)+t(260+i*86,591,58,46,'A',22,TEAL,True,'center',True) for i in range(5))+
      t(125,685,595,45,'Một ID dự đoán được giữ xuyên suốt chuỗi.',24,INK)+
      t(875,305,600,85,'IDSW',62,TEAL,True)+t(875,390,595,70,'Số lần một đối tượng được gán ID khác.',27,INK)+
      t(875,495,115,50,'GT: A',25,MUTED,True)+t(875,595,115,50,'Pred',25,MUTED,True)+
      ''.join(shape(1010+i*86,490,58,48,'#DDE6EF',6)+t(1010+i*86,491,58,46,'A',22,BLUE,True,'center',True) for i in range(5))+
      ''.join(shape(1010+i*86,590,58,48,'#DDEFEA' if i<2 else '#F5E4DC',6)+t(1010+i*86,591,58,46,'4' if i<2 else '8',22,BLUE if i<2 else CORAL,True,'center',True) for i in range(5))+
      t(1190,660,270,48,'1 lần đổi ID',24,CORAL,True,'center',True)+
      t(120,785,1360,38,'IDF1: cao hơn tốt hơn     •     IDSW: thấp hơn tốt hơn',25,MUTED,align='center'))
    r15=[[r['seed'],'MOTIP nền' if r['model']=='e0' else 'MOTIP + G4',r['epoch'],r['AssA'],r['IDF1'],f"{int(float(r['IDSW'])):,}"] for r in rows if r['budget_pct']=='15']
    slide('Phụ lục: tất cả seed tại ngân sách 15%','Epoch được chọn theo cùng quy tắc small-validation AssA.',table(['Seed','Model','Epoch','AssA','IDF1','IDSW'],r15,h=510,size=29))
    slide('Phần kế thừa và phần thực hiện được tách rõ','Đồ án sử dụng mã nguồn tác giả; không nhận phát minh MOTIP hay detector.',table(['Kế thừa','Thực hiện / tích hợp'],[['MOTIP, Deformable DETR, TrackEval','Nhánh G4 theo cặp + kiểm thử'],['Ý tưởng ID prediction của Gao và cộng sự','Matched MOTIP nền/MOTIP + G4, ba ngân sách, ba seed'],['Checkpoint detector tiền huấn luyện','GT-linked geometry pretrain train-only'],['Giấy phép và trích dẫn gốc','Code, manifest, checksum và demo']],h=445,size=29))
    slide('Chạy lại demo bằng checkpoint được bàn giao','Cần CUDA và extension deformable attention tương thích môi trường.',
      graphic(timeline(['Chuẩn bị','Suy luận','Hiển thị'],['Môi trường + DanceTrack','Không đọc annotation','Render dự đoán đã lưu'],h=390),80,260,1440,380)+t(130,690,1340,115,'python scripts/run_demo_inference.py --model both\npython scripts/render_demo.py',29,INK))
    slide('Nguồn khoa học và mã nguồn chính','Liên kết đầy đủ, attribution và protocol nằm trong báo cáo.',
      t(110,290,1370,490,'Gao, Qi, Wang — Multiple Object Tracking as ID Prediction. CVPR 2025.\n\nZhu và cộng sự — Deformable DETR. ICLR 2021.\n\nSun và cộng sự — DanceTrack. CVPR 2022.\n\nLuiten và cộng sự — HOTA. International Journal of Computer Vision.\n\nMã nguồn: github.com/MCG-NJU/MOTIP; github.com/JonathonLuiten/TrackEval',30))
    assert len(scenes)==len(notes)==31
    css='''*{box-sizing:border-box}html,body{margin:0;background:#dfe6ed;font-family:Arial,sans-serif} .slide{position:relative;width:1600px;height:900px;background:white;overflow:hidden;display:none} .slide.active{display:block}.slide.dark{background:#18324A}.slide>*{position:absolute}.text{white-space:pre-wrap;line-height:1.19;overflow:visible}.table table{border-collapse:collapse;width:100%;height:100%;table-layout:fixed}.table td,.table th{text-align:center;vertical-align:middle;padding:12px 16px;line-height:1.18}.table th{background:#18324A;color:white;font-weight:700}.table tbody tr:nth-child(even){background:#EAF1F6}.table tbody tr:nth-child(odd){background:#F7F9FB}button{font:18px Arial;padding:8px 16px}#nav{position:fixed;right:10px;bottom:5px;z-index:100;opacity:.2}#nav:hover{opacity:1}@media print{html,body{background:white}.slide{display:block!important;break-after:page}#nav{display:none}@page{size:1600px 900px;margin:0}}'''
    css+='''html,body{font-family:Arial,sans-serif}.slide{transform-origin:top left}@media screen{body{overflow:hidden}}@media print{.slide{transform:none!important}}'''
    js='''let current=0;const slides=[...document.querySelectorAll('.slide')];const fit=()=>{const z=Math.min(innerWidth/1600,innerHeight/900);slides.forEach(s=>s.style.transform='scale('+z+')')};window.showSlide=i=>{current=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,j)=>s.classList.toggle('active',j===current));location.hash='slide-'+(current+1)};fit();addEventListener('resize',fit);showSlide(0);addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key))showSlide(current+1);if(['ArrowLeft','PageUp'].includes(e.key))showSlide(current-1);if(e.key==='Home')showSlide(0)});'''
    page='<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>MOTIP — Thuyết trình Thị giác máy tính</title><style>'+css+'</style></head><body>'+''.join(scenes)+'<div id="nav"><button onclick="showSlide(current-1)">←</button><button onclick="showSlide(current+1)">→</button></div><script>'+js+'</script></body></html>'
    (OUT/'index.html').write_text(page)
    (ROOT/'presentation/slide_structure.json').write_text(json.dumps(notes,indent=2,ensure_ascii=False)+'\n')
    speech=['# Kịch bản thuyết trình — 20 phút, gồm video','','25 slide chính, 6 phụ lục. Lời trình bày: 19 phút 15 giây; video demo: 45 giây.','']
    elapsed=0
    for row in notes:
        if row['backup']:continue
        speech.extend([f"## Slide {row['number']}. {row['title']}",'',f"Mốc {elapsed//60:02d}:{elapsed%60:02d} · khoảng {row['seconds']} giây",'',row['notes'],''])
        elapsed+=row['seconds']
        if row.get('video_after'):
            speech.extend(['### Chiếu video demo (45 giây)','',f"Mốc {elapsed//60:02d}:{elapsed%60:02d} · demo/DEMO_MOTIP_DanceTrack.mp4",'', 'Đây là demo quy trình suy luận tổng quát, không phải hai cửa sổ GT vừa phân tích.',''])
            elapsed+=45
    assert elapsed==1200
    (ROOT/'presentation/KICH_BAN_20_PHUT.md').write_text('\n'.join(speech)+'\n')
    print('HTML deck created: 31 slides. Next: export_html_pptx.py')


if __name__=='__main__':main()
