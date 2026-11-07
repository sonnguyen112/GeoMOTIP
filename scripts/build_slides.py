#!/usr/bin/env python3
"""Editable, native-shape Vietnamese slides with 20-minute speaker notes."""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'presentation'
INK='15304C';TEAL='008D91';BLUE='3C6A96';CORAL='E48054';BG='F6F8FB';MUTED='52667A';WHITE='FFFFFF'
P=Presentation();P.slide_width=Inches(13.333);P.slide_height=Inches(7.5)
P.core_properties.author='Nguyễn Hồ Trường Sơn — 25C01020'
P.core_properties.title='MOTIP và hình học quỹ đạo đa độ trễ'
P.core_properties.subject='Đồ án Thị giác máy tính — TS. Trương Phước Hưng'
FONT='DejaVu Sans'
SPEECH=[]


def rgb(value):return RGBColor.from_string(value)


def rect(slide,x,y,w,h,fill,line=None,radius=False):
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    shape.fill.solid();shape.fill.fore_color.rgb=rgb(fill)
    shape.line.fill.background() if line is None else None
    if line:shape.line.color.rgb=rgb(line)
    return shape


def text(slide,x,y,w,h,value,size=22,color=INK,bold=False,align=PP_ALIGN.LEFT,valign=MSO_ANCHOR.TOP):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=shape.text_frame;tf.word_wrap=True;tf.vertical_anchor=valign
    tf.margin_left=tf.margin_right=Inches(.015);tf.margin_top=tf.margin_bottom=Inches(.015)
    for i,line in enumerate(value.split('\n')):
        para=tf.paragraphs[0] if i==0 else tf.add_paragraph();para.text=line
        para.alignment=align;para.font.name=FONT;para.font.size=Pt(size);para.font.bold=bold;para.font.color.rgb=rgb(color)
        para.space_after=Pt(8)
    return shape


def add(title,section,subtitle='',notes='',seconds=50,backup=False):
    slide=P.slides.add_slide(P.slide_layouts[6]);slide.background.fill.solid();slide.background.fill.fore_color.rgb=rgb(BG)
    text(slide,.63,.25,12,.3,section.upper(),11,TEAL,True)
    text(slide,.63,.68,12.0,.8,title,30,INK,True)
    if subtitle:text(slide,.65,1.46,12,.48,subtitle,14,MUTED)
    text(slide,.65,7.12,10,.25,'MOTIP · Thị giác máy tính | Nguyễn Hồ Trường Sơn · 25C01020',9,MUTED)
    text(slide,12.0,7.10,.65,.25,f'{len(P.slides):02d}',10,MUTED,align=PP_ALIGN.RIGHT)
    note=f'{"PHỤ LỤC HỎI ĐÁP" if backup else "THUYẾT TRÌNH CHÍNH"}\nThời lượng gợi ý: {seconds} giây\n\n{notes}'
    slide.notes_slide.notes_text_frame.text=note
    SPEECH.append(dict(number=len(P.slides),title=title,seconds=seconds,backup=backup,notes=notes))
    return slide


def image(slide,path,x=.65,y=2.0,w=12,h=4.75):
    iw,ih=Image.open(path).size;scale=min(w/iw,h/ih);sw,sh=iw*scale,ih*scale
    slide.shapes.add_picture(str(path),Inches(x+(w-sw)/2),Inches(y+(h-sh)/2),width=Inches(sw),height=Inches(sh))


def card(slide,x,y,w,h,title,body,color=TEAL,size=21):
    rect(slide,x,y,w,h,WHITE,radius=True)
    text(slide,x+.22,y+.18,w-.44,.6,title,22,color,True,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)
    text(slide,x+.22,y+.96,w-.44,h-1.12,body,size,INK,False,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)


def equation(slide,name,x,y,w,h):
    image(slide,ROOT/f'figures/math/{name}.png',x,y,w,h)


def table(slide,headers,rows,x=.65,y=2.12,w=12,h=4.25,fontsize=16):
    shape=slide.shapes.add_table(len(rows)+1,len(headers),Inches(x),Inches(y),Inches(w),Inches(h))
    t=shape.table
    for ri,row in enumerate([headers]+rows):
        for ci,value in enumerate(row):
            cell=t.cell(ri,ci);cell.text=str(value)
            cell.fill.solid();cell.fill.fore_color.rgb=rgb(INK if ri==0 else WHITE if ri%2 else 'EAF2F6')
            cell.margin_left=Inches(.12);cell.margin_right=Inches(.09);cell.vertical_anchor=MSO_ANCHOR.MIDDLE
            for para in cell.text_frame.paragraphs:
                para.alignment=PP_ALIGN.CENTER
                para.font.name=FONT;para.font.size=Pt(fontsize);para.font.color.rgb=rgb(WHITE if ri==0 else INK);para.font.bold=ri==0
    return t


def flow(slide,labels,y=2.8):
    gap=.18;w=(12-gap*(len(labels)-1))/len(labels)
    for i,label in enumerate(labels):
        x=.65+i*(w+gap);rect(slide,x,y,w,1.45,WHITE,radius=True)
        math_name={'Ảnh I_t':'image','q_n, b_n, f_n':'detection_output'}.get(label)
        if math_name:equation(slide,math_name,x+.15,y+.17,w-.3,1.1)
        else:text(slide,x+.12,y+.10,w-.24,1.25,label,22,TEAL if i%2 else INK,True,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)
        if i<len(labels)-1:text(slide,x+w,y+.5,gap,.4,'›',24,TEAL,True)


def main():
    OUT.mkdir(exist_ok=True)
    data=json.loads((ROOT/'experiments/presentation_summary.json').read_text())
    stats={(r['budget'],r['model']):r for r in data['stats']}
    s=add('MOTIP: từ phát hiện đến dự đoán ID','Đồ án cuối kỳ','Tái hiện phương pháp và bổ sung hình học quỹ đạo đa độ trễ',
        notes='Em trình bày bài báo Multiple Object Tracking as ID Prediction của Gao và cộng sự, công bố tại CVPR 2025. Đồ án dựa trên mã nguồn tác giả, tái hiện pipeline và đánh giá thêm một nhánh hình học nhỏ. Phần thực nghiệm là các run đã huấn luyện và được đối chiếu artifact, không phải số liệu lấy từ bảng SOTA. Em sẽ đi từ bài toán, cách MOTIP dự đoán ID, phần bổ sung geometry, rồi đến thiết kế và kết quả thực nghiệm.',seconds=30)
    rect(s,.15,2.05,13.183,4.85,INK)
    text(s,.8,2.5,5.9,2.2,'Theo dõi đúng vị trí\nchưa đủ.\nCần giữ đúng danh tính.',34,WHITE,True,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)
    image(s,ROOT/'figures/dataset_sample.jpg',7.0,2.55,5.5,3.1)
    text(s,.8,5.75,6,.65,'Nguyễn Hồ Trường Sơn · 25C01020\nGiảng viên: TS. Trương Phước Hưng',16,WHITE,False,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)
    s=add('Bài toán: ai là ai qua thời gian?','01 / Bài toán','Đầu ra mỗi frame: hộp giới hạn + ID nhất quán.',
        notes='Mỗi frame có thể phát hiện đúng tất cả người, nhưng khi hai người đi ngang qua nhau thì tracker vẫn có thể đổi danh tính. Vì vậy phát hiện và liên kết là hai vấn đề khác nhau. Ví dụ một ID ban đầu theo người A rồi chuyển sang người B là lỗi ID switch. Đầu vào là video, đầu ra là các hộp và ID qua thời gian. Thử thách chính của DanceTrack là ngoại hình tương tự, chuyển động thay đổi và che khuất. Các ID ở đây là định danh tạm cho quỹ đạo, không phải nhận dạng cá nhân.',seconds=45)
    card(s,.65,2.2,3.9,3.95,'Phát hiện','Người ở đâu?\nHộp + confidence\nĐúng trên từng ảnh',BLUE)
    card(s,4.72,2.2,3.9,3.95,'Liên kết','Là người nào trước đó?\nKhai thác lịch sử\nĐúng xuyên thời gian',TEAL)
    card(s,8.79,2.2,3.9,3.95,'Tình huống khó','Ngoại hình giống nhau\nGiao cắt, che khuất\nLịch sử dự đoán bị sai',CORAL)
    s=add('DanceTrack: ngoại hình giống, chuyển động khác','02 / Dataset','Dữ liệu thực nghiệm chính: tracking người trong các video múa/nhảy.',
        notes='DanceTrack được thiết kế để làm liên kết khó hơn so với tình huống chỉ cần phân biệt áo quần. Workspace có 40 sequence train với 41.796 frame và 25 sequence validation với 25.508 frame. Đồ án dùng full validation cho các bảng đánh giá. Ba ngân sách train là các subset nguyên sequence, không lấy frame rời rạc. Do đây là validation và có một tập con dùng chọn checkpoint, em không coi kết quả là test độc lập hay đối chiếu trực tiếp với bảng test của paper.',seconds=50)
    image(s,ROOT/'figures/dataset_sample.jpg',.65,2.12,7.0,4.55)
    card(s,8.0,2.25,4.6,1.9,'Train','40 sequence · 41.796 frame',BLUE,size=20)
    card(s,8.0,4.45,4.6,1.9,'Full validation','25 sequence · 25.508 frame',TEAL,size=20)
    s=add('MOTIP: liên kết là dự đoán ID theo ngữ cảnh','03 / Thuật toán paper','Lịch sử đã có ID đóng vai trò ngữ cảnh cho các phát hiện hiện tại.',
        notes='Đây là hình tổng quan của MOTIP. Detector tạo đặc trưng cùng hộp. Lịch sử được lưu với ID cục bộ. Decoder nhận phát hiện chưa biết ID và truy vấn lịch sử để dự đoán ID tương ứng. Sau khi gán ID thì quan sát mới được ghi lại cho frame tiếp theo. Điểm quan trọng là vòng phản hồi: một quyết định sai có thể đi vào lịch sử và ảnh hưởng các quyết định sau. Hình này mô tả MOTIP nền; phần geometry sẽ được thêm ở mức logit sau, không thay detector hay ID Decoder.',seconds=60)
    image(s,ROOT/'figures/pipeline.png')
    s=add('Bước 1 — detector tạo hộp và đặc trưng','03 / Thuật toán paper','Deformable DETR R50; mỗi phát hiện có một vector 256 chiều.',
        notes='Deformable DETR xử lý ảnh để sinh hộp, confidence và đặc trưng query đối tượng. Sau lọc ngưỡng, một phát hiện được biểu diễn bằng hộp chuẩn hóa cx, cy, w, h và vector 256 chiều. Đây không phải embedding ID cuối cùng. Trong thực nghiệm môn học, em dùng detector tiền huấn luyện được giữ cố định để tập trung đánh giá việc liên kết. Paper có thiết lập end-to-end; kết quả của em không được mô tả như đã tái lập hoàn toàn ngân sách huấn luyện đó.',seconds=50)
    flow(s,['Ảnh I_t','Deformable\nDETR','q_n, b_n, f_n','Lọc phát hiện'])
    equation(s,'detection',1,5.0,11,1.0)
    s=add('Bước 2 — tổ chức lịch sử theo track và thời gian','03 / Thuật toán paper','Trajectory modeling thích nghi đặc trưng cho liên kết, không phải motion predictor.',
        notes='Lịch sử có dạng batch, group, time, track, feature. Group ở đây là các nhóm tăng cường quỹ đạo trong huấn luyện, không phải số người. Mỗi track có quan sát ở nhiều thời điểm nhưng có thể bị thiếu; mask giúp attention bỏ qua những vị trí không hợp lệ. Module trajectory modeling gồm feed-forward, residual và LayerNorm để thích nghi đặc trưng detector. Không nên nhầm nó với một mô hình dự đoán vị trí tương lai. Khi suy luận, lịch sử này được xây từ ID do tracker tự dự đoán.',seconds=60)
    card(s,.65,2.15,7.4,3.9,'Tensor lịch sử','',TEAL,size=23)
    equation(s,'history',1.0,3.25,6.7,2.1)
    card(s,8.25,2.15,4.35,3.9,'Mask có vai trò gì?','Padding, thiếu quan sát\nkhông được coi là\ntrack hợp lệ.\n\nChỉ dùng thời điểm < t.',CORAL,size=21)
    s=add('Bước 3 — ghép ngoại hình với ID embedding','03 / Thuật toán paper','ID là slot trong từ vựng cục bộ, không phải lớp người trên toàn dataset.',
        notes='Một token lịch sử ghép đặc trưng 256 chiều với ID embedding 256 chiều. Phát hiện mới thì chưa biết ID nên dùng embedding rỗng học được. Token vào decoder do đó có 512 chiều. ID slot được hoán vị trong huấn luyện để mô hình không ghi nhớ rằng một người luôn mang số cụ thể. Cấu hình này có 50 slot và thêm lớp NEW. ID slot được tái sử dụng nhưng ID toàn cục dùng xuất quỹ đạo là một khái niệm khác, nên phải quản lý reset lịch sử khi tái cấp slot.',seconds=60)
    flow(s,['Đặc trưng\n256 chiều','ID embedding\n256 chiều','Ghép nối\n512 chiều'])
    text(s,.95,5.0,11.6,1.0,'Lịch sử: embedding của slot đã biết\nHiện tại: embedding ID rỗng → cần dự đoán',24,TEAL,True)
    s=add('Bước 4 — ID Decoder khai thác quan hệ','03 / Thuật toán paper','Attention trong frame + truy vấn lịch sử + mask nhân quả.',
        notes='Self-attention giúp các phát hiện hiện tại tương tác với nhau; cross-attention để chúng truy vấn các token lịch sử có ID. Thông tin thời gian tương đối được thêm vào attention, còn mask chặn các quan sát hiện tại hoặc tương lai. Có sáu lớp decoder trong cấu hình được dùng; mã nguồn bỏ self-attention ở lớp đầu nên hình ở đây là sơ đồ chức năng, không khẳng định mọi lớp giống hệt nhau. Đầu phân loại sinh logit cho 50 slot và NEW. Lưu ý đây vẫn là một bài toán dự đoán trong ngữ cảnh, không có một classifier riêng cho mỗi người của dataset.',seconds=70)
    flow(s,['Query hiện tại','Self-attention\ntrong frame','Cross-attention\nvới lịch sử','FFN + ID head'],y=2.4)
    equation(s,'attention',.9,4.05,11.5,.75)
    card(s,1.0,5.0,5.4,1.65,'Ngữ cảnh','Key / Value: đặc trưng + ID quá khứ',BLUE,size=17)
    card(s,6.65,5.0,5.65,1.65,'Nhân quả','',TEAL,size=18)
    equation(s,'causal',7.1,5.95,4.75,.4)
    s=add('Huấn luyện: học ID tương đối từ clip','03 / Thuật toán paper','Không dùng raw GT identity làm chỉ số lớp cố định.',
        notes='Quy trình lấy clip có nhãn, ghép phát hiện với GT và tạo nhãn slot cục bộ. Các hoán vị và tăng cường mô phỏng quỹ đạo thiếu hoặc sai. Loss ID là cross-entropy trên vị trí hợp lệ; padding và frame chưa có ngữ cảnh không được tính như các query bình thường. Với E0, detector đóng băng còn trajectory modeling, ID Decoder và ID embedding được học. Việc dùng GT ở bước tạo nhãn train là hợp lệ; khi inference không được dùng GT để nối track. Đây là ranh giới cần phân biệt rõ khi giải thích tiền huấn luyện geometry ở phần sau.',seconds=60)
    flow(s,['Clip có nhãn','Ghép GT +\nID slot cục bộ','Hoán vị ID +\nmask quỹ đạo','CE trên\nquery hợp lệ'],y=2.5)
    equation(s,'base_loss',1.3,4.8,10.7,1.3)
    s=add('Suy luận: NEW, gán slot và cập nhật bộ nhớ','03 / Thuật toán paper','Quyết định hiện tại được thực hiện trước khi ghi quan sát vào lịch sử.',
        notes='Ở frame đầu, các phát hiện đủ điều kiện được khởi tạo track mới. Sau đó tracker dự đoán ID từ lịch sử, giải quyết xung đột khi các phát hiện cùng thích một slot, và dùng NEW khi cần sinh track. Cấu hình đang đánh giá dùng object-max, không phải luôn dùng Hungarian ở bước suy luận. Các ngưỡng là detector 0,3, newborn 0,6 và ID 0,2. Track có thể thiếu quan sát trong cửa sổ 30 frame, nhưng khi slot được tái sử dụng thì lịch sử cũ phải bị xóa. Cuối cùng chỉ ảnh và trạng thái dự đoán đi vào bước suy luận, không dùng GT.',seconds=60)
    flow(s,['Phát hiện','ID logits','Slot cũ / NEW','Xuất ID\n+ ghi lịch sử'],y=2.5)
    card(s,1.0,4.75,11.4,1.65,'Các ngưỡng được giữ giống nhau','det = 0,3  ·  newborn = 0,6  ·  ID = 0,2  ·  miss tolerance = 30',TEAL,size=20)
    s=add('Cải tiến: cung cấp geometry tường minh','04 / Phương pháp bổ sung','Một tín hiệu tương thích riêng cho từng phát hiện–track.',
        notes='MOTIP có thể học quan hệ liên kết từ đặc trưng, nhưng với dữ liệu hạn chế, cung cấp quan hệ hình học trực tiếp là một giả thuyết đáng kiểm tra. Chúng em không thay ID Decoder hay thêm một Transformer lớn. Với mỗi phát hiện và mỗi track hiện hữu, nhánh nhỏ kiểm tra vị trí, kích thước và lịch sử gần đây có phù hợp hay không. Đây là giả thuyết và một cải tiến thực nghiệm của đồ án; không tuyên bố ý tưởng geometry trong tracking là hoàn toàn mới, cũng không tuyên bố nó luôn có lợi ở mọi quy mô dữ liệu.',seconds=45)
    card(s,.65,2.25,5.85,3.85,'MOTIP nền','Đặc trưng đối tượng\n+ ngữ cảnh ID\n→ điểm liên kết nền',BLUE,size=25)
    card(s,6.75,2.25,5.85,3.85,'Nhánh bổ sung G4','Quan hệ hộp hiện tại / lịch sử\n+ validity theo lag\n→ điểm geometry',TEAL,size=24)
    s=add('G4: giữ chính xác các lag 1, 2, 4, 8','04 / Phương pháp bổ sung','Missing history được biểu diễn bằng mask, không sao chép hộp gần nhất.',
        notes='Đây là phần đặc trưng của nhánh G4. Hộp tại t trừ bốn phải thực sự cách bốn frame, chứ không phải phần tử thứ tư trong bộ nhớ. Nếu bị che khuất ở frame đó thì lag bị thiếu và có mask riêng. Vector gồm 66 chiều: hộp hiện tại, hộp lịch sử, độ dời, log-scale, các vận tốc và phần dư ngoại suy, biến thiên vận tốc và mã lag. Trong code độ dời được chuẩn hóa theo kích thước hộp hiện tại, không phải hộp lịch sử. Các phép biến đổi này là tọa độ ảnh, chưa phải chuyển động 3D hay bird-eye view.',seconds=65)
    image(s,ROOT/'figures/multilag.png')
    s=add('MLP 25.217 tham số và residual-logit fusion','04 / Phương pháp bổ sung','Chấm trực tiếp cặp ứng viên; không biến geometry thành trail token.',
        notes='Mô tả G4 đi qua MLP 66 đến 128 đến 128 đến một, với GELU ở hai tầng ẩn. Mean và std chỉ được tính từ train. Điểm của mỗi cặp được căn đúng local slot của MOTIP rồi cộng vào logit nền đã chia nhiệt độ. Cấu hình được giữ cố định là Tb bằng hai, Tg bằng 0,5 và lambda bằng một. Với NEW, geometry đóng góp bằng không; MOTIP vẫn chịu trách nhiệm bằng chứng newborn. Đây là phép fusion đơn giản và có siêu tham số, là một giới hạn chứ không nên trình bày như cơ chế tự hiệu chỉnh hoàn toàn.',seconds=60)
    flow(s,['G4\n66 chiều','MLP + GELU\n66 → 128 → 128 → 1','Điểm geometry\ntheo ứng viên'],y=2.3)
    equation(s,'fusion',2.0,4.0,9.3,1.5)
    equation(s,'new_neutral',3.1,5.85,7.1,.62)
    s=add('Giám sát geometry: học riêng rồi tích hợp','04 / Phương pháp bổ sung','Chỉ dùng GT của train để liên kết các bbox dự đoán khi tiền huấn luyện.',
        notes='MLP được tiền huấn luyện bốn epoch GT–GT rồi bốn epoch với bbox detector được nối bằng ID của GT train. Không đọc validation trajectory để fit nhánh này. Hai giai đoạn dùng BCE có trọng số cho mất cân bằng. Khi huấn luyện tracker thì dùng CE phụ cho target track hiện hữu có lag hợp lệ. Loss thực chạy là CE của logit hợp nhất cộng CE geometry, mỗi trọng số bằng một. Không cộng thêm một CE base riêng nữa vì ID loss hiện nhận chính logit fused. Thêm direct loss để geometry không chỉ nhận gradient qua đường residual.',seconds=60)
    card(s,.65,2.2,5.85,3.85,'Tiền huấn luyện MLP','GT–GT → GT-linked Pred–Pred\nBCEWithLogits cân bằng\nTrain-only, không gọi ID Decoder',TEAL,size=22)
    card(s,6.75,2.2,5.85,3.85,'Huấn luyện tracker','',BLUE,size=22)
    equation(s,'total_loss',7.05,3.23,5.25,.65)
    equation(s,'loss_parts',7.2,4.08,4.95,1.0)
    text(s,7.1,5.36,5.15,.35,'NEW không nhận giám sát geometry',15,INK,False,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)
    s=add('Ba ngân sách train, một tập đánh giá chung','05 / Thiết kế thí nghiệm','Các tập con lồng nhau ở mức sequence; giữ nguyên liên tục thời gian.',
        notes='Ba ngân sách lần lượt là 2.090, 4.179 và 6.265 frame, tương đương khoảng năm, mười và mười lăm phần trăm của 41.796 frame train. Có ba, sáu và tám sequence, các tập nhỏ nằm trong tập lớn. Lựa chọn dựa trên metadata chứ không nhìn mức cải thiện. Tất cả được đánh giá trên cùng 25.508 frame validation. Chọn checkpoint theo một tập con cố định 3.894 frame, nhưng tập đó vẫn nằm trong full validation. Vì vậy em sẽ gọi đúng đây là kết quả validation, không phải test độc lập.',seconds=50)
    table(s,['Ngân sách','Frame train','Sequence train','Full validation'],[['5%','2.090','3','25.508 frame'],['10%','4.179','6','25.508 frame'],['15%','6.265','8','25.508 frame']],h=3.2,fontsize=20)
    equation(s,'nested',3.2,5.6,6.9,.65)
    text(s,1,6.3,11.3,.35,'Không trùng sequence với validation',17,TEAL,False,PP_ALIGN.CENTER)
    s=add('Đối chiếu matched E0 và V3','05 / Thiết kế thí nghiệm','Cùng detector, dữ liệu, chính sách huấn luyện và ngưỡng vòng đời.',
        notes='Trong từng ngân sách, E0 và V3 dùng cùng detector, batch một, accumulation bốn, FP32, AdamW và lịch học. Ba seed là 42, 123, 3407; mỗi run mười epoch. Điểm khác chủ đích là nhánh geometry và bước tiền huấn luyện của nó. Các run được tái sử dụng có phép kiểm tra nguồn của cả mười tám bảng summary. Cùng epoch không có nghĩa cùng số update giữa các ngân sách; và geometry có chi phí pretraining thêm. Đây là các giới hạn cần nói rõ khi diễn giải hiệu quả dữ liệu.',seconds=50)
    table(s,['Thành phần','Cấu hình'],[['Seed / epoch','42, 123, 3407 / 10'],['Batch / accumulation / precision','1 / 4 / FP32'],['Optimizer / LR','AdamW / 0,0001'],['Detector','Pretrained COCO + DanceTrack; frozen'],['Phần train','Trajectory modeling + ID decoder + ID embedding; thêm G4 ở V3']],h=4.2,fontsize=18)
    s=add('Kết quả: HOTA, AssA và IDF1 cùng tăng','06 / Kết quả đánh giá','Trung bình ± độ lệch chuẩn mẫu của ba seed; cùng full validation.',
        notes='Hình cho thấy đường của V3 nằm trên E0 ở cả ba ngân sách. AssA là chỉ số chính vì câu hỏi tập trung vào association. Các khoảng sai số là độ lệch chuẩn mẫu của ba seed, không phải confidence interval. Kết quả ở năm phần trăm tăng khoảng 7,36 điểm AssA, ở mười phần trăm khoảng 8,82, và mười lăm phần trăm khoảng 7,53. Không có một xu hướng đơn điệu hoàn hảo giữa ba mức, nên em không dùng hình này để kết luận một quy luật scaling tổng quát.',seconds=65)
    image(s,ROOT/'figures/quality_vs_budget.png',y=2.15,h=4.35)
    s=add('Mức cải thiện AssA: +7,36 đến +8,82 điểm','06 / Kết quả đánh giá','Không trộn với các thí nghiệm legacy khác protocol.',
        notes='Bảng này dùng chính số liệu của đồ án được tổng hợp từ artifact đã kiểm tra. Điểm cần nhấn mạnh là cả mean và độ lệch chuẩn, không chỉ một seed đẹp. Ở mười phần trăm, E0 khoảng 26,69 còn V3 khoảng 35,51; ở mười lăm phần trăm E0 30,73 còn V3 38,25. Chênh lệch nhỏ hơn nhiều so với một số kết quả nghiên cứu cũ khác protocol, nên em không đưa các kết quả cũ đó vào bảng chính. Chọn checkpoint là small-validation AssA như đã trình bày, không chọn trên test.',seconds=50)
    rows=[]
    for b in [5,10,15]:
        a=stats[(b,'e0')]['AssA'];v=stats[(b,'v3_gtlinked')]['AssA']
        rows.append([f'{b}%',f"{a['mean']:.2f} ± {a['std']:.2f}",f"{v['mean']:.2f} ± {v['std']:.2f}",f"+{v['mean']-a['mean']:.2f}"])
    table(s,['Train','E0 AssA','V3 AssA','ΔAssA'],rows,h=3.2,fontsize=21)
    text(s,.9,5.8,11.5,.55,'Metric thang 0–100 · chênh lệch là điểm, không phải % tương đối',18,MUTED)
    s=add('Cùng chiều ở cả 9 cặp seed/ngân sách','06 / Kết quả đánh giá','Ba seed hỗ trợ nhận xét nhất quán, chưa phải bằng chứng thống kê dứt khoát.',
        notes='Mỗi cột là V3 trừ E0 theo cùng seed và cùng ngân sách. Tất cả chín cặp đều dương về AssA và giảm IDSW. Nhưng ba ngân sách không độc lập hoàn toàn vì dữ liệu lồng nhau, và ba seed vẫn là số lượng nhỏ. Vì vậy em dùng cách diễn đạt là nhất quán trong các run đã quan sát, không khẳng định đã chứng minh thống kê hay luôn đúng trên mọi dataset. Dữ liệu ghép cặp cũng được bàn giao để tránh việc chỉ giữ những seed thuận lợi.',seconds=50)
    image(s,ROOT/'figures/paired_delta.png',y=2.0,h=4.65)
    s=add('Giảm IDSW, nhưng overhead không bằng không','06 / Kết quả đánh giá','Đọc IDSW cùng IDF1 và AssA; tách tốc độ xử lý khỏi FPS phát lại video.',
        notes='IDSW trung bình giảm ở cả ba mức dữ liệu, khoảng ba mươi ba đến bốn mươi bảy phần trăm. MLP chỉ có 25 nghìn tham số nhưng vẫn phải dựng và chấm các cặp ứng viên. Các cặp seed 42 trên cùng máy cho thấy FPS giảm vài phần trăm; không lấy FPS của máy trường trộn với máy local để kết luận. Trong demo mới chạy lại trên cùng sequence, em cũng ghi throughput thực tế riêng. Video phát lại ở tốc độ gốc không có nghĩa mô hình suy luận realtime ở tốc độ đó.',seconds=50)
    image(s,ROOT/'figures/idsw_detection.png',y=2.0,h=4.25)
    text(s,.95,6.4,11.5,.35,'MLP thêm: 25.217 tham số · FPS seed 42 giảm khoảng 5–6% tùy ngân sách',15,MUTED)
    s=add('Demo: cùng video, hai tracker, không GT oracle','07 / Demo','Checkpoint seed 42, train 15%; hai cảnh chọn theo tên và metadata mật độ.',
        notes='Demo được chạy lại với hai checkpoint, chỉ dùng ảnh và lịch sử dự đoán. Các số ID của hai tracker không có nghĩa phải giống nhau, vì đó là nhãn cục bộ theo từng lần chạy. Vệt tâm hộp chỉ giúp nhìn hướng di chuyển, không phải visual-trail input của mô hình. Một đoạn video chỉ là minh họa định tính; kết luận về mức cải thiện vẫn phải dựa trên toàn bộ validation. Em sẽ mở file demo nếu thời gian cho phép, hoặc dùng ảnh này để giải thích pipeline.',seconds=40)
    image(s,ROOT/'figures/demo_comparison.png',y=2.05,h=4.1)
    text(s,.9,6.4,11.5,.35,'Video: demo/DEMO_MOTIP_DanceTrack.mp4 · đường dẫn tương đối trong bộ nộp',15,TEAL,True)
    s=add('Điều gì đã biết, điều gì chưa được chứng minh?','08 / Thảo luận','Giữ kết luận trong phạm vi dữ liệu và protocol thực sự đã chạy.',
        notes='Đã quan sát được geometry cải thiện association trong cả ba ngân sách và ba seed, với detector cố định. Chưa chứng minh nó luôn hơn khi full-data end-to-end hoặc ngoài DanceTrack. Geometry đã có pretraining riêng nên không được coi toàn bộ gain là chỉ do kiến trúc. Validation dùng chọn checkpoint còn nằm trong tập full validation, nên cần một test độc lập cho kết luận mạnh hơn. Detector tiền huấn luyện cũng là nguồn dữ liệu bổ sung; ít dữ liệu ở đây là ít giám sát association, không phải toàn hệ thống chưa nhìn dữ liệu khác.',seconds=60)
    card(s,.65,2.2,5.85,3.95,'Bằng chứng thực nghiệm','9/9 cặp ΔAssA > 0\n9/9 cặp IDSW thấp hơn\nDetector không được cập nhật',TEAL,size=23)
    card(s,6.75,2.2,5.85,3.95,'Giới hạn','Subset ≤ 15%, n = 3 seed\nCó thêm geometry pretraining\nValidation chưa là test độc lập',CORAL,size=23)
    s=add('Kết luận','09 / Kết luận','Tái hiện cơ chế ID prediction và kiểm chứng một cải tiến nhỏ, có đo lường.',
        notes='Tóm lại, phần chính của đồ án là hiểu và tái hiện MOTIP: detector tạo đặc trưng, decoder dùng lịch sử mang ID để dự đoán ID hiện tại, rồi cập nhật bộ nhớ. Nhánh bổ sung cung cấp geometry đa độ trễ, chấm từng cặp ứng viên bằng một MLP nhỏ và hợp nhất logit. Trên các subset và seed đã khảo sát, nó tăng AssA và giảm ID switch. Các giới hạn về pretraining, chọn checkpoint và dữ liệu được trình bày công khai. Bước tiếp theo nên là test độc lập và đối chứng ngân sách compute trước khi mở rộng kết luận.',seconds=40)
    rect(s,.65,2.15,12,3.9,INK,radius=True)
    text(s,1.0,2.65,11.2,1.35,'Liên kết học được\n+ hình học tường minh',38,WHITE,True,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)
    text(s,1.05,4.55,10.9,.95,'Tín hiệu cải thiện nhất quán trên thiết lập được kiểm chứng.\nKhông đồng nhất kết quả validation với ưu thế SOTA phổ quát.',22,WHITE,False,PP_ALIGN.CENTER,MSO_ANCHOR.MIDDLE)
    # Backup slides do not count toward the 20-minute narrative.
    s=add('Phụ lục — G4 có đúng 66 chiều','Hỏi đáp / A1',notes='Nếu được hỏi 66 chiều gồm gì, dùng bảng này thay vì nói một vector geometry chung chung. Độ dời chuẩn hóa theo hộp hiện tại; lag thiếu có mask.',backup=True,seconds=0)
    table(s,['Nhóm','Chiều'],[['Hộp hiện tại / lịch sử / mask','4 + 16 + 4'],['Vận tốc + extrapolation / validity','8 + 4'],['Biến thiên vận tốc / validity','4 + 2'],['Relative offset và log-scale','16'],['Log-area / lag code','4 + 4'],['Tổng','66']],h=4.3,fontsize=20)
    s=add('Phụ lục — IDF1 và IDSW không tương đương','Hỏi đáp / A2',notes='IDF1 dùng ghép danh tính ở mức quỹ đạo; IDSW là sự kiện đổi theo matching CLEAR. Một phương pháp ít theo dõi hoặc tạo quỹ đạo ngắn có thể có số switch khác mà không tốt hơn. Vì vậy không tối ưu một mình IDSW.',backup=True,seconds=0)
    card(s,.65,2.2,5.85,3.9,'IDF1 ↑','Tính nhất quán ID ở mức quỹ đạo\nXem độ phủ của đúng danh tính\nCao hơn thường tốt hơn',BLUE,size=23)
    card(s,6.75,2.2,5.85,3.9,'IDSW ↓','Số sự kiện đổi ID\nPhụ thuộc matching theo thời gian\nKhông đủ để kết luận một mình',TEAL,size=23)
    s=add('Phụ lục — kết quả từng seed ở 15%','Hỏi đáp / A3',notes='Tất cả seed được giữ trong báo cáo. Epoch được chọn theo small validation cố định, khác nhau giữa run là bình thường. Không lấy checkpoint full-validation đẹp nhất làm primary.',backup=True,seconds=0)
    raw=list(csv.DictReader((ROOT/'experiments/metrics_all_seeds.csv').open()))
    table(s,['Seed','Model','Epoch','AssA','IDF1','IDSW'],[[r['seed'],'E0' if r['model']=='e0' else 'V3',r['epoch'],r['AssA'],r['IDF1'],f"{int(float(r['IDSW'])):,}"] for r in raw if r['budget_pct']=='15'],h=4.4,fontsize=18)
    s=add('Phụ lục — đâu là đóng góp của đồ án?','Hỏi đáp / A4',notes='Không nhận phát minh MOTIP hay detector là đóng góp mới. Tái hiện dùng mã tác giả có ghi nhận, nhánh G4 và tích hợp được phân biệt. Bàn giao có source, config, manifest, checksum và demo. Geometry trong tracking không mới nói chung; đây là khảo sát cấu hình cụ thể.',backup=True,seconds=0)
    table(s,['Kế thừa','Tự triển khai / thực nghiệm'],[['MOTIP, Deformable DETR, TrackEval','Nhánh G4 theo cặp + tích hợp và kiểm thử'],['Ý tưởng ID prediction của Gao và cộng sự','Matched E0/V3 trên các subset lồng nhau'],['Checkpoint detector đã tiền huấn luyện','Train-only geometry pretrain + phân tích kết quả'],['Giấy phép và trích dẫn được giữ','Đóng gói độc lập, demo và báo cáo']],h=4.1,fontsize=18)
    s=add('Phụ lục — chạy lại demo','Hỏi đáp / A5',notes='Demo yêu cầu CUDA và extension deformable attention tương thích. File ghi môi trường và README có đường dẫn. Lệnh inference không dùng GT; dữ liệu dùng symlink local để tránh sao chép toàn bộ dataset.',backup=True,seconds=0)
    card(s,.65,2.2,12,3.9,'Ba bước','1. Chuẩn bị môi trường và đường dẫn DanceTrack\n2. python scripts/run_demo_inference.py --model both\n3. python scripts/render_demo.py',TEAL,size=22)
    s=add('Tài liệu tham khảo chính','Hỏi đáp / A6','Nguồn đầy đủ và liên kết trong báo cáo.',notes='Paper gốc CVPR 2025, Deformable DETR ICLR 2021, DanceTrack CVPR 2022, HOTA và TrackEval. Những số liệu thực nghiệm riêng không được gán cho tác giả paper.',backup=True,seconds=0)
    text(s,.8,2.15,11.8,4.5,'Gao, Qi, Wang — Multiple Object Tracking as ID Prediction. CVPR 2025.\nZhu và cộng sự — Deformable DETR. ICLR 2021.\nSun và cộng sự — DanceTrack. CVPR 2022.\nLuiten và cộng sự — HOTA. International Journal of Computer Vision.\nMã nguồn: github.com/MCG-NJU/MOTIP\nĐánh giá: github.com/JonathonLuiten/TrackEval',20,INK)
    total=sum(x['seconds'] for x in SPEECH if not x['backup'])
    scale=1200/total
    for row,slide in zip(SPEECH,P.slides):
        if not row['backup']:
            row['seconds']=round(row['seconds']*scale)
            slide.notes_slide.notes_text_frame.text=f"Thời lượng gợi ý: {row['seconds']} giây\n\n{row['notes']}"
    correction=1200-sum(row['seconds'] for row in SPEECH if not row['backup'])
    last_main=next(i for i in range(len(SPEECH)-1,-1,-1) if not SPEECH[i]['backup'])
    SPEECH[last_main]['seconds']+=correction
    P.slides[last_main].notes_slide.notes_text_frame.text=f"Thời lượng gợi ý: {SPEECH[last_main]['seconds']} giây\n\n{SPEECH[last_main]['notes']}"
    P.save(OUT/'THUYET_TRINH_MOTIP.pptx')
    notes=['# Kịch bản thuyết trình — khoảng 20 phút','',
        '23 slide chính, 6 slide phụ lục. Không cần đọc nguyên văn; tập trung diễn giải hình và logic. Video minh họa có thể chỉ chiếu một đoạn 30–40 giây để không quá giờ.','']
    elapsed=0
    for row in SPEECH:
        if row['backup']:continue
        notes += [f"## Slide {row['number']}. {row['title']}",'',f"Mốc {elapsed//60:02d}:{elapsed%60:02d} · khoảng {row['seconds']} giây",'',row['notes'],'']
        elapsed+=row['seconds']
    (OUT/'KICH_BAN_20_PHUT.md').write_text('\n'.join(notes))
    (OUT/'slide_structure.json').write_text(json.dumps(SPEECH,indent=2,ensure_ascii=False)+'\n')
    print(f'Built {len(P.slides)} slides; main narrative {elapsed/60:.2f} minutes.',flush=True)


if __name__=='__main__':
    # Old layout helpers remain above for provenance. The default authoring
    # path is now HTML/CSS -> browser-measured editable PowerPoint.
    from build_html_presentation import main as html_main
    from export_html_pptx import export
    html_main();export()
