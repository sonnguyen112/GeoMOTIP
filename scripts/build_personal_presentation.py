#!/usr/bin/env python3
"""Independent browser presentation: no PPTX conversion, no PDF export.

Shares verified content with the submission deck, but keeps browser-native
SVG/CSS, fullscreen, overview, image zoom, notes and video playback. The HTML
bundle is included in submission and is also available as a portable ZIP.
"""
from pathlib import Path
import html
import json
import re
import shutil
import zipfile
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'personal_presentation'


CSS=r'''
@font-face{font-family:Lato;src:url('assets/Lato-Regular.ttf');font-weight:400;font-display:block}
@font-face{font-family:Lato;src:url('assets/Lato-Bold.ttf');font-weight:700;font-display:block}
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#101d2e;color:#18324A;font-family:Lato,sans-serif}
button,input{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible{outline:3px solid #47d3c0;outline-offset:4px}
.slide{position:absolute;left:50%;top:50%;width:1600px;height:900px;transform-origin:center;display:none;overflow:hidden;background:#fff}
.slide.active{display:block;animation:enter .20s ease-out}.slide>*{position:absolute}.slide.dark{background:#18324A}.slide svg{font-family:Lato,sans-serif}
.text{white-space:pre-wrap;line-height:1.18}.table table{border-collapse:collapse;table-layout:fixed;width:100%;height:100%}
.table th,.table td{text-align:center;vertical-align:middle;padding:12px 17px;line-height:1.18}.table th{background:#18324A;color:white;font-weight:700}.table tbody tr:nth-child(odd){background:#f4f7fa}.table tbody tr:nth-child(even){background:#eaf1f6}
.hero-bg{inset:0;width:1600px;height:900px;object-fit:cover;filter:brightness(.77)}
.hero-shade{inset:0;background:linear-gradient(90deg,rgba(10,24,40,.98) 0%,rgba(10,24,40,.86) 39%,rgba(10,24,40,.24) 80%)}
.hero-title{left:85px;top:190px;margin:0;width:1110px;font-size:86px;line-height:1.04;letter-spacing:-1.5px;color:white;font-weight:700}
.hero-kicker{left:90px;top:103px;font-size:28px;color:#b9ddd9}.hero-sub{left:90px;top:540px;width:920px;font-size:34px;line-height:1.4;color:#dbe9f1}
.hero-author{left:90px;top:738px;font-size:29px;line-height:1.55;color:#fff}.hero-meta{right:80px;bottom:52px;font-size:23px;color:#cedce8;text-align:right}
.ending-title{left:90px;top:100px;margin:0;font-size:74px;color:white;line-height:1.08;width:1300px}.ending-body{left:95px;top:365px;width:1060px;font-size:40px;line-height:1.55;color:#eff7fa}.ending-note{left:95px;top:720px;font-size:27px;color:#b8ddd7}
.controls{position:fixed;right:20px;bottom:14px;z-index:30;display:flex;align-items:center;gap:6px;padding:7px;border-radius:14px;background:rgba(12,27,44,.94);box-shadow:0 4px 25px #0003;opacity:0;transition:opacity .18s;color:#e8f0f7}
body.controls-visible .controls,.controls:hover,.controls:focus-within{opacity:1}.controls button{border:0;border-radius:8px;background:transparent;color:inherit;padding:8px 11px;font-size:15px}.controls button:hover{background:#284354}.counter{padding:0 10px;min-width:85px;font-size:15px}.timer{font-variant-numeric:tabular-nums;font-size:14px;color:#a1d8cf;padding:0 6px}
.overlay{position:fixed;inset:0;z-index:50;display:none;background:rgba(8,20,34,.97);color:white}.overlay.open{display:flex}.close{position:absolute;top:20px;right:25px;z-index:60;border:1px solid #688092;background:#18324A;color:white;border-radius:10px;font-size:20px;padding:9px 16px}
#overview{padding:60px 45px 30px;overflow:auto;flex-direction:column}.overview-head{display:flex;align-items:center;gap:25px;max-width:1200px;margin-bottom:25px}.overview-head h2{font-size:30px;margin:0}.overview-head input{border:1px solid #7895a6;background:#173044;color:white;font-size:18px;padding:10px 15px;border-radius:8px;width:330px}
.overview-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:22px}.thumb{border:1px solid #3b5469;background:#18324A;border-radius:8px;padding:0;overflow:hidden;text-align:left;color:white}.thumb:hover{border-color:#54cabb}.thumb-picture{position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;background:white;pointer-events:none}.thumb-picture .slide{left:0;top:0;display:block!important;transform-origin:top left;animation:none;pointer-events:none}.thumb-label{padding:12px;font-size:16px;line-height:1.35;min-height:64px}
#notes{left:auto;width:min(640px,90vw);padding:80px 35px 35px;overflow:auto;flex-direction:column;background:rgba(12,29,45,.98);box-shadow:-10px 0 35px #0004}#notes h2{font-size:28px;margin:0 0 24px}#notes p{font-size:21px;line-height:1.65;white-space:pre-wrap}
#zoom{align-items:center;justify-content:center;padding:55px}#zoom img{position:static;max-width:96vw;max-height:89vh;object-fit:contain;background:white}
#video{align-items:center;justify-content:center;padding:45px}#video video{width:min(1500px,96vw);max-height:90vh;background:black}
#help{align-items:center;justify-content:center}.help-content{max-width:700px;padding:40px;font-size:23px;line-height:1.7}.help-content h2{font-size:36px}.help-content kbd{padding:3px 9px;background:#28495e;border-radius:5px;font:inherit}
@keyframes enter{from{opacity:.55}to{opacity:1}}@media(prefers-reduced-motion:reduce){*,.slide.active{animation:none!important;transition:none!important}}
'''


JS=r'''
const slides=[...document.querySelectorAll('main>.slide')];
const meta=JSON.parse(document.getElementById('speaker-data').textContent);const mainCount=meta.filter(x=>!x.backup).length;
let current=0,hideHandle,elapsed=0,running=false;
const $=id=>document.getElementById(id);
function fit(){const scale=Math.min(innerWidth/1600,innerHeight/900);slides.forEach(s=>s.style.transform=`translate(-50%,-50%) scale(${scale})`)}
function notes(){const row=meta[current];$('notes-title').textContent=`${current+1}. ${row.title}`;$('notes-text').textContent=row.notes}
function controls(){document.body.classList.add('controls-visible');clearTimeout(hideHandle);hideHandle=setTimeout(()=>document.body.classList.remove('controls-visible'),1800)}
function closeOverlays(){document.querySelectorAll('.overlay').forEach(x=>x.classList.remove('open'));$('demo-video').pause();if(document.activeElement?.blur)document.activeElement.blur()}
window.showSlide=function(i){current=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,j)=>s.classList.toggle('active',j===current));$('counter').textContent=current<mainCount?`${current+1} / ${mainCount}`:`Phụ lục ${current-mainCount+1}`;history.replaceState(null,'',`#slide-${current+1}`);document.title=`${current+1}. ${meta[current].title} — MOTIP`;notes();controls()};
function next(){closeOverlays();showSlide(current+1)}function prev(){closeOverlays();showSlide(current-1)}
async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen()}catch(e){$('help').classList.add('open')}fit()}
function toggle(id){const was=$(id).classList.contains('open');closeOverlays();if(!was)$(id).classList.add('open');if(id==='notes')notes();if(id==='overview'){resizeThumbs();$('search').focus()}}
function zoomImage(target){let imgs=[...slides[current].querySelectorAll('img')];let chosen=target||imgs.sort((a,b)=>b.getBoundingClientRect().width*b.getBoundingClientRect().height-a.getBoundingClientRect().width*a.getBoundingClientRect().height)[0];if(chosen){$('zoom-image').src=chosen.src;toggle('zoom')}}
function video(){toggle('video');if($('video').classList.contains('open'))$('demo-video').play().catch(()=>{})}
function resizeThumbs(){document.querySelectorAll('.thumb-picture').forEach(x=>{x.firstElementChild.style.transform=`scale(${x.clientWidth/1600})`})}
function overview(){const grid=$('overview-grid');slides.forEach((s,i)=>{const b=document.createElement('button');b.className='thumb';b.dataset.title=meta[i].title.toLocaleLowerCase('vi');b.innerHTML='<div class="thumb-picture"></div><div class="thumb-label"></div>';const c=s.cloneNode(true);c.removeAttribute('id');c.removeAttribute('data-notes');b.firstElementChild.append(c);b.lastElementChild.textContent=`${i+1}. ${meta[i].title}${meta[i].backup?' (phụ lục)':''}`;b.onclick=()=>{closeOverlays();showSlide(i)};grid.append(b)})}
$('search').oninput=e=>{const q=e.target.value.toLocaleLowerCase('vi');document.querySelectorAll('.thumb').forEach(b=>b.hidden=!b.dataset.title.includes(q))};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'){if(e.key==='Escape')closeOverlays();return}if(e.key==='Escape'){closeOverlays();return}const key=e.key.toLowerCase();if(key==='f'){e.preventDefault();fullscreen();return}if(key==='o'){e.preventDefault();toggle('overview');return}if(key==='n'){toggle('notes');return}if(key==='h'||key==='?'){toggle('help');return}if(key==='v'){video();return}if(key==='z'){zoomImage();return}if(key==='t'){running=!running;controls();return}if(document.querySelector('.overlay.open'))return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();next()}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();prev()}if(e.key==='Home')showSlide(0);if(e.key==='End')showSlide(mainCount-1)});
slides.forEach(s=>s.querySelectorAll('img').forEach(im=>{im.style.cursor='zoom-in';im.addEventListener('click',()=>zoomImage(im))}));
let touchX=null;document.addEventListener('touchstart',e=>{touchX=e.touches[0].clientX},{passive:true});document.addEventListener('touchend',e=>{if(touchX===null||document.querySelector('.overlay.open'))return;let dx=e.changedTouches[0].clientX-touchX;if(Math.abs(dx)>70){if(dx<0)next();else prev()}touchX=null},{passive:true});
addEventListener('resize',()=>{fit();resizeThumbs()});addEventListener('fullscreenchange',fit);addEventListener('mousemove',controls);setInterval(()=>{if(running){elapsed++;$('timer').textContent=`${String(Math.floor(elapsed/60)).padStart(2,'0')}:${String(elapsed%60).padStart(2,'0')}`}},1000);
overview();fit();const match=location.hash.match(/slide-(\d+)/);showSlide(match?Number(match[1])-1:0);
'''


def main():
    OUT.mkdir(exist_ok=True);assets=OUT/'assets';assets.mkdir(exist_ok=True)
    for name in ['Lato-Regular.ttf','Lato-Bold.ttf']:
        shutil.copy2(Path('/usr/share/fonts/truetype/lato')/name,assets/name)
    shutil.copy2('/usr/share/doc/fonts-lato/copyright',assets/'FONT_LICENSE.txt')
    source=ROOT/'presentation/html/index.html';page=source.read_text()
    slides=re.findall(r'<section class="slide .*?</section>',page,re.S)
    notes=json.loads((ROOT/'presentation/slide_structure.json').read_text())
    assert len(slides)==len(notes)==31
    built={}
    def image(match):
        rel=html.unescape(match.group(1));path=(source.parent/rel).resolve()
        if path not in built:
            if path.parent.name=='math':pdf=ROOT/f'report/latex/math_assets/build/{path.stem}.pdf'
            else:pdf=path.with_suffix('.pdf')
            stem=('math_' if path.parent.name=='math' else '')+path.stem
            if pdf.exists():
                dest=assets/(stem+'.svg')
                doc=pymupdf.open(pdf);dest.write_text(doc[0].get_svg_image(text_as_path=True))
            else:
                dest=assets/path.name;shutil.copy2(path,dest)
            built[path]=dest.relative_to(OUT).as_posix()
        return f'src="{built[path]}"'
    slides=[re.sub(r'src="([^"]+)"',image,s) for s in slides]
    photo='assets/dataset_sample.jpg'
    if not (OUT/photo).exists():shutil.copy2(ROOT/'figures/dataset_sample.jpg',OUT/photo)
    slides[0]=f'''<section class="slide dark" id="slide-1">
      <img class="hero-bg" src="{photo}" alt="Khung hình DanceTrack thực tế"><div class="hero-shade"></div>
      <div class="hero-kicker">Multiple Object Tracking as ID Prediction · CVPR 2025</div>
      <h1 class="hero-title">MOTIP<br>Duy trì danh tính<br>qua từng khung hình</h1>
      <div class="hero-sub">Tái hiện phương pháp và bổ sung<br>hình học quỹ đạo đa độ trễ G4</div>
      <div class="hero-author">Nguyễn Hồ Trường Sơn · 25C01020<br>Giảng viên: TS. Trương Phước Hưng</div>
      <div class="hero-meta">Đồ án Thị giác máy tính<br>20 phút trình bày</div></section>'''
    end=next(i for i,n in enumerate(notes) if n['title'].startswith('Liên kết học được'))
    slides[end]=f'''<section class="slide dark" id="slide-{end+1}"><img class="hero-bg" src="{photo}" alt="DanceTrack"><div class="hero-shade"></div>
      <h1 class="ending-title">Liên kết học được<br>và hình học tường minh</h1>
      <div class="ending-body">MOTIP dự đoán ID từ lịch sử.<br>G4 chấm tương thích theo từng cặp.<br>9/9 cặp seed/ngân sách tăng AssA.</div>
      <div class="ending-note">Kết luận trong phạm vi dữ liệu và giao thức đã khảo sát.</div>
      <div class="hero-meta">Cảm ơn thầy<br>Phần hỏi đáp: 10 phút</div></section>'''
    # Native typography can be larger without conversion to Office text boxes.
    tuning=r'''<script>document.querySelectorAll('main>.slide').forEach(s=>{const title=s.querySelector(':scope>.text');if(!title)return;const length=title.textContent.length;title.style.fontSize=(length>66?54:length>46?58:64)+'px';title.style.lineHeight='1.03';title.style.letterSpacing='-.6px';const sub=title.nextElementSibling;if(sub?.classList.contains('text')){sub.style.fontSize='26px';sub.style.height='60px';sub.style.lineHeight='1.15'}})</script>'''
    controls='''<nav class="controls" aria-label="Điều khiển trình chiếu"><button onclick="prev()" aria-label="Trang trước">←</button><span class="counter" id="counter"></span><button onclick="next()" aria-label="Trang sau">→</button><button onclick="fullscreen()">Toàn màn hình · F</button><button onclick="toggle('overview')">Tổng quan · O</button><button onclick="toggle('notes')">Ghi chú · N</button><button onclick="video()">Demo · V</button><button onclick="toggle('help')">?</button><span class="timer" id="timer" title="T để bắt đầu/tạm dừng">00:00</span></nav>'''
    overlays='''<div class="overlay" id="overview"><button class="close" onclick="closeOverlays()">Đóng · Esc</button><div class="overview-head"><h2>Tổng quan</h2><input id="search" placeholder="Tìm tên slide" aria-label="Tìm slide"></div><div class="overview-grid" id="overview-grid"></div></div>
    <aside class="overlay" id="notes"><button class="close" onclick="closeOverlays()">Đóng · Esc</button><h2 id="notes-title"></h2><p id="notes-text"></p></aside>
    <div class="overlay" id="zoom"><button class="close" onclick="closeOverlays()">Đóng · Esc</button><img id="zoom-image" alt="Hình phóng to"></div>
    <div class="overlay" id="video"><button class="close" onclick="closeOverlays()">Đóng · Esc</button><video id="demo-video" src="assets/demo.mp4" controls preload="metadata" playsinline></video></div>
    <div class="overlay" id="help"><button class="close" onclick="closeOverlays()">Đóng · Esc</button><div class="help-content"><h2>Điều khiển trình chiếu</h2><p><kbd>F</kbd> toàn màn hình (hoặc F11)<br><kbd>←</kbd> / <kbd>→</kbd> / <kbd>Space</kbd> đổi trang<br><kbd>O</kbd> tổng quan · <kbd>N</kbd> ghi chú<br><kbd>Z</kbd> hoặc bấm vào hình để phóng to<br><kbd>V</kbd> demo video 45 giây<br><kbd>T</kbd> bắt đầu/tạm dừng đồng hồ<br><kbd>Home</kbd> mở đầu · <kbd>End</kbd> kết luận<br><kbd>Esc</kbd> đóng bảng/hình/video</p><p>25 slide chính + 6 phụ lục. Chỉ đổi trang khi bạn thao tác; không tự chạy.</p></div></div>'''
    speaker=json.dumps(notes,ensure_ascii=False).replace('<','\\u003c')
    result='<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MOTIP — Bản trình chiếu cá nhân</title><style>'+CSS+'</style></head><body><main>'+''.join(slides)+'</main>'+controls+overlays+'<script type="application/json" id="speaker-data">'+speaker+'</script>'+tuning+'<script>'+JS+'</script></body></html>'
    (OUT/'index.html').write_text(result)
    shutil.copy2(ROOT/'demo/DEMO_MOTIP_DanceTrack.mp4',assets/'demo.mp4')
    (OUT/'README.md').write_text('''# Trình chiếu HTML — đi kèm gói bài nộp

Mở `index.html` bằng Chrome hoặc Edge. Không cần server, Internet hay cài thêm.
Giữ thư mục `assets` cạnh `index.html` khi di chuyển sang máy khác.

- F hoặc nút Toàn màn hình; nếu trình duyệt chặn, dùng F11.
- Mũi tên, PageUp/PageDown, Space để đổi trang.
- O: tổng quan; N: lời trình bày; Z hoặc bấm hình: phóng to.
- V: video demo tổng quát 45 giây; Esc: đóng video/hình/bảng.
- T: bắt đầu hoặc tạm dừng đồng hồ. Home: mở đầu; End: kết luận.

25 slide chính + 6 phụ lục. Kịch bản gồm 19 phút 15 giây nói và 45 giây video.
Video minh họa quy trình, không phải hai cửa sổ thời gian được kiểm chứng GT.
Hai chuỗi GT là ví dụ tích cực chọn có chủ đích; thống kê tổng hợp không đổi.

Bản HTML dùng SVG/công thức vector, không qua PPTX; không tạo PDF.
''')
    archive=OUT/'THUYET_TRINH_MOTIP_HTML.zip'
    files=[OUT/'index.html',OUT/'README.md']+sorted(assets.iterdir())
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in files:z.write(f,str(Path('MOTIP_HTML')/f.relative_to(OUT)))
    print('Personal offline HTML deck:',OUT/'index.html')
    print('Slides:',len(slides),'Portable ZIP:',archive)


if __name__=='__main__':main()
