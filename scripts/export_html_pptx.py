#!/usr/bin/env python3
"""Export the browser-measured HTML deck without flattening whole slides."""
from pathlib import Path
import json
import math
import re
from urllib.parse import unquote,urlparse
from PIL import Image
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE,XL_LEGEND_POSITION,XL_MARKER_STYLE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'presentation'
HTML=OUT/'html'
PX=120 # 1600 CSS px = 13 1/3 PowerPoint inches.


def color(css):
    if css.startswith('#'):return RGBColor.from_string(css[1:])
    nums=[int(float(v)) for v in re.findall(r'[\d.]+',css)[:3]]
    return RGBColor(*nums)


def add_errorbars(ser,std):
    node=OxmlElement('c:errBars')
    for name,value in [('errDir','y'),('errBarType','both'),('errValType','cust'),('noEndCap','0')]:
        ch=OxmlElement('c:'+name);ch.set('val',value);node.append(ch)
    for tag in ['plus','minus']:
        outer=OxmlElement('c:'+tag);lit=OxmlElement('c:numLit')
        count=OxmlElement('c:ptCount');count.set('val',str(len(std)));lit.append(count)
        for i,s in enumerate(std):
            p=OxmlElement('c:pt');p.set('idx',str(i));v=OxmlElement('c:v');v.text=str(s);p.append(v);lit.append(p)
        outer.append(lit);node.append(outer)
    ser.insert(list(ser).index(ser.find(qn('c:cat'))),node)


def add_chart(slide,r):
    spec=r['chart'];data=CategoryChartData();data.categories=spec['categories']
    for name,vs in zip(spec['names'],spec['values']):data.add_series(name,vs)
    kind=XL_CHART_TYPE.COLUMN_CLUSTERED if spec.get('kind')=='bar' else XL_CHART_TYPE.LINE_MARKERS
    chart=slide.shapes.add_chart(kind,Inches(r['x']/PX),Inches(r['y']/PX),Inches(r['w']/PX),Inches(r['h']/PX),data).chart
    chart.chart_style=10;chart.has_title=True
    chart.chart_title.text_frame.text=spec['title']
    for p in chart.chart_title.text_frame.paragraphs:p.font.name='Aptos';p.font.size=Pt(22);p.font.bold=True
    chart.has_legend=True;chart.legend.position=XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout=False;chart.legend.font.name='Aptos';chart.legend.font.size=Pt(15)
    for axis in [chart.category_axis,chart.value_axis]:axis.tick_labels.font.name='Aptos';axis.tick_labels.font.size=Pt(16)
    chart.value_axis.has_major_gridlines=True
    if spec.get('kind')!='bar':
        errors=spec.get('std',[[0]*len(spec['categories'])]*len(spec['values']))
        chart.value_axis.minimum_scale=spec.get('axis_min',math.floor(min(v-e for vs,es in zip(spec['values'],errors) for v,e in zip(vs,es))-2))
        chart.value_axis.maximum_scale=math.ceil(max(v+e for vs,es in zip(spec['values'],errors) for v,e in zip(vs,es))+2)
    else:chart.value_axis.minimum_scale=0
    sp=OxmlElement('c:spPr');no=OxmlElement('a:noFill');sp.append(no)
    ln=OxmlElement('a:ln');ln.append(OxmlElement('a:noFill'));sp.append(ln)
    existing=chart._chartSpace.find(qn('c:spPr'))
    if existing is not None:chart._chartSpace.remove(existing)
    chart._chartSpace.append(sp)
    chart.value_axis.major_gridlines.format.line.color.rgb=RGBColor.from_string('E2E8EE')
    for i,(series,c) in enumerate(zip(chart.series,spec['colors'])):
        series.format.fill.solid();series.format.fill.fore_color.rgb=color(c)
        series.format.line.color.rgb=color(c);series.format.line.width=Pt(2)
        if kind==XL_CHART_TYPE.LINE_MARKERS:
            series.marker.style=XL_MARKER_STYLE.CIRCLE if i==0 else XL_MARKER_STYLE.SQUARE
            series.marker.size=7;series.marker.format.fill.solid();series.marker.format.fill.fore_color.rgb=color(c)
            series.marker.format.line.color.rgb=color(c)
            if 'std' in spec:add_errorbars(series._element,spec['std'][i])
    return chart


def export():
    (HTML/'renders').mkdir(exist_ok=True)
    (HTML/'assets').mkdir(exist_ok=True)
    pres=Presentation();pres.slide_width=Inches(1600/PX);pres.slide_height=Inches(900/PX)
    pres.core_properties.author='Nguyễn Hồ Trường Sơn — 25C01020'
    pres.core_properties.title='MOTIP và hình học quỹ đạo đa độ trễ'
    pres.core_properties.subject='Đồ án Thị giác máy tính — TS. Trương Phước Hưng'
    notes=json.loads((OUT/'slide_structure.json').read_text())
    audit=dict(slides=[],overflow=[],native_texts=0,native_charts=0,native_tables=0,full_slide_rasterization=False)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport=dict(width=1600,height=900),device_scale_factor=2)
        page.goto((HTML/'index.html').as_uri());page.wait_for_function('Array.from(document.images).every(i=>i.complete)')
        page.evaluate("document.getElementById('nav').style.display='none'")
        for idx in range(len(notes)):
            page.evaluate('(i)=>showSlide(i)',idx);page.evaluate('()=>document.fonts.ready')
            section=page.locator('.slide.active')
            records=section.evaluate('''s=>[...s.querySelectorAll('[data-kind]')].map((e,i)=>{const b=e.getBoundingClientRect(),o=s.getBoundingClientRect(),c=getComputedStyle(e);return {index:i,kind:e.dataset.kind,x:b.x-o.x,y:b.y-o.y,w:b.width,h:b.height,text:e.innerText||'',font:parseFloat(c.fontSize),lineHeight:parseFloat(c.lineHeight),bold:parseInt(c.fontWeight)>=600,align:c.textAlign,middle:c.alignItems==='center',color:c.color,bg:c.backgroundColor,radius:parseFloat(c.borderRadius),src:e.src||'',table:e.dataset.table?JSON.parse(e.dataset.table):null,chart:e.dataset.chart?JSON.parse(e.dataset.chart):null,overflow:e.scrollHeight>e.clientHeight+3||e.scrollWidth>e.clientWidth+3}})''')
            slide=pres.slides.add_slide(pres.slide_layouts[6])
            slide.background.fill.solid();slide.background.fill.fore_color.rgb=color('#18324A' if 'dark' in section.get_attribute('class') else '#FFFFFF')
            shapes=section.locator('[data-kind]')
            for r in records:
                if r['overflow'] and r['kind']=='text':audit['overflow'].append(dict(slide=idx+1,text=r['text'],box=[r['x'],r['y'],r['w'],r['h']]))
                x,y,w,h=[Inches(r[k]/PX) for k in ['x','y','w','h']]
                if r['kind']=='text':
                    sh=slide.shapes.add_textbox(x,y,w,h);tf=sh.text_frame
                    tf.word_wrap=True;tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                    tf.vertical_anchor=MSO_ANCHOR.MIDDLE if r['middle'] else MSO_ANCHOR.TOP
                    for j,line in enumerate(r['text'].split('\n')):
                        p=tf.paragraphs[0] if j==0 else tf.add_paragraph();p.text=line
                        p.alignment={'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}.get(r['align'],PP_ALIGN.LEFT)
                        p.font.name='Aptos';p.font.size=Pt(r['font']*.6);p.font.bold=r['bold'];p.font.color.rgb=color(r['color'])
                        p.line_spacing=r['lineHeight']/r['font'];p.space_after=Pt(0);p.space_before=Pt(0)
                    audit['native_texts']+=1
                elif r['kind']=='image':
                    p=Path(unquote(urlparse(r['src']).path));iw,ih=Image.open(p).size
                    scale=min(r['w']/iw,r['h']/ih);ww,hh=iw*scale,ih*scale
                    slide.shapes.add_picture(str(p),Inches((r['x']+(r['w']-ww)/2)/PX),Inches((r['y']+(r['h']-hh)/2)/PX),width=Inches(ww/PX),height=Inches(hh/PX))
                elif r['kind']=='graphic':
                    p=HTML/f'assets/slide_{idx+1:02}_graphic_{r["index"]:02}.png'
                    # Capture only this diagram, not overlapping sibling labels
                    # that are exported separately as native PowerPoint text.
                    shapes.evaluate_all('(els,index)=>els.forEach((e,i)=>{e.dataset.exportVisibility=e.style.visibility;if(i!==index)e.style.visibility="hidden"})',r['index'])
                    try:
                        shapes.nth(r['index']).screenshot(path=str(p),omit_background=True)
                    finally:
                        shapes.evaluate_all('(els)=>els.forEach(e=>{e.style.visibility=e.dataset.exportVisibility;delete e.dataset.exportVisibility})')
                    slide.shapes.add_picture(str(p),x,y,width=w,height=h)
                elif r['kind']=='chart':add_chart(slide,r);audit['native_charts']+=1
                elif r['kind']=='table':
                    d=r['table'];values=[d['headers']]+d['rows']
                    tab=slide.shapes.add_table(len(values),len(values[0]),x,y,w,h).table
                    for ri,row in enumerate(values):
                        for ci,c in enumerate(row):
                            cell=tab.cell(ri,ci);cell.text=str(c);cell.vertical_anchor=MSO_ANCHOR.MIDDLE
                            cell.margin_left=cell.margin_right=Inches(.08);cell.margin_top=cell.margin_bottom=0
                            cell.fill.solid();cell.fill.fore_color.rgb=color('#18324A' if ri==0 else '#EAF1F6' if ri%2==0 else '#F7F9FB')
                            for p in cell.text_frame.paragraphs:
                                p.alignment=PP_ALIGN.CENTER;p.font.name='Aptos';p.font.size=Pt(r['font']*.6);p.font.bold=ri==0
                                p.font.color.rgb=color('#FFFFFF' if ri==0 else '#18324A');p.space_after=Pt(0)
                    audit['native_tables']+=1
                else:
                    sh=slide.shapes.add_shape(MSO_SHAPE.OVAL if r['kind']=='circle' else MSO_SHAPE.ROUNDED_RECTANGLE if r['radius'] else MSO_SHAPE.RECTANGLE,x,y,w,h)
                    sh.fill.solid();sh.fill.fore_color.rgb=color(r['bg']);sh.line.fill.background()
            slide.notes_slide.notes_text_frame.text=f'Thời lượng gợi ý: {notes[idx]["seconds"]} giây\n\n{notes[idx]["notes"]}'
            section.screenshot(path=str(HTML/f'renders/slide_{idx+1:02}.png'))
            audit['slides'].append(dict(slide=idx+1,objects=len(records),images=sum(r['kind'] in ['image','graphic'] for r in records)))
        # Browser-rendered PDF is also retained, but the main PDF is exported
        # from the final PPTX with Impress to verify the delivered document.
        page.emulate_media(media='print')
        page.pdf(path=str(HTML/'BROWSER_PREVIEW.pdf'),width='1600px',height='900px',print_background=True,prefer_css_page_size=True)
        browser.close()
    pres.save(OUT/'THUYET_TRINH_MOTIP.pptx')
    (HTML/'EXPORT_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='slides'},indent=2,ensure_ascii=False))
    assert not audit['overflow'],'Fix HTML text overflow before delivery.'


if __name__=='__main__':export()
