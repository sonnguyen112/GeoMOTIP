#!/usr/bin/env python3
"""Build the editable DOCX. The primary PDF is compiled by build_latex.py.

The old ReportLab renderer remains available for provenance only and is no
longer called by default, so it cannot overwrite the XeLaTeX report.
"""
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
from docx import Document
from docx.shared import Cm, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'report'
INK=colors.HexColor('#15304C');TEAL=colors.HexColor('#008D91')


def blocks(text):
    lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in row):rows.append(row)
                i+=1
            yield 'table',rows;continue
        if line.startswith('#'):
            match=re.match(r'^(#+) (.+)',line)
            yield 'heading',(len(match[1]),match[2]);i+=1;continue
        if line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            yield 'image',(match[1],(REPORT/match[2]).resolve());i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
            paragraph.append(lines[i].strip());i+=1
        yield 'paragraph',' '.join(paragraph)


def xml(text):
    text=escape(text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`(.*?)`',r'<font name="DMono" size="9.2">\1</font>',text)
    return text


def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#DBE4EB'));canvas.line(56,37,A4[0]-56,37)
    canvas.setFont('DSans',8);canvas.setFillColor(INK)
    canvas.drawString(56,23,'Nguyễn Hồ Trường Sơn · 25C01020 · Thị giác máy tính')
    canvas.drawRightString(A4[0]-56,23,f'{doc.page}')


def render_pdf(items):
    fontdir=Path('/usr/share/fonts/truetype/dejavu')
    for name,file in [('DSerif','DejaVuSerif.ttf'),('DSerifB','DejaVuSerif-Bold.ttf'),('DSerifI','DejaVuSerif-Italic.ttf'),('DSans','DejaVuSans.ttf'),('DSansB','DejaVuSans-Bold.ttf'),('DMono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
    pdfmetrics.registerFontFamily('DSerif',normal='DSerif',bold='DSerifB',italic='DSerifI',boldItalic='DSerifB')
    pdfmetrics.registerFontFamily('DSans',normal='DSans',bold='DSansB',italic='DSans',boldItalic='DSansB')
    body=ParagraphStyle('Body',fontName='DSerif',fontSize=10.5,leading=16.1,alignment=TA_JUSTIFY,spaceAfter=8)
    styles={1:ParagraphStyle('Title',fontName='DSerifB',fontSize=19,leading=26,alignment=TA_CENTER,spaceAfter=14),
            2:ParagraphStyle('Section',fontName='DSansB',fontSize=13.2,leading=18,spaceBefore=15,spaceAfter=8,keepWithNext=True,textColor=INK),
            3:ParagraphStyle('Subsection',fontName='DSansB',fontSize=11,leading=16,spaceBefore=10,spaceAfter=6,keepWithNext=True,textColor=INK)}
    cell=ParagraphStyle('Cell',fontName='DSans',fontSize=8.2,leading=11.6,spaceAfter=0)
    caption=ParagraphStyle('Caption',parent=body,fontName='DSerifI',fontSize=9,leading=13,alignment=TA_LEFT,textColor=INK)
    story=[]
    width=A4[0]-112
    pending_heading=None
    for index,(kind,value) in enumerate(items):
        if kind=='heading':
            heading=Paragraph(xml(value[1]),styles[min(value[0],3)])
            if index+1<len(items) and items[index+1][0]=='table':pending_heading=heading
            else:story.append(heading)
        elif kind=='paragraph':story.append(Paragraph(xml(value),caption if value.startswith(('Hình ','Bảng ')) else body))
        elif kind=='image':
            _,path=value
            assert path.exists(),path
            w,h=PILImage.open(path).size
            story.extend([Spacer(1,6),Image(str(path),width=width,height=width*h/w),Spacer(1,6)])
        elif kind=='table':
            n=len(value[0])
            if n==2:widths=[width*.30,width*.70]
            elif n==6:widths=[width*.075,width*.125]+[width*.2]*4
            elif n==8:widths=[width*.07,width*.08,width*.10,width*.11]+[width*.16]*4
            else:widths=[width/n]*n
            rows=[[Paragraph(xml(x),cell) for x in row] for row in value]
            t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9F0F5')),
                ('LINEBELOW',(0,0),(-1,0),1,TEAL),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#B5C8D4')),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F9FB')])]))
            if pending_heading is not None:
                story.append(KeepTogether([pending_heading,Spacer(1,7),t]))
                pending_heading=None
            else:story.extend([Spacer(1,7),t])
            story.append(Spacer(1,10))
    doc=SimpleDocTemplate(str(REPORT/'BAO_CAO_MOTIP.pdf'),pagesize=A4,leftMargin=56,rightMargin=56,topMargin=46,bottomMargin=54,
        title='MOTIP và hình học quỹ đạo đa độ trễ — đánh giá thực nghiệm trên DanceTrack',author='Nguyễn Hồ Trường Sơn')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def plain(text):return text.replace('**','').replace('`','')


def render_docx(items):
    doc=Document();section=doc.sections[0]
    section.page_width=Cm(21);section.page_height=Cm(29.7)
    section.top_margin=Cm(2);section.bottom_margin=Cm(2)
    section.left_margin=Cm(2);section.right_margin=Cm(2)
    normal=doc.styles['Normal'];normal.font.name='DejaVu Serif';normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=1.25;normal.paragraph_format.space_after=Pt(7)
    for kind,value in items:
        if kind=='heading':
            level,title=value
            p=doc.add_heading(plain(title),0 if level==1 else level-1)
            if level==1:p.alignment=1
        elif kind=='image':doc.add_picture(str(value[1]),width=Cm(16.8))
        elif kind=='paragraph':
            p=doc.add_paragraph();p.alignment=3
            for part in re.split(r'(\*\*.*?\*\*)',value):
                run=p.add_run(plain(part));run.bold=part.startswith('**')
            if value.startswith(('Hình ','Bảng ')):
                p.alignment=0
                for run in p.runs:run.italic=True;run.font.size=Pt(9)
        elif kind=='table':
            table=doc.add_table(rows=1,cols=len(value[0]));table.style='Light Shading Accent 1'
            for i,text in enumerate(value[0]):table.rows[0].cells[i].text=plain(text)
            header=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(header)
            for row in value[1:]:
                cells=table.add_row().cells
                for i,text in enumerate(row):
                    cells[i].text=plain(text)
                    for run in cells[i].paragraphs[0].runs:run.bold=text.startswith('**')
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for run in p.runs:run.font.name='DejaVu Sans';run.font.size=Pt(8)
            doc.add_paragraph()
    foot=section.footer.paragraphs[0]
    foot.add_run('Nguyễn Hồ Trường Sơn · 25C01020   |   ')
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
    doc.save(REPORT/'BAO_CAO_MOTIP.docx')


if __name__=='__main__':
    text=(REPORT/'BAO_CAO_MOTIP.md').read_text()
    assert '%TABLE_' not in text
    items=list(blocks(text));render_docx(items)
    print('Editable DOCX built. Use scripts/build_latex.py for the primary PDF.',flush=True)
