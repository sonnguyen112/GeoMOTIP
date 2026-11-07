#!/usr/bin/env python3
"""Typeset the verified course manuscript using the MIT-licensed arxiv template.

The Markdown remains a readable source. Mathematical expressions below are
explicit mappings, not a heuristic translation of research equations.
"""
from pathlib import Path
import re
from build_report import blocks
from latex_math_sections import PARAGRAPHS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/latex'
MATH = {
    'A = softmax(QKᵀ / √d + M + B_time)': r'A=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt d}+M+B_{\mathrm{time}}\right)',
    'Attention(Q,K,V) = AV': r'\operatorname{Attention}(Q,K,V)=AV',
    'z_t = Decoder(query_t, history_<t, masks)': r'z_t=\operatorname{Decoder}(q_t,\mathcal H_{<t},M)',
    'l_B = IDHead(z_t)': r'\ell_B=\operatorname{IDHead}(z_t)',
    'L_ID = mean[-log softmax(l_B)[y]]': r'\mathcal L_{\mathrm{ID}}=\operatorname{mean}_{\text{hợp lệ}}[-\log\operatorname{softmax}(\ell_B)_y]',
    'd_x(Δ) = (c_x,t − c_x,t−Δ) / max(w_t, ε)': r'd_x^{\Delta}=\frac{c_{x,t}-c_{x,t-\Delta}}{\max(w_t,\varepsilon)}',
    'log((w_t + ε)/(w_t−Δ + ε))': r'\log\frac{w_t+\varepsilon}{w_{t-\Delta}+\varepsilon}',
    '66 → 128 → 128 → 1': r'66\to128\to128\to1',
    'l_final = l_B / T_b + λ · l_G / T_g': r'\ell_{\mathrm{final}}=\frac{\ell_B}{T_b}+\lambda\frac{\ell_G}{T_g}',
    'T_b = 2': r'T_b=2', 'T_g = 0,5': r'T_g=0{,}5', 'λ = 1': r'\lambda=1',
    'L_total = ν · CE(l_final, y) + μ · CE(l_G, y)': r'\mathcal L_{\mathrm{total}}=\nu\operatorname{CE}(\ell_{\mathrm{final}},y)+\mu\operatorname{CE}(\ell_G,y)',
    'μ = ν = 1': r'\mu=\nu=1',
    'HOTA_α = sqrt(DetA_α · AssA_α)': r'\mathrm{HOTA}_{\alpha}=\sqrt{\mathrm{DetA}_{\alpha}\,\mathrm{AssA}_{\alpha}}',
    'ddof = 1': r'\mathrm{ddof}=1',
    '(IDSW_base − IDSW_geom) / IDSW_base': r'\frac{\mathrm{IDSW}_{\mathrm{base}}-\mathrm{IDSW}_{\mathrm{geom}}}{\mathrm{IDSW}_{\mathrm{base}}}',
}
CITES={1:'motip',2:'deformable',3:'dancetrack',4:'hota',5:'motipcode',6:'trackeval'}


def esc(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}',
             '~':r'\textasciitilde{}','^':r'\textasciicircum{}','↑':r'\ensuremath{\uparrow}','↓':r'\ensuremath{\downarrow}',
             '→':r'\ensuremath{\to}','−':r'\ensuremath{-}','Δ':r'\ensuremath{\Delta}',
             'λ':r'\ensuremath{\lambda}','α':r'\ensuremath{\alpha}','ε':r'\ensuremath{\varepsilon}',
             'μ':r'\ensuremath{\mu}','ν':r'\ensuremath{\nu}','±':r'\ensuremath{\pm}'}
    return ''.join(mapping.get(c,c) for c in s)


def inline(s):
    result=[]
    for p in re.split(r'(`[^`]+`|\*\*.*?\*\*|\[\d\])',s):
        if p.startswith('`'):
            content=p[1:-1]
            result.append('$'+MATH[content]+'$' if content in MATH else r'\path{'+content+'}')
        elif p.startswith('**'):result.append(r'\textbf{'+inline(p[2:-2])+'}')
        elif re.fullmatch(r'\[\d\]',p):result.append(r'\cite{'+CITES[int(p[1])]+'}')
        else:result.append(esc(p))
    return ''.join(result)


def main():
    source=(ROOT/'report/BAO_CAO_MOTIP.md').read_text()
    items=list(blocks(source))
    out=[];abstract=False;references=False;appendix=False;table_no=0;figure_no=0
    math_replaced=set()
    for i,(kind,value) in enumerate(items):
        if kind=='heading':
            level,title=value
            if level==1:continue
            if title=='Tóm tắt':abstract=True;out.append(r'\begin{abstract}');continue
            if title=='Tài liệu tham khảo':
                out.append(r'\FloatBarrier\bibliographystyle{unsrt}\bibliography{references}')
                references=True;continue
            if title.startswith('Phụ lục'):
                references=False
                if not appendix:out.append(r'\clearpage\appendix');appendix=True
            title=re.sub(r'^\d+(?:\.\d+)*\.\s*','',title)
            title=re.sub(r'^Phụ lục [AB]\.\s*','',title)
            out.append(r'\FloatBarrier'+'\n'+('\\section{' if level==2 else '\\subsection{')+inline(title)+'}')
        elif references:continue
        elif kind=='paragraph':
            if i<5 and not abstract:continue
            if abstract:
                if value.startswith('**Từ khóa:'):
                    out.append(r'\end{abstract}'+'\n'+inline(value));abstract=False
                else:out.append(inline(value)+'\n')
                continue
            # Captions are attached to their corresponding float below.
            if re.match(r'^(Hình|Bảng) \d+[.]',value):continue
            replacement=next((key for key in PARAGRAPHS if value.startswith(key)),None)
            if replacement:
                assert replacement not in math_replaced
                math_replaced.add(replacement)
                out.append(PARAGRAPHS[replacement]+'\n')
                continue
            value=value.replace('Bảng 6 dùng',r'Bảng 6 dùng')
            out.append(inline(value)+'\n')
        elif kind=='image':
            alt,path=value;figure_no+=1
            # Embed the vector drawing in the paper; PNG is for PowerPoint.
            if path.with_suffix('.pdf').exists():path=path.with_suffix('.pdf')
            cap=alt
            if i+1<len(items) and items[i+1][0]=='paragraph' and items[i+1][1].startswith('Hình '):
                cap=re.sub(r'^Hình \d+\.\s*','',items[i+1][1])
            maxheight='.46' if path.stem.startswith('association_case_') else '.34'
            out.append('\\begin{figure}[!htbp]\n\\centering\n'+
                r'\includegraphics[width=\linewidth,height='+maxheight+r'\textheight,keepaspectratio]{../../figures/'+path.name+'}\n'+
                '\\caption{'+inline(cap)+'}\\label{fig:'+str(figure_no)+'}\n\\end{figure}\n')
            if path.stem=='geometry_pretraining' or path.stem.startswith('association_case_'):
                out.append(r'\FloatBarrier')
        elif kind=='table':
            table_no+=1;n=len(value[0])
            captions={1:'Các nhóm đặc trưng trong biểu diễn G4.',2:'Ngân sách dữ liệu huấn luyện và split đánh giá.',
                3:'Cấu hình đối chiếu MOTIP nền và MOTIP + G4.',4:'Kết quả chính trên toàn bộ validation; trung bình và độ lệch chuẩn mẫu của ba seed.',
                5:'Số ID switch và chênh lệch AssA; mọi seed đều được giữ lại.',6:'Thông lượng và bộ nhớ: các cặp seed 42 trên cùng hệ thống local.',
                7:'Kết quả từng seed; epoch được chọn bằng AssA trên small validation.'}
            spec=('>{\\raggedright\\arraybackslash}p{.64\\linewidth}r' if table_no==1 else
                  '>{\\raggedright\\arraybackslash}p{.29\\linewidth}X' if n==2 else 'l'*n)
            out.append('\\begin{table}[!htbp]\n\\centering\n\\caption{'+captions[table_no]+'}\\label{tab:'+str(table_no)+'}\n')
            if n>=5:out.append(r'\small\setlength{\tabcolsep}{4.2pt}')
            if n==2:out.append(r'\begin{tabularx}{\linewidth}{'+spec+'}')
            else:out.append(r'\begin{tabular}{'+spec+'}')
            out.append(r'\toprule')
            for j,row in enumerate(value):
                def cell(c):
                    if c=='Mã lag: 0, 1/3, 2/3, 1':return r'Mã lag: $0,\frac{1}{3},\frac{2}{3},1$'
                    if c=='T_b=2; T_g=0,5; λ=1 cố định':return r'$T_b=2$; $T_g=0{,}5$; $\lambda=1$ cố định'
                    if c=='66 → 128 → 128 → 1; 25.217 tham số':return r'$66\to128\to128\to1$; 25.217 tham số'
                    return inline(c)
                out.append(' & '.join((r'\textbf{'+cell(c)+'}') if j==0 else cell(c) for c in row)+r' \\')
                if j==0:out.append(r'\midrule')
                elif table_no==4 and j in [2,4]:out.append(r'\addlinespace[4pt]')
            out.extend([r'\bottomrule',r'\end{tabularx}' if n==2 else r'\end{tabular}',r'\end{table}'+'\n'])
    assert len(math_replaced)==len(PARAGRAPHS),(set(PARAGRAPHS)-math_replaced)
    (OUT/'content.tex').write_text('\n'.join(out)+'\n')
    print(f'Generated LaTeX: {table_no} tables, {figure_no} figures, {len(MATH)} explicit math mappings.')


if __name__=='__main__':main()
