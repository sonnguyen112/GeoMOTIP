#!/usr/bin/env python3
"""Render true XeLaTeX equations as transparent 450-DPI assets for PowerPoint."""
from pathlib import Path
import subprocess
import os
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'figures/math'
FORMULAS={
 'detection':r'\mathbf b_{t,n}\in\mathbb R^4\qquad\mathbf f_{t,n}\in\mathbb R^{256}\qquad q_{t,n}\in[0,1]',
 'image':r'I_t',
 'detection_output':r'q_{t,n},\ \mathbf b_{t,n},\ \mathbf f_{t,n}',
 'history':r'\begin{aligned}\mathbf F&\in\mathbb R^{B\times G\times T\times K\times256}\\\mathbf B&\in\mathbb R^{B\times G\times T\times K\times4}\\\mathbf Y,\mathbf M&:\ B\times G\times T\times K\end{aligned}',
 'attention':r'\operatorname{Attention}(\mathbf Q,\mathbf K,\mathbf V)=\operatorname{softmax}\!\left(\frac{\mathbf Q\mathbf K^{\mathsf T}}{\sqrt{d_h}}+\mathbf M+\mathbf B_{\mathrm{time}}\right)\mathbf V',
 'causal':r't_{\mathrm{history}}<t_{\mathrm{query}}',
 'base_loss':r'\mathcal L_{\mathrm{ID}}=-\frac{1}{|\Omega|}\sum_{u\in\Omega}\log\left[\operatorname{softmax}(\boldsymbol\ell_{B,u})\right]_{y_u}',
 'pair_score':r'\ell_{G,nk}',
 'base_temperature':r'\boldsymbol\ell_B/2',
 'fusion':r'\boldsymbol\ell_{\mathrm{final}}=\frac{\boldsymbol\ell_B}{2}+\frac{\boldsymbol\ell_G}{0{,}5}',
 'new_neutral':r'\lambda=1\qquad\ell_{G,n,\mathrm{NEW}}=0',
 'total_loss':r'\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{fused}}+\mathcal L_{\mathrm{geom}}',
 'loss_parts':r'\begin{aligned}\mathcal L_{\mathrm{fused}}&=\operatorname{CE}_{\Omega}(\boldsymbol\ell_{\mathrm{final}},\mathbf y)\\\mathcal L_{\mathrm{geom}}&=\operatorname{CE}_{\Omega_G}(\boldsymbol\ell_G,\mathbf y)\end{aligned}',
 'nested':r'\mathcal D_{5\%}\subset\mathcal D_{10\%}\subset\mathcal D_{15\%}',
}


def main():
    DEST.mkdir(exist_ok=True)
    tex=ROOT/'report/latex/math_assets';tex.mkdir(exist_ok=True)
    preamble=r'''\documentclass[border=3pt]{standalone}
\usepackage{fix-cm,fontspec,amsmath,amssymb,xcolor}
\setmainfont{TeX Gyre Termes}
\definecolor{ink}{HTML}{15304C}
\begin{document}
\color{ink}\fontsize{24}{32}\selectfont
'''
    for name,formula in FORMULAS.items():
        (tex/f'{name}.tex').write_text(preamble+'$\\displaystyle '+formula+'$\n\\end{document}\n')
    script='set -eu\ncd /work/report/latex/math_assets\nmkdir -p build\n'
    for name in FORMULAS:
        script+=f'xelatex -no-shell-escape -interaction=batchmode -halt-on-error -output-directory=build {name}.tex > /dev/null\n'
    result=subprocess.run(['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}',
        '-v',f'{ROOT}:/work','--entrypoint','/bin/sh','motip-course-latex:local','-c',script],
        text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode:print(result.stdout);raise SystemExit(result.returncode)
    for name in FORMULAS:
        log=(tex/f'build/{name}.log').read_text()
        assert 'Missing character:' not in log and 'Overfull' not in log,name
        doc=pymupdf.open(tex/f'build/{name}.pdf')
        assert len(doc)==1
        doc[0].get_pixmap(matrix=pymupdf.Matrix(450/72,450/72),alpha=True).save(DEST/f'{name}.png')
    print(f'Rendered {len(FORMULAS)} XeLaTeX equations at 450 DPI.')


if __name__=='__main__':main()
