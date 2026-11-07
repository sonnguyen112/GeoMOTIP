#!/usr/bin/env python3
"""Compile the paper locally with XeLaTeX, preserving the previous PDF."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import re

ROOT=Path(__file__).resolve().parents[1]
LATEX=ROOT/'report/latex'


def main(rebuild=False):
    subprocess.run([sys.executable,str(ROOT/'scripts/make_latex.py')],check=True)
    if rebuild:
        subprocess.run(['docker','build','-f',str(ROOT/'scripts/latex.Dockerfile'),
                        '-t','motip-course-latex:local',str(ROOT/'scripts')],check=True)
    if shutil.which('xelatex') and shutil.which('latexmk'):
        cmd=['latexmk','-xelatex','-no-shell-escape','-interaction=nonstopmode',
             '-halt-on-error','-outdir=build','main.tex']
        cwd=LATEX
    else:
        cmd=['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}',
             '-v',f'{ROOT}:/work','motip-course-latex:local']
        cwd=ROOT
    result=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (LATEX/'build/compile_transcript.txt').write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-5000:]);raise SystemExit(result.returncode)
    log=(LATEX/'build/main.log').read_text()
    for pattern in [r'^!',r'undefined references',r'Citation .+ undefined',r'Missing character:',r'Overfull \\hbox']:
        assert not re.search(pattern,log,re.M),f'LaTeX QA failed: {pattern}'
    output=ROOT/'report/BAO_CAO_MOTIP.pdf'
    if output.exists():
        backup=ROOT/'report/.backup';backup.mkdir(exist_ok=True)
        destination=backup/'BAO_CAO_MOTIP_before_latex.pdf'
        if not destination.exists():shutil.copy2(output,destination)
    shutil.copy2(LATEX/'build/main.pdf',output)
    print('XeLaTeX/BibTeX completed. Primary report PDF updated; inspect rendered pages before handoff.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-image',action='store_true')
    main(parser.parse_args().rebuild_image)
