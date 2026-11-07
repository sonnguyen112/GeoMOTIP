#!/usr/bin/env python3
"""Run the included geometry/runtime regression suite in the MOTIP environment."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'src/motip'
env=os.environ.copy()
env['PYTHONPATH']=os.pathsep.join([str(VENDOR),str(VENDOR/'models/ops/build/lib.linux-x86_64-cpython-312'),env.get('PYTHONPATH','')])
raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],cwd=VENDOR,env=env))
