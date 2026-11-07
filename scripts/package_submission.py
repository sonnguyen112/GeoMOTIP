#!/usr/bin/env python3
"""Export a light submission and a separate private checkpoint archive."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
EXPORT=ROOT/'submission'
ALLOWED_DIRS={'report','presentation','demo','figures','src','configs','experiments','sources','scripts','tests'}
ALLOWED_ROOT={'README.md','START_HERE.md','QUALITY_CHECK.json','requirements-inference.txt','requirements-documents.txt','.gitignore'}
SKIP_NAMES={'contact_sheet.png','page_01_preview.png','final_video_preview.png','crowded_preview.png'}


def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):d.update(chunk)
    return d.hexdigest()


def export(archive,files):
    temp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in files:
            relative=path.relative_to(ROOT)
            assert not any(p in relative.parts for p in ['.env','.session.json','.venv-docs','_cancelled_mot17'])
            kind=zipfile.ZIP_STORED if path.suffix=='.mp4' else zipfile.ZIP_DEFLATED
            z.write(path,arcname=str(Path(ROOT.name)/relative),compress_type=kind)
    temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    return dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),entries=len(files))


def documents_only():
    """Final submission policy: report PDF + presentation PPTX/PDF, nothing else."""
    EXPORT.mkdir(exist_ok=True)
    files=[ROOT/'report/BAO_CAO_MOTIP.pdf',ROOT/'presentation/THUYET_TRINH_MOTIP.pptx',ROOT/'presentation/THUYET_TRINH_MOTIP.pdf']
    assert all(p.exists() for p in files)
    previous=json.loads((EXPORT/'PACKAGE_MANIFEST.json').read_text()) if (EXPORT/'PACKAGE_MANIFEST.json').exists() else {}
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),submission=export(EXPORT/'BAI_NOP_MOTIP.zip',files),
        contents=[str(p.relative_to(ROOT)) for p in files],documents_only=True,
        excluded='HTML, video, source code, DOCX, checkpoints and datasets. Original local files are preserved.',
        no_upload_or_submission_performed=True)
    if previous.get('checkpoints'):manifest['checkpoints']=previous['checkpoints']
    (EXPORT/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(manifest,indent=2,ensure_ascii=False),flush=True)


def main(light_only=False):
    EXPORT.mkdir(exist_ok=True)
    files=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.is_symlink():continue
        relative=path.relative_to(ROOT)
        if relative.parts[0] not in ALLOWED_DIRS and str(relative) not in ALLOWED_ROOT:continue
        if relative.parts[0]=='personal_presentation' and (path.suffix=='.zip' or 'qa' in relative.parts or path.name=='QA.json'):continue
        if '__pycache__' in relative.parts or path.suffix in ['.pyc','.log','.tmp','.o','.ninja'] or path.name in SKIP_NAMES:continue
        if any(part.startswith('temp.linux-') for part in relative.parts):continue
        if any(part.startswith('.') for part in relative.parts[:-1]):continue
        if 'latex' in relative.parts and 'build' in relative.parts:continue
        if 'html' in relative.parts and 'renders' in relative.parts:continue
        if path.name=='upstream-template.tex':continue
        if path.name.startswith('.') and path.name!='.gitignore':continue
        files.append(path)
    main_archive=export(EXPORT/'BAI_NOP_MOTIP.zip',files)
    weights=[ROOT/'checkpoints'/name for name in ['e0_15pct_epoch6.pth','v3_15pct_epoch9.pth','g4_train_only_stats.pt']]
    for path in weights:assert path.exists()
    if light_only:
        checkpoint_archive=json.loads((EXPORT/'PACKAGE_MANIFEST.json').read_text())['checkpoints']
        assert sha(EXPORT/checkpoint_archive['file'])==checkpoint_archive['sha256']
    else:
        checkpoint_archive=export(EXPORT/'CHECKPOINTS_DEMO_MOTIP.zip',weights)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        submission=main_archive,checkpoints=checkpoint_archive,
        excluded='Full datasets, unused MOT17 preparation, virtualenv, credential/VPN files, cancelled jobs and large detector-pretrain weights not needed for inference',
        extraction='Extract both ZIPs into the same parent folder to merge MOTIP_CV_Course_Project/; only the light ZIP is needed to read/view the project.',
        no_upload_or_submission_performed=True)
    (EXPORT/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(manifest,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--light-only',action='store_true',help='Reuse the verified unchanged checkpoint archive.')
    parser.add_argument('--full-bundle',action='store_true',help='Explicitly include sources/video; not the final submission policy.')
    args=parser.parse_args()
    if args.full_bundle:main(args.light_only)
    else:documents_only()
