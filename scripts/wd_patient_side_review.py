#!/usr/bin/env python
"""Create a four-card patient-side HTML review and safely apply its decisions."""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def unresolved_cases(cohort_root):
    excluded=pd.read_csv(cohort_root/'excluded_patient_side.csv',encoding='utf-8-sig',low_memory=False)
    audit=pd.read_csv(cohort_root/'video_role_audit.csv',encoding='utf-8-sig',low_memory=False)
    if excluded.empty:raise ValueError('No unresolved patient-side rows remain')
    counts=(excluded.groupby(['patient_id','video'],as_index=False).size()
            .rename(columns={'size':'segments'}))
    counts['patient_id']=counts.patient_id.astype(str);audit['patient_id']=audit.patient_id.astype(str)
    choices=(counts.merge(audit[['patient_id','video','video_path','video_found']],
                          on=['patient_id','video'],validate='one_to_one')
             .sort_values(['patient_id','segments','video'],ascending=[True,False,True])
             .groupby('patient_id',as_index=False).first())
    if not choices.video_found.astype(bool).all():raise ValueError('A selected review video is missing')
    return choices,counts


def extract_frames(video_path,destination,frames=3):
    import cv2
    capture=cv2.VideoCapture(str(video_path))
    if not capture.isOpened():raise RuntimeError(f'Cannot open {video_path}')
    total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(capture.get(cv2.CAP_PROP_FPS))
    if total<1:raise RuntimeError(f'Video has no frames: {video_path}')
    paths=[]
    for number,fraction in enumerate([.25,.5,.75][:frames],1):
        capture.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(total-1,int(total*fraction))))
        ok,image=capture.read()
        if not ok:raise RuntimeError(f'Could not read frame {fraction:.0%} from {video_path}')
        height,width=image.shape[:2]
        if width>900:
            scale=900/width;image=cv2.resize(image,(900,max(1,int(height*scale))))
            height,width=image.shape[:2]
        cv2.line(image,(width//2,0),(width//2,height),(30,220,255),3)
        cv2.putText(image,'LEFT',(18,42),cv2.FONT_HERSHEY_SIMPLEX,1.2,(30,220,255),3,cv2.LINE_AA)
        cv2.putText(image,'RIGHT',(width//2+18,42),cv2.FONT_HERSHEY_SIMPLEX,1.2,(30,220,255),3,cv2.LINE_AA)
        path=destination/f'frame_{number}.jpg'
        if not cv2.imwrite(str(path),image,[int(cv2.IMWRITE_JPEG_QUALITY),88]):raise RuntimeError(f'Cannot write {path}')
        paths.append(path)
    capture.release();return paths,fps,total


def build(args):
    cohort=args.cohort_root.resolve();output=args.output.resolve();assets=output/'assets'
    output.mkdir(parents=True,exist_ok=True);assets.mkdir(exist_ok=True)
    choices,counts=unresolved_cases(cohort);cards=[];manifest=[]
    for row in choices.itertuples(index=False):
        folder=assets/str(row.patient_id);folder.mkdir(exist_ok=True)
        paths,fps,total=extract_frames(Path(row.video_path),folder)
        rel=[p.relative_to(output).as_posix() for p in paths]
        total_patient=int(counts.loc[counts.patient_id==str(row.patient_id),'segments'].sum())
        manifest.append({'patient_id':str(row.patient_id),'video':str(row.video),
                         'video_path':str(row.video_path),'segments':total_patient})
        images=''.join(f'<img src="{html.escape(path)}" alt="Review frame">' for path in rel)
        cards.append(f'''<section class="card" data-patient="{html.escape(str(row.patient_id))}" data-video="{html.escape(str(row.video))}">
<h2>Patient {html.escape(str(row.patient_id))}</h2><p>{html.escape(str(row.video))} · {total_patient} eligible segments</p>
<div class="frames">{images}</div><div class="choices">
<button data-side="left">Patient is LEFT</button><button data-side="right">Patient is RIGHT</button>
</div><p class="answer">Choose the patient's side as seen by you.</p></section>''')
    template='''<!doctype html><html><head><meta charset="utf-8"><title>WD patient-side review</title>
<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;background:#f5f7fa;color:#172b4d}.intro,.card{background:white;border:1px solid #d8dee9;border-radius:12px;padding:18px;margin:16px 0}.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.frames img{width:100%;border-radius:6px}.choices{display:flex;gap:12px;margin-top:14px}.choices button,#save{font-size:16px;padding:10px 18px;border:2px solid #2867b2;border-radius:8px;background:white;cursor:pointer}.choices button.selected{background:#2867b2;color:white}.answer{font-weight:600}#save:disabled{opacity:.45;cursor:not-allowed}@media(max-width:800px){.frames{grid-template-columns:1fr}}</style></head>
<body><div class="intro"><h1>Four patient-side decisions</h1><p>For each card, identify the patient—not the therapist. LEFT and RIGHT mean the viewer's left and right in the image. Three frames come from one representative session.</p><p>Nothing is uploaded. When all four are selected, download the decision file.</p></div>
__CARDS__<button id="save" disabled>Download decisions JSON</button>
<script>const decisions={};document.querySelectorAll('.card button').forEach(button=>button.onclick=()=>{const card=button.closest('.card');card.querySelectorAll('button').forEach(x=>x.classList.remove('selected'));button.classList.add('selected');decisions[card.dataset.patient]={patient_id:card.dataset.patient,video:card.dataset.video,patient_side:button.dataset.side};card.querySelector('.answer').textContent='Selected: '+button.dataset.side.toUpperCase();document.querySelector('#save').disabled=Object.keys(decisions).length!==document.querySelectorAll('.card').length;localStorage.setItem('wdPatientSides',JSON.stringify(decisions));});
const saved=JSON.parse(localStorage.getItem('wdPatientSides')||'{}');Object.values(saved).forEach(d=>{const b=document.querySelector(`[data-patient="${d.patient_id}"] button[data-side="${d.patient_side}"]`);if(b)b.click();});
document.querySelector('#save').onclick=()=>{const payload={format:'wd_patient_side_decisions_v1',created_at:new Date().toISOString(),decisions:Object.values(decisions)};const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='wd_patient_side_decisions.json';a.click();URL.revokeObjectURL(a.href);};</script></body></html>'''
    (output/'index.html').write_text(template.replace('__CARDS__','\n'.join(cards)),encoding='utf-8')
    (output/'review_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'patients':len(manifest),'cards':manifest,'html':str((output/'index.html').resolve())},indent=2))


def apply(args):
    cohort=args.cohort_root.resolve();decisions=json.loads(args.decisions.read_text(encoding='utf-8-sig'))
    if decisions.get('format')!='wd_patient_side_decisions_v1':raise ValueError('Unexpected decisions format')
    choices,_=unresolved_cases(cohort);expected={str(x) for x in choices.patient_id};rows=decisions.get('decisions',[])
    observed={str(x.get('patient_id')) for x in rows}
    if observed!=expected:raise ValueError(f'Decisions must cover exactly {sorted(expected)}; found {sorted(observed)}')
    allowed={(str(r.patient_id),str(r.video)) for r in choices.itertuples(index=False)}
    role_path=args.role_cache.resolve();cache=json.loads(role_path.read_text(encoding='utf-8-sig'))
    changes=[]
    for row in rows:
        patient,video,side=str(row['patient_id']),str(row['video']),str(row['patient_side']).lower()
        if (patient,video) not in allowed:raise ValueError(f'{patient}/{video} was not the reviewed representative video')
        if side not in {'left','right'}:raise ValueError(f'Invalid side for {patient}: {side}')
        existing=cache.get(video,{})
        if isinstance(existing,dict) and existing.get('patient_side') not in {None,'',side}:
            raise ValueError(f'Conflicting existing cache decision for {video}')
        cache[video]={**(existing if isinstance(existing,dict) else {}),'patient_side':side,
                      'patient_id':patient,'source':'manual_html_review','reviewed_at':decisions.get('created_at','')}
        changes.append({'patient_id':patient,'video':video,'patient_side':side})
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S');backup=role_path.with_name(role_path.stem+f'.backup_{stamp}'+role_path.suffix)
    shutil.copy2(role_path,backup)
    role_path.write_text(json.dumps(cache,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'updated':str(role_path),'backup':str(backup),'changes':changes},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    common={'cohort_root':ROOT/'output/wd_multimodal_master_expanded'}
    b=sub.add_parser('build');b.add_argument('--cohort-root',type=Path,default=common['cohort_root']);b.add_argument('--output',type=Path,default=ROOT/'output/wd_patient_side_review');b.set_defaults(func=build)
    a=sub.add_parser('apply');a.add_argument('--cohort-root',type=Path,default=common['cohort_root']);a.add_argument('--decisions',type=Path,required=True);a.add_argument('--role-cache',type=Path,default=ROOT/'output/qwen3vl_visual_experiment_v5/patient_role_cache.json');a.set_defaults(func=apply)
    args=p.parse_args();args.func(args)
