#!/usr/bin/env python
"""Direct visual zero-shot or 3+3 few-shot WD_P inference on one fixed fold."""
from __future__ import annotations

import argparse, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MANUAL = """Rate only PATIENT withdrawal (WD_P) using the 3RS v2022 concept.
Withdrawal is movement away from the therapist or therapeutic work, such as
visible disengagement, shutting down, distancing, prolonged avoidance, or
constricted participation. Ordinary stillness, looking down while thinking,
blinking, posture shifts, or neutral facial expression alone are insufficient.
Use only visible behavior in the chronological frames. Do not infer speech,
tone, diagnosis, identity, or behavior from filenames. Score 1 means no marker;
2 subtle/ambiguous; 3 clear; 4 strong; 5 very strong. WD_P is positive at >=2.
Return JSON only: {"wd_p_score": 1, "probability": 0.0, "reason": "visible evidence"}.
"""

def load_frames(cache: Path, sample_id: str, count: int | None = None):
    paths = sorted((cache / sample_id).glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No cached frames for {sample_id}")
    if count and len(paths) > count:
        paths = [paths[i] for i in np.linspace(0, len(paths)-1, count, dtype=int)]
    return [Image.open(p).convert("RGB") for p in paths]

def image_content(frames):
    return [{"type":"image", "image":x} for x in frames]

def parse(raw: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    m = re.search(r"\{.*\}", text, flags=re.S)
    obj = json.loads(m.group(0) if m else text)
    score = int(round(float(obj["wd_p_score"])))
    if score < 1 or score > 5: raise ValueError(f"Invalid score {score}")
    prob = float(obj.get("probability", 1.0 if score >= 2 else 0.0))
    return score, min(1.0, max(0.0, prob)), str(obj.get("reason", ""))

def metrics(df):
    ok=df[df.status.eq("ok")].copy(); y=ok.WD_consensus.astype(int).to_numpy(); p=ok.WD_probability.astype(float).to_numpy(); pred=(p>=.5).astype(int)
    tp=int(((y==1)&(pred==1)).sum()); tn=int(((y==0)&(pred==0)).sum()); fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum())
    rec=tp/(tp+fn) if tp+fn else 0.; spec=tn/(tn+fp) if tn+fp else 0.; prec=tp/(tp+fp) if tp+fp else 0.; f1=2*prec*rec/(prec+rec) if prec+rec else 0.
    return dict(N=len(ok),TP=tp,TN=tn,FP=fp,FN=fn,accuracy=(tp+tn)/len(ok),balanced_accuracy=(rec+spec)/2,precision=prec,recall=rec,specificity=spec,f1=f1,predicted_positive_rate=float(pred.mean()),errors=int((df.status!="ok").sum()))

def main(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); pred_path=out/"predictions.csv"
    data=pd.read_csv(a.manifest,encoding="utf-8-sig",low_memory=False); test=data[(data.split.astype(str)=="test") & data.WD_consensus.notna()].copy()
    demos=[]
    if a.shot=="few":
        train=data[(data.split.astype(str)=="train") & data.WD_consensus.notna()].copy()
        for label in (0,1): demos.extend(train[train.WD_consensus.astype(float).eq(label)].sort_values("sample_id").head(a.examples_per_class).to_dict("records"))
        if len(demos)!=2*a.examples_per_class: raise ValueError("Insufficient consensus demonstrations")
    existing=pd.read_csv(pred_path,encoding="utf-8-sig") if pred_path.exists() else pd.DataFrame(); done=set(existing.loc[existing.status.eq("ok"),"sample_id"].astype(str)) if not existing.empty else set()
    if a.prepare_only:
        print(json.dumps(dict(shot=a.shot,test=len(test),demonstrations=[x["sample_id"] for x in demos]),indent=2)); return
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,device_map="auto",torch_dtype=torch.bfloat16,quantization_config=quant,attn_implementation=a.attention).eval(); processor=AutoProcessor.from_pretrained(a.model)
    device=next(model.parameters()).device
    for i,row in enumerate(test.to_dict("records"),1):
        sid=str(row["sample_id"])
        if sid in done: continue
        try:
            messages=[]
            for d in demos:
                frames=load_frames(Path(a.frame_cache),str(d["sample_id"]),a.demo_frames)
                messages.append({"role":"user","content":image_content(frames)+[{"type":"text","text":MANUAL}]})
                score=2 if int(float(d["WD_consensus"]))==1 else 1
                messages.append({"role":"assistant","content":json.dumps({"wd_p_score":score,"probability":float(score>=2),"reason":"labeled demonstration"})})
            frames=load_frames(Path(a.frame_cache),sid,a.target_frames)
            messages.append({"role":"user","content":image_content(frames)+[{"type":"text","text":MANUAL}]})
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            images=[c["image"] for m in messages for c in (m["content"] if isinstance(m["content"],list) else []) if c.get("type")=="image"]
            inputs=processor(text=[text],images=images,padding=True,return_tensors="pt").to(device); started=time.time()
            with torch.inference_mode(): ids=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
            raw=processor.batch_decode([o[len(x):] for x,o in zip(inputs.input_ids,ids)],skip_special_tokens=True)[0]; score,prob,reason=parse(raw)
            result=dict(sample_id=sid,segment_uid=row.get("segment_uid",sid),patient_id=row["patient_id"],WD_consensus=int(float(row["WD_consensus"])),WD_probability=prob,wd_p_score=score,status="ok",error="",reason=reason,raw_output=raw,inference_sec=time.time()-started)
        except Exception as e:
            result=dict(sample_id=sid,segment_uid=row.get("segment_uid",sid),patient_id=row["patient_id"],WD_consensus=int(float(row["WD_consensus"])),WD_probability=np.nan,wd_p_score=np.nan,status="error",error=repr(e),reason="",raw_output="",inference_sec=np.nan)
        if not existing.empty: existing=existing[existing.sample_id.astype(str)!=sid]
        existing=pd.concat([existing,pd.DataFrame([result])],ignore_index=True); existing.to_csv(pred_path,index=False,encoding="utf-8-sig")
        print(f"[{i:04d}/{len(test):04d}] {sid} {result['status']}",flush=True); torch.cuda.empty_cache()
    summary=dict(shot=a.shot,selected=len(test),successful=int((existing.status=="ok").sum()),errors=int((existing.status!="ok").sum()),consensus_evaluation_metrics=metrics(existing))
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--frame-cache",type=Path,required=True); p.add_argument("--shot",choices=["zero","few"],required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen3-VL-8B-Instruct"); p.add_argument("--attention",choices=["sdpa","eager","flash_attention_2"],default="sdpa"); p.add_argument("--examples-per-class",type=int,default=3); p.add_argument("--demo-frames",type=int,default=4); p.add_argument("--target-frames",type=int,default=16); p.add_argument("--max-new-tokens",type=int,default=96); p.add_argument("--prepare-only",action="store_true"); main(p.parse_args())
