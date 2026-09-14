#!/usr/bin/env python
"""Treat each WD_P model as a third rater and calculate Gwet AC2 and ICC(A,1)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def icc_a1(x):
    x=np.asarray(x,float); n,k=x.shape
    grand=x.mean(); row=x.mean(1); col=x.mean(0)
    msr=k*np.square(row-grand).sum()/(n-1)
    msc=n*np.square(col-grand).sum()/(k-1)
    resid=x-row[:,None]-col[None,:]+grand
    mse=np.square(resid).sum()/((n-1)*(k-1))
    den=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/den) if den else np.nan

def gwet_ac2(x,categories,quadratic=True):
    x=np.asarray(x,int); n,r=x.shape; cats=np.asarray(categories,int); q=len(cats)
    if quadratic: w=1-((cats[:,None]-cats[None,:])/(q-1))**2
    else: w=(cats[:,None]==cats[None,:]).astype(float)
    index={v:i for i,v in enumerate(cats)}
    pa=[]
    for row in x:
        vals=[index[v] for v in row]
        pa.append(sum(w[vals[i],vals[j]] for i in range(r) for j in range(i+1,r))/(r*(r-1)/2))
    p=np.array([(x==v).sum()/(n*r) for v in cats])
    pe=sum(w[i,j]*p[i]*(1-p[j]) for i in range(q) for j in range(q))/(q-1)
    return float((np.mean(pa)-pe)/(1-pe)) if pe < 1 else np.nan

def find_col(df,names):
    for n in names:
        if n in df: return n
    raise ValueError(f"None of these prediction columns exists: {names}")

def load_prediction(path,labels,kind):
    p=pd.read_csv(path,encoding="utf-8-sig",low_memory=False)
    if "status" in p:
        p=p[p.status.astype(str).str.lower().isin(["ok","success"])]
    key="sample_id" if "sample_id" in p and "sample_id" in labels else "segment_uid"
    if key not in p or key not in labels: raise ValueError(f"No shared identifier in {path}")
    if kind=="regression":
        col=find_col(p,["WD_prediction","WD_P_pred","prediction"]); p["ai_rating"]=pd.to_numeric(p[col],errors="coerce").clip(1,5); p["ai_binary"]=(p.ai_rating>=2).astype(float)
    else:
        score=next((c for c in ["wd_p_score","WD_P_score"] if c in p),None)
        prob=find_col(p,["WD_probability","probability"])
        p["ai_binary"]=(pd.to_numeric(p[prob],errors="coerce")>=.5).astype(float)
        p["ai_rating"]=pd.to_numeric(p[score],errors="coerce").clip(1,5) if score else 1+p.ai_binary
        p["ordinal_available"]=bool(score)
    if "ordinal_available" not in p: p["ordinal_available"]=True
    keep=p[[key,"ai_rating","ai_binary","ordinal_available"]].dropna().drop_duplicates(key,keep="last")
    return labels.merge(keep,on=key,how="inner",validate="one_to_one")

def main(a):
    labels=pd.read_csv(a.labels,encoding="utf-8-sig",low_memory=False)
    labels["human1"]=pd.to_numeric(labels.WD_P_rater1,errors="coerce"); labels["human2"]=pd.to_numeric(labels.WD_P_rater2,errors="coerce")
    labels=labels.dropna(subset=["human1","human2"])
    specs=json.loads(Path(a.specs).read_text(encoding="utf-8")); rows=[]
    for s in specs:
        d=load_prediction(Path(s["path"]),labels,s["kind"])
        ordinal=np.column_stack([d.human1,d.human2,d.ai_rating.clip(1,5).round()]).astype(int)
        binary=np.column_stack([(d.human1>=2).astype(int),(d.human2>=2).astype(int),d.ai_binary.astype(int)])
        continuous=np.column_stack([d.human1,d.human2,d.ai_rating])
        has_ordinal=bool(d.ordinal_available.all())
        rows.append(dict(experiment=s["experiment"],model=s["model"],kind=s["kind"],N=len(d),AC2_ordinal_quadratic=gwet_ac2(ordinal,range(1,6),True) if has_ordinal else np.nan,ICC_A1_ordinal_continuous=icc_a1(continuous) if has_ordinal else np.nan,AC2_binary=gwet_ac2(binary,range(2),False),ICC_A1_binary=icc_a1(binary)))
    out=pd.DataFrame(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False,encoding="utf-8-sig"); print(out.to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--labels",type=Path,required=True); p.add_argument("--specs",type=Path,required=True,help="JSON list with experiment, model, kind, path"); p.add_argument("--output",type=Path,required=True); main(p.parse_args())
