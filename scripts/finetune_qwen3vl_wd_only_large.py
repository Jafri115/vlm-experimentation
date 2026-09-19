#!/usr/bin/env python
"""Large patient-disjoint WD_P-only Qwen3-VL QLoRA experiment.

Uses nearly all double-rated, role-cache-resolved segments. CF_P is removed
from the objective. The model uses 16 frames across the full labelled minute,
mean-all multimodal pooling, and selects the best checkpoint by WD_P validation
MAE only.

Place this file beside finetune_qwen3vl_rupture_pilot.py in the scripts folder.
"""
from __future__ import annotations

import argparse, gc, itertools, json, math, time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import average_precision_score, roc_auc_score

import finetune_qwen3vl_rupture_pilot as base

WD_PROMPT = """The images are chronological frames sampled across one labelled minute
of a psychotherapy session.

Focus only on the visible patient/person being observed.

Create an internal visual representation useful for predicting the human-rated
3RS v2022 Patient Moves Away (WD_P) salience score on the 1-5 scale.

Represent directly visible patient behavior across the WHOLE MINUTE.
Pay particular attention to:
- persistence versus isolated behavior,
- repetition,
- strength/prominence of visible behavior,
- changes across the sequence,
- whether a behavioral pattern clearly shapes or dominates much of the minute.

Preserve distinctions corresponding to:
1 = little or no visual evidence associated with withdrawal
2 = weak or ambiguous visual evidence
3 = clear visual evidence
4 = clearly elevated evidence through persistence, repetition, or prominence
5 = very strong or dominant visual evidence across the minute

A single weak visual cue should not by itself imply a high WD_P score.
Multiple weak cues should not automatically imply a 4 or 5.
A high score should be supported by strong, sustained, repeated, or dominant visual
patterns across the minute.

The target is the human 3RS WD_P rating. Visual behavior alone may not establish
whether a rupture actually occurred, so do not invent therapeutic meaning that is
not visible.

Use only directly visible patient behavior.
Do not infer speech content, motivation, diagnosis, personality, hidden emotion,
or unavailable therapist-patient dialogue.
Do not use audio or transcript.
"""


class WDHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return 1.0 + 4.0 * torch.sigmoid(self.net(x.float()))


def patient_stats(df, threshold):
    x = df.copy()
    x["positive"] = (x["WD_P_mean"] >= threshold).astype(int)
    s = x.groupby("patient_id").agg(
        n=("segment_id", "size"),
        pos=("positive", "sum"),
        wd_mean=("WD_P_mean", "mean"),
    ).reset_index()
    s["rate"] = s["pos"] / s["n"]
    return s


def choose_split(df, threshold, n_val, n_test, seed):
    """Choose 2 val + 2 test patients with roughly representative WD prevalence."""
    s = patient_stats(df, threshold)
    ids = s["patient_id"].astype(str).tolist()
    if len(ids) < n_val + n_test + 3:
        raise RuntimeError(f"Only {len(ids)} patients available after filtering")

    stat = {str(r.patient_id): r for r in s.itertuples(index=False)}
    total_n = float(s["n"].sum())
    overall_rate = float(s["pos"].sum()) / total_n
    target_val_n = total_n * n_val / len(ids)
    target_test_n = total_n * n_test / len(ids)
    rng = np.random.default_rng(seed + 991)
    jitter = {p: float(rng.uniform(0, 1e-7)) for p in ids}

    def info(group):
        n = sum(int(stat[p].n) for p in group)
        pos = sum(int(stat[p].pos) for p in group)
        neg = n - pos
        rate = pos / n if n else 0.0
        return n, pos, neg, rate

    def score(group, target_n):
        n, pos, neg, rate = info(group)
        penalty = 0.0
        if pos == 0: penalty += 20.0
        elif pos < 3: penalty += 2.0
        if neg == 0: penalty += 20.0
        return abs(n-target_n)/max(target_n,1) + 2*abs(rate-overall_rate) + penalty + sum(jitter[p] for p in group)

    best = None
    for test in itertools.combinations(ids, n_test):
        remain = [p for p in ids if p not in test]
        ts = score(test, target_test_n)
        for val in itertools.combinations(remain, n_val):
            train = [p for p in remain if p not in val]
            _, train_pos, train_neg, _ = info(train)
            if train_pos == 0 or train_neg == 0:
                continue
            sc = ts + score(val, target_val_n)
            if best is None or sc < best[0]:
                best = (sc, train, list(val), list(test))

    if best is None:
        raise RuntimeError("Could not construct patient-disjoint split")

    _, train, val, test = best
    out = {p: "train" for p in train}
    out.update({p: "val" for p in val})
    out.update({p: "test" for p in test})
    return out


def build_manifest(args):
    print("\nPREPARING LARGE WD-ONLY MANIFEST")
    print("=" * 72)
    grouped = base.aggregate_labels(Path(args.labels_csv), args.min_coders)
    print("Double-rated candidate segments:", len(grouped))
    print("Patients before media filtering:", grouped["patient_id"].nunique())

    role_cache = base.load_role_cache(Path(args.role_cache) if args.role_cache else None)
    cand = base.resolve_media(
        grouped,
        video_root=Path(args.video_root),
        role_cache=role_cache,
        require_role_cache=args.require_role_cache,
    ).dropna(subset=["WD_P_mean"]).copy()

    split_map = choose_split(cand, args.positive_threshold, args.val_patients, args.test_patients, args.seed)
    cand["split"] = cand["patient_id"].astype(str).map(split_map)
    cand = cand[cand["split"].notna()].copy()
    cand["WD_positive"] = (cand["WD_P_mean"] >= args.positive_threshold).astype(int)
    cand["sample_id"] = cand.apply(lambda r: f"{Path(str(r['video'])).stem}_seg{int(r['segment_id']):03d}", axis=1)

    keep = ["sample_id","split","video","video_path","patient_id","session_id","segment_id",
            "segment_start","segment_end","start_sec","end_sec","patient_side","n_coders","coders",
            "WD_P_mean","WD_P_disagreement","WD_positive"]
    m = cand[keep].sort_values(["split","patient_id","video","segment_id"]).reset_index(drop=True)
    out = Path(args.manifest_out); out.parent.mkdir(parents=True, exist_ok=True); m.to_csv(out, index=False)

    summary = m.groupby("split").agg(
        segments=("sample_id","size"), patients=("patient_id","nunique"),
        WD_positive=("WD_positive","sum"), WD_mean=("WD_P_mean","mean"), WD_std=("WD_P_mean","std")
    )
    summary["WD_positive_rate"] = summary["WD_positive"] / summary["segments"]
    print("\nSaved manifest:", out)
    print("Total segments:", len(m), "| patients:", m["patient_id"].nunique())
    print("\nSplit summary\n" + summary.to_string())
    print("\nPatient split")
    print(m.groupby(["split","patient_id"]).agg(segments=("sample_id","size"),WD_positive=("WD_positive","sum"),WD_mean=("WD_P_mean","mean")).reset_index().to_string(index=False))
    return m


def load_fixed_manifest(path):
    """Load an existing patient-disjoint master manifest without resplitting."""
    m = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"sample_id", "split", "patient_id", "WD_P_mean", "video_path", "patient_side"}
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"Fixed manifest missing columns: {sorted(missing)}")
    m["split"] = m["split"].astype(str).str.lower().str.strip()
    if not set(m["split"]).issubset({"train", "val", "test"}):
        raise ValueError("Fixed manifest split must contain only train/val/test")
    m["WD_P_mean"] = pd.to_numeric(m["WD_P_mean"], errors="raise")
    if "segment_id" not in m and "segment_number" in m:
        m["segment_id"] = pd.to_numeric(m["segment_number"], errors="raise")
    print("\nLOADED FIXED PATIENT-DISJOINT MANIFEST:", path)
    print(m.groupby("split").agg(segments=("sample_id", "size"), patients=("patient_id", "nunique")))
    patient_sets = [set(m.loc[m.split == s, "patient_id"].astype(str)) for s in ("train", "val", "test")]
    if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Fixed manifest is not patient-disjoint")
    return m


def prepare_inputs(processor, frames: Sequence[Image.Image], device):
    content = [{"type":"image","image":f} for f in frames]
    content.append({"type":"text","text":WD_PROMPT})
    inputs = processor.apply_chat_template(
        [{"role":"user","content":content}], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    moved = {}
    for k,v in inputs.items():
        if not torch.is_tensor(v): moved[k] = v
        elif k in {"pixel_values","pixel_values_videos"}: moved[k] = v.to(device=device, dtype=torch.bfloat16)
        else: moved[k] = v.to(device=device)
    return moved


def predict_wd(model, head, processor, frames, device):
    inputs = prepare_inputs(processor, frames, device)
    hidden = base.get_backbone(model)(**inputs, use_cache=False, return_dict=True).last_hidden_state
    mask = inputs.get("attention_mask")
    if mask is None:
        pooled = hidden.mean(dim=1)
    else:
        w = mask.bool().unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
    return head(pooled)


def metrics(rows, threshold):
    d = pd.DataFrame(rows)
    if d.empty: return {}
    yt = d.WD_P_true.to_numpy(float); yp = d.WD_P_pred.to_numpy(float)
    err = yp-yt
    rho = float(spearmanr(yt,yp).statistic) if len(yt)>=3 and not np.allclose(yt,yt[0]) else float("nan")
    t = (yt>=threshold).astype(int); p = (yp>=threshold).astype(int)
    tp=int(((t==1)&(p==1)).sum()); fp=int(((t==0)&(p==1)).sum()); fn=int(((t==1)&(p==0)).sum()); tn=int(((t==0)&(p==0)).sum())
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    out = dict(
        n=len(d), WD_P_MAE=float(np.abs(err).mean()), WD_P_RMSE=float(np.sqrt((err**2).mean())), WD_P_spearman=rho,
        WD_P_true_mean=float(yt.mean()), WD_P_true_std=float(yt.std()), WD_P_true_min=float(yt.min()), WD_P_true_max=float(yt.max()),
        WD_P_pred_mean=float(yp.mean()), WD_P_pred_std=float(yp.std()), WD_P_pred_min=float(yp.min()), WD_P_pred_max=float(yp.max()),
        WD_ge3_TP=tp, WD_ge3_FP=fp, WD_ge3_FN=fn, WD_ge3_TN=tn,
        WD_ge3_precision=precision, WD_ge3_recall=recall, WD_ge3_f1=f1,
    )
    if len(np.unique(t))==2:
        out["WD_ge3_AUPRC"] = float(average_precision_score(t,yp))
        out["WD_ge3_AUROC"] = float(roc_auc_score(t,yp))
    return out


@torch.no_grad()
def evaluate(df, name, model, head, processor, frame_cache, device, args, outdir, max_examples: Optional[int]):
    model.eval(); head.eval(); rows=[]
    e = df.copy()
    if max_examples and max_examples>0: e=e.head(max_examples)
    print(f"\nEvaluating {name}: {len(e)} segments")
    for i,row in enumerate(e.itertuples(index=False),1):
        try:
            pred=float(predict_wd(model,head,processor,frame_cache.build(row),device)[0,0].detach().float().cpu())
            rows.append(dict(sample_id=row.sample_id,patient_id=row.patient_id,video=row.video,segment_id=row.segment_id,WD_P_true=float(row.WD_P_mean),WD_P_pred=pred))
            print(f"  [{i:04d}/{len(e):04d}] {row.sample_id} | WD {row.WD_P_mean:.1f}->{pred:.2f}", flush=True)
        except Exception as exc:
            print(f"  ERROR {row.sample_id}: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(outdir/f"{name}_predictions.csv",index=False)
    met=metrics(rows,args.positive_threshold); base.write_json(outdir/f"{name}_metrics.json",met)
    print(f"\n{name.upper()} METRICS\n"+json.dumps(met,indent=2))
    return met


def train(args, manifest):
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True); base.seed_everything(args.seed)
    tr=manifest[manifest.split=="train"].copy(); va=manifest[manifest.split=="val"].copy(); te=manifest[manifest.split=="test"].copy()

    model,processor,old_head=base.build_model_and_processor(args); del old_head; gc.collect(); torch.cuda.empty_cache()
    hidden=int(model.get_base_model().config.text_config.hidden_size)
    head=WDHead(hidden,args.head_dropout).to("cuda",dtype=torch.float32)
    cropper=base.PatientCropper(Path(args.yunet_model),output_width=args.frame_width)
    cache=base.FrameCache(Path(args.frame_cache),cropper,args.num_frames)
    device=torch.device("cuda:0")

    opt=torch.optim.AdamW([
        {"params":[p for p in model.parameters() if p.requires_grad],"lr":args.learning_rate,"weight_decay":args.weight_decay},
        {"params":list(head.parameters()),"lr":args.head_learning_rate,"weight_decay":args.weight_decay},
    ])
    planned=math.ceil(len(tr)/args.grad_accum_steps)*args.epochs
    if args.max_train_steps>0: planned=min(planned,args.max_train_steps)

    print("\nTRAINING PLAN\n"+"="*72)
    print("Experiment: WD_P-only large-data QLoRA")
    print("Train:",len(tr),"segments /",tr.patient_id.nunique(),"patients")
    print("Val:",len(va),"segments /",va.patient_id.nunique(),"patients")
    print("Test:",len(te),"segments /",te.patient_id.nunique(),"patients")
    print("Frames/minute:",args.num_frames,"| width:",args.frame_width,"| pooling: mean_all")
    print("Epochs:",args.epochs,"| planned optimizer steps:",planned,"| grad accumulation:",args.grad_accum_steps)
    print("Checkpoint criterion: WD_P validation MAE only")
    base.write_json(outdir/"run_config.json",{**vars(args),"target":"WD_P_mean","CF_P_in_loss":False,"pooling":"mean_all","prompt":WD_PROMPT})

    best=float("inf"); best_epoch=None; best_step=None; best_lora=None; best_head=None
    step=0; accum=0; opt.zero_grad(set_to_none=True); stop=False
    for epoch in range(1,args.epochs+1):
        print(f"\nEPOCH {epoch}/{args.epochs}\n"+"="*72); model.train(); head.train()
        edf=tr.sample(frac=1.0,random_state=args.seed+epoch).reset_index(drop=True); runloss=0.0; ok=0
        for idx,row in enumerate(edf.itertuples(index=False),1):
            if args.max_train_steps>0 and step>=args.max_train_steps: stop=True; break
            t0=time.time()
            try:
                target=torch.tensor([[float(row.WD_P_mean)]],device=device,dtype=torch.float32)
                pred=predict_wd(model,head,processor,cache.build(row),device)
                loss=F.smooth_l1_loss(pred.float(),target,beta=args.huber_beta,reduction="mean")
                (loss/args.grad_accum_steps).backward(); accum+=1; ok+=1; runloss+=float(loss.detach().cpu())
                if accum>=args.grad_accum_steps or idx==len(edf):
                    trainable=[p for p in model.parameters() if p.requires_grad]+list(head.parameters())
                    torch.nn.utils.clip_grad_norm_(trainable,args.max_grad_norm); opt.step(); opt.zero_grad(set_to_none=True); step+=1; accum=0
                pv=float(pred[0,0].detach().float().cpu())
                print(f"  epoch={epoch} sample={idx:04d}/{len(edf):04d} opt_step={step:04d} loss={float(loss.detach()):.4f} WD={row.WD_P_mean:.1f}->{pv:.2f} time={time.time()-t0:.1f}s",flush=True)
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True); accum=0; torch.cuda.empty_cache(); raise
            except Exception as exc:
                print(f"  ERROR {getattr(row,'sample_id','?')}: {type(exc).__name__}: {exc}",flush=True); opt.zero_grad(set_to_none=True); accum=0
            finally:
                gc.collect(); torch.cuda.empty_cache()

        meanloss=runloss/max(ok,1); print(f"\nEpoch {epoch} mean training loss: {meanloss:.4f}")
        vm=evaluate(va,"val",model,head,processor,cache,device,args,outdir,args.max_val_examples)
        val_mae=vm.get("WD_P_MAE",float("inf"))
        ck=outdir/f"checkpoint_epoch_{epoch}"; ck.mkdir(parents=True,exist_ok=True); model.save_pretrained(ck/"adapter"); processor.save_pretrained(ck/"processor"); torch.save(head.state_dict(),ck/"wd_head.pt"); base.write_json(ck/"metrics.json",{"epoch":epoch,"optimizer_step":step,"mean_train_loss":meanloss,"val_metrics":vm})
        if val_mae<best:
            best=val_mae; best_epoch=epoch; best_step=step
            bd=outdir/"best"; bd.mkdir(parents=True,exist_ok=True); model.save_pretrained(bd/"adapter"); processor.save_pretrained(bd/"processor"); torch.save(head.state_dict(),bd/"wd_head.pt"); base.write_json(bd/"metrics.json",{"epoch":epoch,"optimizer_step":step,"mean_train_loss":meanloss,"val_metrics":vm})
            best_lora={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}; best_head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            print(f"New best checkpoint: epoch {epoch}, WD val MAE={best:.4f}",flush=True)
        if stop: break

    if best_lora is None: raise RuntimeError("No best checkpoint captured")
    print("\nRESTORING BEST WD CHECKPOINT\n"+"="*72)
    set_peft_model_state_dict(model,best_lora,adapter_name="default"); head.load_state_dict(best_head); head.to(device=device,dtype=torch.float32)
    print(f"Restored epoch {best_epoch} (step {best_step}, WD val MAE={best:.4f})")
    tm=evaluate(te,"test",model,head,processor,cache,device,args,outdir,args.max_test_examples)
    base.write_json(outdir/"final_summary.json",{"best_val_WD_P_MAE":best,"best_epoch":best_epoch,"best_optimizer_step":best_step,"test_metrics":tm})
    print("\nFINISHED\nBest adapter:",outdir/"best"/"adapter","\nBest WD head:",outdir/"best"/"wd_head.pt","\nTest predictions:",outdir/"test_predictions.csv")


def parser():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--labels-csv"); p.add_argument("--input-manifest",help="Existing train/val/test manifest; bypasses label aggregation and split generation."); p.add_argument("--video-root",default=r"C:\Data\Sequence_model\Memopsy_videos\CONVERTED")
    p.add_argument("--role-cache",default=r".\output\qwen3vl_visual_experiment_v5\patient_role_cache.json"); p.add_argument("--yunet-model",default=r".\models\face_detection_yunet\face_detection_yunet_2026may.onnx")
    p.add_argument("--manifest-out",default=r".\output\qwen3vl_wd_only_large\manifest.csv"); p.add_argument("--output-dir",default=r".\output\qwen3vl_wd_only_large\run1"); p.add_argument("--frame-cache",default=r".\output\qwen3vl_rupture_finetune_16f_stable\frame_cache_16")
    p.add_argument("--min-coders",type=int,default=2); p.add_argument("--positive-threshold",type=float,default=3.0); p.add_argument("--val-patients",type=int,default=2); p.add_argument("--test-patients",type=int,default=2)
    p.add_argument("--require-role-cache",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--model-id",default="Qwen/Qwen3-VL-8B-Instruct"); p.add_argument("--attn-implementation",default="sdpa",choices=["sdpa","eager","flash_attention_2"]); p.add_argument("--no-4bit",action="store_true")
    p.add_argument("--num-frames",type=int,default=16); p.add_argument("--frame-width",type=int,default=224); p.add_argument("--lora-r",type=int,default=4); p.add_argument("--lora-alpha",type=int,default=8); p.add_argument("--lora-dropout",type=float,default=0.05); p.add_argument("--head-dropout",type=float,default=0.10)
    p.add_argument("--epochs",type=int,default=1); p.add_argument("--learning-rate",type=float,default=5e-5); p.add_argument("--head-learning-rate",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=0.01); p.add_argument("--grad-accum-steps",type=int,default=4); p.add_argument("--huber-beta",type=float,default=0.5); p.add_argument("--max-grad-norm",type=float,default=1.0)
    p.add_argument("--max-train-steps",type=int,default=0,help="0=all steps"); p.add_argument("--max-val-examples",type=int,default=0); p.add_argument("--max-test-examples",type=int,default=0); p.add_argument("--seed",type=int,default=42); p.add_argument("--prepare-only",action="store_true")
    return p


def main():
    args=parser().parse_args(); args.max_val_examples=None if args.max_val_examples<=0 else args.max_val_examples; args.max_test_examples=None if args.max_test_examples<=0 else args.max_test_examples
    print("QWEN3-VL WD_P-ONLY LARGE-DATA QLORA\n"+"="*72); print("Labels:",args.labels_csv); print("Video root:",args.video_root); print("Target: WD_P_mean only | pooling: mean_all")
    if args.input_manifest:
        m=load_fixed_manifest(args.input_manifest)
    else:
        if not args.labels_csv:
            raise SystemExit("Provide --input-manifest or --labels-csv")
        m=build_manifest(args)
    if args.prepare_only: print("\nPREPARE-ONLY complete. No model loaded."); return
    train(args,m)

if __name__=="__main__": main()
