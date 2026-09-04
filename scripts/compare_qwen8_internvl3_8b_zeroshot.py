#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd

CONDITIONS = ("direct", "windowed_direct", "describe_judge")
CLASSES = ("NO_RUPTURE", "WD_P", "CF_P", "MIXED_P")


def binary_metrics(y_true, y_pred):
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    s = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "N": tp + tn + fp + fn, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": round(acc, 4), "balanced_accuracy": round((r + s) / 2, 4),
        "precision": round(p, 4), "recall": round(r, 4),
        "specificity": round(s, 4), "f1": round(f1, 4),
    }


def patient_type(row):
    wd = float(row.WD_P_mean) > 1
    cf = float(row.CF_P_mean) > 1
    if wd and cf: return "MIXED_P"
    if wd: return "WD_P"
    if cf: return "CF_P"
    return "NO_RUPTURE"


def load_preds(path, model):
    df = pd.read_csv(path)
    need = {"segment_idx", "condition", "status", "primary_label", "rupture_present"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{model}: missing columns {sorted(miss)}")
    df = df[df.status.astype(str).str.lower().eq("ok")].copy()
    df["segment_idx"] = pd.to_numeric(df.segment_idx, errors="raise").astype(int)
    df["rupture_present"] = pd.to_numeric(df.rupture_present, errors="coerce").fillna(0).astype(int)
    df["primary_label"] = df.primary_label.astype(str)
    df["condition"] = df.condition.astype(str)
    return df.drop_duplicates(["segment_idx", "condition"], keep="last")


def load_labels(path):
    x = pd.read_csv(path)
    need = {"eval_id", "human_binary", "WD_P_mean", "CF_P_mean"}
    miss = need - set(x.columns)
    if miss:
        raise ValueError(f"labels: missing columns {sorted(miss)}")
    x = x.rename(columns={"eval_id": "segment_idx"}).copy()
    x["segment_idx"] = pd.to_numeric(x.segment_idx, errors="raise").astype(int)
    x["legacy_any_3rs_marker"] = pd.to_numeric(x.human_binary, errors="coerce").astype(int)
    x["patient_only_rupture"] = ((pd.to_numeric(x.WD_P_mean) > 1) | (pd.to_numeric(x.CF_P_mean) > 1)).astype(int)
    x["patient_type_label"] = x.apply(patient_type, axis=1)
    return x


def class_metrics(y_true, y_pred):
    rows, f1s = [], []
    for c in CLASSES:
        tp = sum(a == c and b == c for a, b in zip(y_true, y_pred))
        fp = sum(a != c and b == c for a, b in zip(y_true, y_pred))
        fn = sum(a == c and b != c for a, b in zip(y_true, y_pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f)
        rows.append({"class": c, "support": sum(a == c for a in y_true),
                     "precision": round(p,4), "recall": round(r,4), "f1": round(f,4)})
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    return round(acc,4), round(sum(f1s)/len(f1s),4), rows


def main(a):
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    q = load_preds(a.qwen_predictions, "Qwen3-VL-8B")
    i = load_preds(a.internvl_predictions, "InternVL3-8B")
    lab = load_labels(a.labels)

    coverage, metrics, type_sum, type_class, side = [], [], [], [], []

    for cond in CONDITIONS:
        qids = set(q.loc[q.condition.eq(cond), "segment_idx"])
        iids = set(i.loc[i.condition.eq(cond), "segment_idx"])
        common = sorted(qids & iids & set(lab.segment_idx))
        coverage.append({"condition": cond, "qwen_completed": len(qids),
                         "internvl_completed": len(iids), "common_completed": len(common),
                         "qwen_only": len(qids-iids), "internvl_only": len(iids-qids)})
        if not common: continue

        l = lab[lab.segment_idx.isin(common)][["segment_idx","legacy_any_3rs_marker","patient_only_rupture",
                                               "patient_type_label","WD_P_mean","CF_P_mean"]].copy()
        for model, src in [("Qwen3-VL-8B", q), ("InternVL3-8B", i)]:
            s = src[src.condition.eq(cond) & src.segment_idx.isin(common)].merge(l, on="segment_idx", validate="one_to_one")
            for target in ("legacy_any_3rs_marker", "patient_only_rupture"):
                metrics.append({"model": model, "condition": cond, "target": target,
                                **binary_metrics(s[target].astype(int).tolist(), s.rupture_present.astype(int).tolist())})
            acc, macro, rows = class_metrics(s.patient_type_label.astype(str).tolist(), s.primary_label.astype(str).tolist())
            type_sum.append({"model": model, "condition": cond, "N": len(s), "type_accuracy": acc, "macro_f1_4class": macro})
            for r in rows: type_class.append({"model": model, "condition": cond, **r})

        qs = q[q.condition.eq(cond) & q.segment_idx.isin(common)][["segment_idx","primary_label","rupture_present"]].rename(
            columns={"primary_label":"qwen_label","rupture_present":"qwen_rupture"})
        ins = i[i.condition.eq(cond) & i.segment_idx.isin(common)][["segment_idx","primary_label","rupture_present"]].rename(
            columns={"primary_label":"internvl_label","rupture_present":"internvl_rupture"})
        sbs = l.merge(qs,on="segment_idx").merge(ins,on="segment_idx")
        sbs.insert(1,"condition",cond)
        sbs["models_agree_binary"] = sbs.qwen_rupture.eq(sbs.internvl_rupture)
        sbs["models_agree_type"] = sbs.qwen_label.eq(sbs.internvl_label)
        side.append(sbs)

    cov = pd.DataFrame(coverage)
    met = pd.DataFrame(metrics)
    ts = pd.DataFrame(type_sum)
    tc = pd.DataFrame(type_class)
    sbs = pd.concat(side, ignore_index=True) if side else pd.DataFrame()

    cov.to_csv(out/"coverage.csv", index=False, encoding="utf-8-sig")
    met.to_csv(out/"model_condition_binary_metrics_common.csv", index=False, encoding="utf-8-sig")
    ts.to_csv(out/"patient_type_summary_common.csv", index=False, encoding="utf-8-sig")
    tc.to_csv(out/"patient_type_metrics_by_class_common.csv", index=False, encoding="utf-8-sig")
    if not sbs.empty:
        sbs.sort_values(["condition","segment_idx"]).to_csv(out/"side_by_side_predictions_common.csv", index=False, encoding="utf-8-sig")
        sbs[(~sbs.models_agree_binary) | (~sbs.models_agree_type)].to_csv(out/"model_disagreements_common.csv", index=False, encoding="utf-8-sig")

    head = met[met.target.eq("legacy_any_3rs_marker")][["model","condition","N","TP","TN","FP","FN",
          "balanced_accuracy","precision","recall","specificity","f1"]].sort_values(["condition","f1"], ascending=[True,False])
    head.to_csv(out/"headline_model_comparison.csv", index=False, encoding="utf-8-sig")

    print("\nCOVERAGE")
    print(cov.to_string(index=False))
    print("\nFAIR COMMON-SET COMPARISON — original human_binary target")
    print(head.to_string(index=False) if not head.empty else "No common completed cases yet.")
    print(f"\nOutputs: {out}")
    if (cov.common_completed < a.expected_segments).any():
        print(f"\nNOTE: fewer than {a.expected_segments} common cases in at least one condition; results are preliminary.")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-predictions", default="./output/qwen3vl_v7_zeroshot_ablation/v7_condition_predictions.csv")
    p.add_argument("--internvl-predictions", default="./output/internvl3_v8_2_zeroshot_ablation/internvl3_condition_predictions.csv")
    p.add_argument("--labels", default="./output/visual_pilot_100/visual_pilot_100_labels.csv")
    p.add_argument("--output-dir", default="./output/vlm_zeroshot_model_comparison")
    p.add_argument("--expected-segments", type=int, default=100)
    return p

if __name__ == "__main__":
    main(parser().parse_args())