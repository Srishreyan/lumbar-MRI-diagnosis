#!/usr/bin/env python
# tta_analysis.py
# ------------------------------------------------------------------
# Re-analyse the TTA per-view logits (results/tta_views_<tag>.npz) under
# several *proper* uncertainty definitions, to see whether the anti-predictive
# TTA result (and by extension MC-Dropout's) is an artifact of measuring raw
# logit variance instead of predictive disagreement.
#
# Raw logit variance is a poor uncertainty proxy: a sample whose logits move
# 3 -> 6 across views has high variance yet is confidently correct. The
# disagreement-based measures below are what an epistemic-uncertainty metric
# should be. High score = uncertain, in every column.
#
# Columns (unc-AUC: 1.0 = uncertainty perfectly flags errors, 0.5 = no info):
#   logit_var  : variance of logits across views           (the baseline)
#   prob_var   : variance of sigmoid/softmax probs across views
#   flip_rate  : fraction of views that DISAGREE with the view-mean prediction
#   entropy    : entropy of the view-mean probabilities (peaks at 0.5 / uniform)
#   margin_inv : negative mean confidence margin (|logit| / max-minus-2nd-max)
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import json
import argparse

import numpy as np

from evaluate_baseline import BIN_COLS, roc_auc
from eval_uncertainty import selective_accuracy


def entropy_bin(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def entropy_cat(P):
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=pathlib.Path, default="results/tta_views_tta_evit_b1.npz")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="optional JSON to save the uncertainty report")
    args = ap.parse_args()

    d = np.load(args.npz)
    views = d["views"]          # [N, V, 12]
    y_bin = d["y_bin"]          # [N, 7]
    y_pf = d["y_pf"]            # [N]
    N, V = views.shape[:2]
    print(f"Loaded {args.npz}: {N} samples x {V} views x {views.shape[2]} logits")

    report = {"npz": str(args.npz), "n": N, "n_views": V, "binary_tasks": [], "pfirrmann": {}}

    measures = ["logit_var", "prob_var", "flip_rate", "entropy", "margin_inv"]
    hdr = f"{'task':18s}" + "".join(f"{m:>11s}" for m in measures)
    print(hdr + f"   [best column should be closest to 1.0]")
    print("-" * (18 + 11 * len(measures)))

    # Risk-coverage summary for the two best disagreement measures: accuracy
    # on the most-confident 50% / 90% of samples, using each uncertainty score.
    sel = {"flip_rate": [], "entropy": []}

    # --- Binary tasks ------------------------------------------------------
    for i, col in enumerate(BIN_COLS):
        z = views[:, :, i]                       # [N, V]
        p = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)
        pmean = 1 / (1 + np.exp(-z.mean(axis=1)))
        pred = pmean > 0.5
        err = (pred != y_bin[:, i]).astype(int)
        scores = {
            "logit_var": z.var(axis=1),
            "prob_var": p.var(axis=1),
            "flip_rate": (z > 0).astype(int).mean(axis=1),   # views disagreeing with mean-pred
            "entropy": entropy_bin(pmean),
            "margin_inv": -np.abs(z).mean(axis=1),
        }
        row = f"{col:18s}"
        for m in measures:
            row += f"{roc_auc(err, scores[m]):10.3f} "
        print(row)
        for m in sel:
            sel[m].append({
                "task": col,
                "sel50": selective_accuracy(err, scores[m], 0.50),
                "sel90": selective_accuracy(err, scores[m], 0.90),
            })
        report["binary_tasks"].append({
            "task": col, "accuracy": float(1.0 - err.mean()),
            "unc_auc_flip_rate": float(roc_auc(err, scores["flip_rate"])),
            "unc_auc_entropy": float(roc_auc(err, scores["entropy"])),
            "sel50_flip": float(selective_accuracy(err, scores["flip_rate"], 0.50)),
            "sel90_flip": float(selective_accuracy(err, scores["flip_rate"], 0.90)),
            "sel50_ent": float(selective_accuracy(err, scores["entropy"], 0.50)),
            "sel90_ent": float(selective_accuracy(err, scores["entropy"], 0.90)),
        })

    # --- Pfirrmann ---------------------------------------------------------
    lb = views[:, :, 7:]                         # [N, V, 5]
    sm = np.exp(lb - lb.max(axis=-1, keepdims=True))
    sm /= sm.sum(axis=-1, keepdims=True)
    sm_mean = sm.mean(axis=1)                    # [N, 5]
    pred = sm_mean.argmax(axis=1)
    err = (pred != y_pf).astype(int)
    lb_mean = lb.mean(axis=1)                    # [N, 5]
    order = np.sort(lb_mean, axis=1)
    scores = {
        "logit_var": lb.var(axis=1).mean(axis=1),
        "prob_var": sm.var(axis=1).mean(axis=1),
        "flip_rate": (lb.argmax(axis=-1) != pred[:, None]).mean(axis=1),
        "entropy": entropy_cat(sm_mean),
        "margin_inv": -(order[:, -1] - order[:, -2]),
    }
    row = f"{'Pfirrmann':18s}"
    for m in measures:
        row += f"{roc_auc(err, scores[m]):10.3f} "
    print(row)
    for m in sel:
        sel[m].append({"task": "Pfirrmann",
                       "sel50": selective_accuracy(err, scores[m], 0.50),
                       "sel90": selective_accuracy(err, scores[m], 0.90)})
    report["pfirrmann"] = {
        "accuracy": float(1.0 - err.mean()),
        "unc_auc_flip_rate": float(roc_auc(err, scores["flip_rate"])),
        "unc_auc_entropy": float(roc_auc(err, scores["entropy"])),
        "sel50_flip": float(selective_accuracy(err, scores["flip_rate"], 0.50)),
        "sel90_flip": float(selective_accuracy(err, scores["flip_rate"], 0.90)),
        "sel50_ent": float(selective_accuracy(err, scores["entropy"], 0.50)),
        "sel90_ent": float(selective_accuracy(err, scores["entropy"], 0.90)),
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved -> {args.out}")

    # --- Risk-coverage table ----------------------------------------------
    print(f"\n--- Selective accuracy (acc on most-confident X%, by uncertainty) ---")
    print(f"{'task':18s} {'flip@50':>8s} {'flip@90':>8s} {'ent@50':>8s} {'ent@90':>8s}")
    print("-" * 54)
    for r_bin, r_ent in zip(sel["flip_rate"], sel["entropy"]):
        assert r_bin["task"] == r_ent["task"]
        print(f"{r_bin['task']:18s} {r_bin['sel50']*100:7.1f}% {r_bin['sel90']*100:7.1f}% "
              f"{r_ent['sel50']*100:7.1f}% {r_ent['sel90']*100:7.1f}%")


if __name__ == "__main__":
    main()
