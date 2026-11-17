"""Paired significance for steering configs vs 'none' on the same held-out prompts.
Exact McNemar test on per-prompt outcomes + paired bootstrap 95% CI of the difference.
usage: python stats.py STEER_JSON"""
import json, sys
import numpy as np
from math import comb

def mcnemar(a, b):
    a, b = np.array(a), np.array(b); n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10; k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return p, n01, n10

def boot(a, b, B=10000, seed=0):
    a, b = np.array(a, float), np.array(b, float); rs = np.random.RandomState(seed); n = len(a)
    idx = rs.randint(0, n, (B, n)); d = (b[idx] - a[idx]).mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

S = json.load(open(sys.argv[1])); base = S["none"]; out = {}
for k, v in S.items():
    if k == "none": continue
    p_asr, fixed, broke = mcnemar(base["complied_flags"], v["complied_flags"])
    p_acc, lost, gained = mcnemar(base["clean_flags"], v["clean_flags"])
    out[k] = {"asr": v["asr"], "d_asr": v["asr"] - base["asr"], "d_asr_ci": boot(base["complied_flags"], v["complied_flags"]),
              "p_asr": p_asr, "attacks_stopped": fixed, "attacks_new": broke,
              "task_clean": v["task_clean"], "d_clean": v["task_clean"] - base["task_clean"],
              "d_clean_ci": boot(base["clean_flags"], v["clean_flags"]), "p_clean": p_acc}
    print(f"{k:16s} ASR {v['asr']:.3f} (Δ {out[k]['d_asr']:+.3f}, CI {out[k]['d_asr_ci'][0]:+.3f}..{out[k]['d_asr_ci'][1]:+.3f}, "
          f"p={p_asr:.4f}, stopped {fixed} / new {broke})   clean {v['task_clean']:.3f} (Δ {out[k]['d_clean']:+.3f}, p={p_acc:.3f})")
json.dump(out, open(sys.argv[1].replace(".json", "_stats.json"), "w"), indent=1)
