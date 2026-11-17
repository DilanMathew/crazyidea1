"""Compliance-only probe analysis for a second model (layer sweep + held-out families).
usage: python comply_small.py DATASET RUNDIR OUTJSON"""
import json, os, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from common import read_jsonl
DS, RUN, OUT = sys.argv[1:4]
rows = read_jsonl(os.path.join(RUN, "outputs_base.jsonl"))
last = np.load(os.path.join(RUN, "acts_last_base.npy"))
kind = np.array([r["kind"] for r in rows]); fam = np.array([r["family"] for r in rows])
y = np.array([r["complied"] for r in rows]).astype(int); inj = np.where(kind == "injected")[0]
probe = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
def cv(l):
    s = np.zeros(len(inj))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(inj, y[inj]):
        s[te] = probe().fit(last[inj[tr], l].astype(np.float32), y[inj[tr]]).predict_proba(last[inj[te], l].astype(np.float32))[:, 1]
    return roc_auc_score(y[inj], s)
sweep = [cv(l) for l in range(last.shape[1])]; bl = int(np.argmax(sweep))
lofo = {}
for f in ["ignore", "authority", "spoof", "ps", "roleplay", "fakechat", "spanish"]:
    tr = np.where((kind == "injected") & (fam != f) & (fam != "base64"))[0]; te = np.where(fam == f)[0]
    if y[te].sum() < 5: continue
    m = probe().fit(last[tr, bl].astype(np.float32), y[tr])
    lofo[f] = {"probe": float(roc_auc_score(y[te], m.predict_proba(last[te, bl].astype(np.float32))[:, 1])), "n_pos": int(y[te].sum())}
res = {"asr": float(y[inj].mean()), "n_complied": int(y[inj].sum()), "sweep": sweep, "best_layer": bl, "best_auc": sweep[bl], "lofo": lofo}
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "sweep"}, indent=1))
