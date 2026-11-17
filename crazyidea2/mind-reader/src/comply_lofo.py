"""Compliance prediction on attack families never seen in training (leave-one-family-out).
Metadata can't help on an unseen family, so the fair baseline is TF-IDF on the email text.
The layer is fixed in advance at the CV-best layer from probes.py (stated in the write-up).
usage: python comply_lofo.py DATASET RUNDIR PROBES_JSON"""
import json, os, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from common import read_jsonl

DS, RUN, PJ = sys.argv[1:4]
data = {e["id"]: e for e in read_jsonl(DS)}
rows = read_jsonl(os.path.join(RUN, "outputs_base.jsonl"))
P = json.load(open(PJ)); bl = P["comply_best_layer"]
last = np.load(os.path.join(RUN, "acts_last_base.npy"), mmap_mode="r")[:, bl].astype(np.float32)
kind = np.array([r["kind"] for r in rows]); fam = np.array([r["family"] for r in rows])
y = np.array([r["complied"] for r in rows]).astype(int)
docs = np.array([data[r["id"]]["doc"] for r in rows], dtype=object)
probe = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
tfidf = lambda: make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                         TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
                              LogisticRegression(C=4, max_iter=3000))
out = {}
for f in ["ignore", "authority", "spoof", "ps", "roleplay", "fakechat", "spanish"]:
    tr = np.where((kind == "injected") & (fam != f) & (fam != "base64"))[0]; te = np.where(fam == f)[0]
    a_p = roc_auc_score(y[te], probe().fit(last[tr], y[tr]).predict_proba(last[te])[:, 1])
    a_t = roc_auc_score(y[te], tfidf().fit(docs[tr], y[tr]).predict_proba(docs[te])[:, 1])
    out[f] = {"probe": float(a_p), "tfidf": float(a_t), "n_pos": int(y[te].sum()), "n": int(len(te))}
    print(f, out[f], flush=True)
P["comply_lofo"] = out; P["comply_lofo_layer"] = bl
json.dump(P, open(PJ, "w"), indent=1)
