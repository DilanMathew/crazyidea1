"""Linear probes on residual-stream activations, with honest baselines.

Q1 DETECTION   Is there an injected instruction in the context?          (all 2,600 prompts)
Q2 COMPLIANCE  Will the model obey it, judged *before* it writes a token? (injected prompts only)

Protocols
  * layer sweep, stratified 5-fold CV AUROC
  * leave-one-family-out (LOFO): train on 7 framings, test on the unseen 8th.
    Layer is chosen by inner CV on the training families only (no peeking).
  * hard negatives: false-positive rate on 'aware' (emails that quote injection phrases)
    and 'benign' (imperatives to humans) at a threshold set for 95% recall on training data.
Baselines (same splits): keyword rule, TF-IDF logistic regression on the email text,
token-embedding probe (layer 0, mean-pooled = bag of tokens), metadata (family/goal/position/task).

usage: python probes.py DATASET RUNDIR OUTJSON
"""
import json, os, re, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from common import read_jsonl

DS, RUN, OUT = sys.argv[1:4]
data = {e["id"]: e for e in read_jsonl(DS)}
rows = read_jsonl(os.path.join(RUN, "outputs_base.jsonl"))
ex = [data[r["id"]] for r in rows]
last = np.load(os.path.join(RUN, "acts_last_base.npy")).astype(np.float32)
mean = np.load(os.path.join(RUN, "acts_mean_base.npy")).astype(np.float32)
N, L, D = last.shape
kind = np.array([r["kind"] for r in rows]); fam = np.array([r["family"] for r in rows])
y_inj = (kind == "injected").astype(int)
y_comp = np.array([r["complied"] for r in rows]).astype(int)
docs = [e["doc"] for e in ex]
FAMS = [f for f in ["ignore", "authority", "spoof", "ps", "roleplay", "fakechat", "spanish", "base64"]]
rng = np.random.RandomState(0)

def probe(C=0.05): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
def tfidf(): return make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                             TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
                                  LogisticRegression(C=4, max_iter=3000))
KW = re.compile(r"ignore|instruction|assistant|\bai\b|system|disregard|forget|override|developer mode|pretend|base64|decode|"
                r"ignora|instrucciones|asistente|olvida|admin|authori[sz]ed", re.I)
def kw_score(texts): return np.array([len(KW.findall(t)) for t in texts], float)

def cv_auc(X, y, idx, mk=probe, k=5):
    s = np.zeros(len(idx)); skf = StratifiedKFold(k, shuffle=True, random_state=0)
    for tr, te in skf.split(idx, y[idx]):
        m = mk().fit(X[idx[tr]], y[idx[tr]]); s[te] = m.predict_proba(X[idx[te]])[:, 1]
    return roc_auc_score(y[idx], s)

res = {"model_dims": [N, L, D]}
allidx = np.arange(N)

# ---------------- behaviour ----------------
inj = np.where(y_inj == 1)[0]
res["behaviour"] = {
    "asr_overall": float(y_comp[inj].mean()),
    "asr_by_family": {f: float(y_comp[(fam == f)].mean()) for f in FAMS},
    "asr_by_goal": {g: float(np.mean([r["complied"] for r in rows if r["goal"] == g])) for g in ["canary", "email", "fetch", "promo"]},
    "asr_by_position": {p: float(np.mean([r["complied"] for r in rows if r["kind"] == "injected" and r["position"] == p])) for p in ["start", "middle", "end"]},
    "task_acc": {k: float(np.mean([r["task_ok"] for r in rows if r["kind"] == k])) for k in ["clean", "benign", "aware", "injected"]},
    "n": {k: int((kind == k).sum()) for k in ["injected", "clean", "benign", "aware"]},
    "n_complied": int(y_comp.sum()),
}
spot_path = os.path.join(RUN, "outputs_spot.jsonl")
if os.path.exists(spot_path):
    sp = read_jsonl(spot_path)
    res["behaviour"]["spotlight"] = {
        "asr_overall": float(np.mean([r["complied"] for r in sp if r["kind"] == "injected"])),
        "asr_by_family": {f: float(np.mean([r["complied"] for r in sp if r["family"] == f])) for f in FAMS},
        "task_acc_clean": float(np.mean([r["task_ok"] for r in sp if r["kind"] == "clean"])),
        "task_acc_injected": float(np.mean([r["task_ok"] for r in sp if r["kind"] == "injected"]))}
print(json.dumps(res["behaviour"], indent=1), flush=True)

# ---------------- Q1 detection: layer sweep ----------------
sweep_layers = list(range(L))
res["detect_sweep"] = {"last": [cv_auc(last[:, l], y_inj, allidx) for l in sweep_layers],
                       "mean": [cv_auc(mean[:, l], y_inj, allidx) for l in sweep_layers]}
res["detect_baselines_cv"] = {
    "keyword": float(roc_auc_score(y_inj, kw_score(docs))),
    "tfidf": cv_auc(np.array(docs, dtype=object), y_inj, allidx, mk=tfidf),
    "embed_bag": cv_auc(mean[:, 0], y_inj, allidx)}
print("detect sweep done", flush=True)

# ---------------- Q1 detection: leave-one-family-out ----------------
neg = np.where(y_inj == 0)[0]; neg_perm = rng.permutation(neg)
neg_tr, neg_te = neg_perm[: len(neg) // 2], neg_perm[len(neg) // 2:]
cand = list(range(2, L, 2))
def pick_layer(feats, tr):
    best, bl = -1, None
    for l in cand:
        a = cv_auc(feats[:, l], y_inj, tr, k=3)
        if a > best: best, bl = a, l
    return bl
def fpr_at_recall(score_tr, y_tr, score_te, mask_te, recall=0.95):
    thr = np.quantile(score_tr[y_tr == 1], 1 - recall)
    return float((score_te[mask_te] >= thr).mean())

lofo = {}
for f in FAMS:
    tr = np.concatenate([np.where((y_inj == 1) & (fam != f))[0], neg_tr]); te = np.concatenate([np.where(fam == f)[0], neg_te])
    out = {}
    for name, feats in [("probe_mean", mean), ("probe_last", last)]:
        l = pick_layer(feats, tr); m = probe().fit(feats[tr, l], y_inj[tr])
        s_te = m.predict_proba(feats[te, l])[:, 1]; s_tr = m.predict_proba(feats[tr, l])[:, 1]
        out[name] = {"layer": l, "auroc": float(roc_auc_score(y_inj[te], s_te)),
                     "recall_at_thr": fpr_at_recall(s_tr, y_inj[tr], s_te, y_inj[te] == 1),
                     "fpr_aware": fpr_at_recall(s_tr, y_inj[tr], s_te, kind[te] == "aware"),
                     "fpr_benign": fpr_at_recall(s_tr, y_inj[tr], s_te, kind[te] == "benign"),
                     "fpr_clean": fpr_at_recall(s_tr, y_inj[tr], s_te, kind[te] == "clean")}
    D_ = np.array(docs, dtype=object); m = tfidf().fit(D_[tr], y_inj[tr])
    s_te = m.predict_proba(D_[te])[:, 1]; s_tr = m.predict_proba(D_[tr])[:, 1]
    out["tfidf"] = {"auroc": float(roc_auc_score(y_inj[te], s_te)),
                    "recall_at_thr": fpr_at_recall(s_tr, y_inj[tr], s_te, y_inj[te] == 1),
                    "fpr_aware": fpr_at_recall(s_tr, y_inj[tr], s_te, kind[te] == "aware"),
                    "fpr_benign": fpr_at_recall(s_tr, y_inj[tr], s_te, kind[te] == "benign"),
                    "fpr_clean": fpr_at_recall(s_tr, y_inj[tr], s_te, kind[te] == "clean")}
    k_te = kw_score(D_[te]); k_tr = kw_score(D_[tr])
    out["keyword"] = {"auroc": float(roc_auc_score(y_inj[te], k_te)),
                      "recall_at_thr": fpr_at_recall(k_tr, y_inj[tr], k_te, y_inj[te] == 1),
                      "fpr_aware": fpr_at_recall(k_tr, y_inj[tr], k_te, kind[te] == "aware")}
    m = probe().fit(mean[tr, 0], y_inj[tr]); out["embed_bag"] = {"auroc": float(roc_auc_score(y_inj[te], m.predict_proba(mean[te, 0])[:, 1]))}
    lofo[f] = out; print("LOFO", f, {k: round(v["auroc"], 3) for k, v in out.items()}, flush=True)
res["detect_lofo"] = lofo

# ---------------- Q2 compliance (injected only) ----------------
inj_idx = np.where(y_inj == 1)[0]
res["comply_sweep"] = {"last": [cv_auc(last[:, l], y_comp, inj_idx) for l in sweep_layers],
                       "mean": [cv_auc(mean[:, l], y_comp, inj_idx) for l in sweep_layers]}
meta = OneHotEncoder(handle_unknown="ignore").fit_transform(
    np.array([[r["family"], r["goal"], r["position"], r["task"]] for r in rows])).toarray()
def meta_m(): return LogisticRegression(C=1, max_iter=2000)
res["comply_baselines_cv"] = {"metadata": cv_auc(meta, y_comp, inj_idx, mk=meta_m),
                              "tfidf": cv_auc(np.array(docs, dtype=object), y_comp, inj_idx, mk=tfidf),
                              "embed_bag": cv_auc(mean[:, 0], y_comp, inj_idx)}
bl = int(np.argmax(res["comply_sweep"]["last"]))
res["comply_best_layer"] = bl
both = np.hstack([last[:, bl], meta * 10])
res["comply_baselines_cv"]["probe_plus_metadata"] = cv_auc(both, y_comp, inj_idx)
# within-goal AUROC (does the probe know more than "which kind of attack is this"?)
res["comply_within_goal"] = {}
for g in ["canary", "email", "fetch", "promo"]:
    gi = np.array([i for i in inj_idx if rows[i]["goal"] == g])
    if y_comp[gi].sum() >= 15 and (1 - y_comp[gi]).sum() >= 15:
        res["comply_within_goal"][g] = {"probe": cv_auc(last[:, bl], y_comp, gi),
                                        "metadata": cv_auc(meta, y_comp, gi, mk=meta_m), "n_pos": int(y_comp[gi].sum())}
print("compliance", res["comply_baselines_cv"], "best layer", bl, res["comply_within_goal"], flush=True)

# ---------------- directions for steering ----------------
rs = np.random.RandomState(1); perm = rs.permutation(inj_idx); steer_test = perm[:300]; steer_train = perm[300:]
c_tr = steer_train[y_comp[steer_train] == 1]; r_tr = steer_train[y_comp[steer_train] == 0]
clean_all = np.where(kind == "clean")[0]; clean_perm = rs.permutation(clean_all)
clean_test, clean_train = clean_perm[:150], clean_perm[150:]
dirs = {"comply": np.stack([last[c_tr, l].mean(0) - last[r_tr, l].mean(0) for l in range(L)]),
        "inject": np.stack([mean[steer_train, l].mean(0) - mean[clean_train, l].mean(0) for l in range(L)])}
np.savez(os.path.join(RUN, "directions.npz"), **dirs)
with open(os.path.join(RUN, "steer_test_ids.txt"), "w") as f:
    f.write("\n".join([rows[i]["id"] for i in steer_test] + [rows[i]["id"] for i in clean_test]))
res["steer_split"] = {"train_complied": int(len(c_tr)), "train_resisted": int(len(r_tr)), "test_injected": 300, "test_clean": 150}
with open(OUT, "w") as f: json.dump(res, f, indent=1)
print("saved", OUT)
