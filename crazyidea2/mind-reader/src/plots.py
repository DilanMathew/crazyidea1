"""Figures for the write-up. usage: python plots.py RESULTS_DIR"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1]
S = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e2dc",
     "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s7": "#6250d6", "muted": "#a3a19a"}
plt.rcParams.update({"figure.facecolor": S["surface"], "axes.facecolor": S["surface"], "axes.edgecolor": S["grid"],
                     "axes.labelcolor": S["ink2"], "xtick.color": S["ink2"], "ytick.color": S["ink2"], "text.color": S["ink"],
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": S["grid"], "grid.linewidth": .8, "axes.titleweight": "bold", "axes.titlesize": 13,
                     "axes.titlelocation": "left", "legend.frameon": False})
P = json.load(open(os.path.join(R, "probes_1.5B.json")))
FAMS = ["ignore", "authority", "spoof", "ps", "roleplay", "fakechat", "spanish", "base64"]

def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(R, name), dpi=200); plt.close(fig)

# 1. layer sweep (detection + compliance)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, key, base, title in [(ax[0], "detect_sweep", "detect_baselines_cv", "Is there an injection? (detection)"),
                            (ax[1], "comply_sweep", "comply_baselines_cv", "Will the model obey it? (compliance)")]:
    sw = P[key]; x = np.arange(len(sw["last"]))
    a.plot(x, sw["mean"], color=S["s1"], lw=2, label="probe: mean over prompt")
    a.plot(x, sw["last"], color=S["s7"], lw=2, label="probe: last prompt token")
    b = P[base]
    a.axhline(b["tfidf"], color=S["s2"], lw=1.5, ls="--", label=f"TF-IDF text classifier ({b['tfidf']:.2f})")
    if "metadata" in b: a.axhline(b["metadata"], color=S["s4"], lw=1.5, ls=":", label=f"attack metadata ({b['metadata']:.2f})")
    if "keyword" in b: a.axhline(b["keyword"], color=S["muted"], lw=1.5, ls=":", label=f"keyword rule ({b['keyword']:.2f})")
    a.set_title(title); a.set_xlabel("layer (0 = token embeddings)"); a.set_ylim(0.45, 1.01); a.legend(fontsize=8.5, loc="lower right")
ax[0].set_ylabel("AUROC, 5-fold CV")
save(fig, "fig1_layer_sweep.png")

# 2. LOFO detection: probe vs tfidf vs keyword
lofo = P["detect_lofo"]
fig, ax = plt.subplots(figsize=(11, 4.2)); w = .26; x = np.arange(len(FAMS))
for k, (name, col, lab) in enumerate([("probe_mean", S["s1"], "activation probe"), ("tfidf", S["s2"], "TF-IDF on text"), ("keyword", S["muted"], "keyword rule")]):
    v = [lofo[f][name]["auroc"] for f in FAMS]
    ax.bar(x + (k - 1) * w, v, w * .92, color=col, label=lab)
ax.set_xticks(x, FAMS); ax.set_ylim(0.4, 1.02); ax.set_ylabel("AUROC on unseen family")
ax.set_title("Held-out attack styles: train on 7 framings, test on the 8th"); ax.legend(ncol=3, loc="lower left", fontsize=9)
save(fig, "fig2_lofo.png")

# 3. hard negatives FPR
fig, ax = plt.subplots(figsize=(7.5, 3.8))
cats = [("fpr_aware", "security-awareness emails\n(quote 'ignore instructions')"), ("fpr_benign", "benign imperatives\n('forward to legal@…')"), ("fpr_clean", "clean emails")]
for k, (name, col, lab) in enumerate([("probe_mean", S["s1"], "activation probe"), ("tfidf", S["s2"], "TF-IDF on text")]):
    v = [np.mean([lofo[f][name][c] for f in FAMS]) for c, _ in cats]
    ax.bar(np.arange(3) + (k - .5) * .38, v, .35, color=col, label=lab)
    for i, vv in enumerate(v): ax.text(i + (k - .5) * .38, vv + .01, f"{vv:.0%}", ha="center", fontsize=9, color=S["ink2"])
ax.set_xticks(np.arange(3), [c[1] for c in cats], fontsize=9); ax.set_ylabel("false-positive rate")
ax.set_title("False alarms at 95% recall (mean over held-out folds)"); ax.legend()
save(fig, "fig3_false_alarms.png")

# 4. steering
sp = os.path.join(R, "steer_1.5B.json")
if os.path.exists(sp):
    st = json.load(open(sp))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for k, v in st.items():
        col = S["muted"] if k == "none" else S["s2"] if k.startswith("spot") and k == "spot" else S["s3"] if k.startswith("spot+") else S["s1"]
        ax.scatter(v["task_clean"], v["asr"], s=60, color=col, zorder=3, edgecolor=S["surface"], linewidth=1.5)
        ax.annotate(k, (v["task_clean"], v["asr"]), textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=S["ink2"])
    ax.set_xlabel("task accuracy on clean emails  →  better"); ax.set_ylabel("attack success rate  ↓  better")
    ax.set_title("Steering the model's internals vs a prompt defense")
    save(fig, "fig4_steering.png")
print("ok")
