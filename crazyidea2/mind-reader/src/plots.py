"""Figures for the write-up. usage: python plots.py RESULTS_DIR"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

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

# 2. compliance on unseen attack styles (LOFO): probe vs TF-IDF
cl = P["comply_lofo"]; fams = list(cl)
fig, ax = plt.subplots(figsize=(11, 4.2)); w = .38; x = np.arange(len(fams))
for k, (name, col, lab) in enumerate([("probe", S["s1"], f"activation probe (layer {P['comply_lofo_layer']}, last prompt token)"), ("tfidf", S["s2"], "TF-IDF on the email text")]):
    v = [cl[f][name] for f in fams]
    ax.bar(x + (k - .5) * w, v, w * .92, color=col, label=lab)
    for i, vv in enumerate(v): ax.text(i + (k - .5) * w, vv + .008, f"{vv:.2f}", ha="center", fontsize=8.5, color=S["ink2"])
ax.set_xticks(x, [f"{f}\n({cl[f]['n_pos']}/{cl[f]['n']} obeyed)" for f in fams], fontsize=9); ax.set_ylim(0.5, 1.04)
ax.set_ylabel("AUROC: will it obey?"); ax.set_title("Predicting obedience on attack styles never seen in training")
ax.legend(loc="lower left", fontsize=9)
save(fig, "fig2_comply_unseen.png")

# 3. threshold transfer for detection on unseen families
lofo = P["detect_lofo"]
fig, ax = plt.subplots(figsize=(11, 3.8)); x = np.arange(len(FAMS))
for k, (name, col, lab) in enumerate([("probe_last", S["s1"], "activation probe"), ("tfidf", S["s2"], "TF-IDF on text")]):
    v = [lofo[f][name]["recall_at_thr"] for f in FAMS]
    ax.bar(x + (k - .5) * .38, v, .35, color=col, label=f"{lab} (mean {np.mean(v):.0%})")
ax.set_xticks(x, FAMS); ax.set_ylim(0, 1.05); ax.set_ylabel("share of new attacks caught")
ax.set_title("Detection threshold set on 7 known styles, applied to an unseen 8th (0 false alarms for both)")
ax.legend(loc="upper left", fontsize=9)
save(fig, "fig3_threshold_transfer.png")

# 4. steering: three metrics per config (small multiples, same % scale)
sp = os.path.join(R, "steer_1.5B.json"); lp = os.path.join(R, "legit_1.5B.json"); rp = os.path.join(R, "steer_random_1.5B.json")
if os.path.exists(sp) and os.path.exists(lp):
    st = json.load(open(sp)); lg = json.load(open(lp)); rd = json.load(open(rp)) if os.path.exists(rp) else {}
    rows = [("none", "no defense"), ("spot", "spotlight prompt"), ("add:16:1", "steer @16, a=1"), ("add:23:1", "steer @23, a=1"),
            ("add:23:2", "steer @23, a=2"), ("ablate:23", "ablate @23 dir"), ("ablate:16", "ablate @16 dir")]
    RL = {"add:23:1": "steer @23, a=1", "add:23:2": "steer @23, a=2", "ablate:23": "ablate @23"}
    rows += [("rand:" + k, f"control: RANDOM dir, {RL.get(k, k)}") for k in rd if k != "none"]
    def get(k, metric):
        if k.startswith("rand:"): src = rd[k[5:]]
        else: src = st[k]
        if metric == "legit": return lg.get(k, {}).get("legit_tool_rate", np.nan)
        return src[metric]
    fig, ax = plt.subplots(1, 3, figsize=(12, 0.42 * len(rows) + 1.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for a, (metric, title, col) in zip(ax, [("asr", "attack success  (lower = better)", S["s2"]), ("task_clean", "task accuracy, clean emails", S["s1"]), ("legit", "legit tool calls still work", S["s3"])]):
        v = [get(k, metric) for k, _ in rows]
        a.barh(y, v, .62, color=[S["muted"] if k == "none" else col for k, _ in rows])
        for yy, vv in zip(y, v):
            if not np.isnan(vv): a.text(vv + .01, yy, f"{vv:.0%}", va="center", fontsize=8.5, color=S["ink2"])
            else: a.text(.01, yy, "not run", va="center", fontsize=8.5, color=S["muted"])
        base = get("none", metric); a.axvline(base, color=S["ink2"], lw=1, ls=":")
        a.set_title(title, fontsize=11); a.set_xlim(0, 1.12 if metric != "asr" else max(v) * 1.35); a.grid(axis="y", visible=False)
        a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[0].set_yticks(y, [lab for _, lab in rows])
    fig.suptitle("Steering Qwen2.5-1.5B's residual stream: 300 held-out attacks, 150 clean emails, 150 legit tool requests", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "fig4_steering.png")
print("ok")
