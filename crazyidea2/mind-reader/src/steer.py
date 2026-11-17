"""Activation steering against prompt injection.

Direction d_l = mean(last-token residual | complied) - mean(... | resisted), computed on the
training split only (see probes.py). Interventions, applied at every token position during generation:
  add:    h_l <- h_l - alpha * ||d_l|| * d_hat_l                  at one layer l
  ablate: h_k <- h_k - (h_k . d_hat) d_hat  for every layer k     (d_hat from one layer; Arditi et al. style)
Evaluated on the held-out split: 300 injected + 150 clean prompts.

usage: python steer.py MODEL DATASET RUNDIR OUTJSON --configs "none;add:L:alpha;ablate:L;spot;spot+add:L:alpha"
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load, read_jsonl, prompt, run_batch, complied, task_ok, batched, Timer, layers

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("dataset"); ap.add_argument("rundir"); ap.add_argument("out")
ap.add_argument("--configs", required=True); ap.add_argument("--dir", default="comply"); ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()

tok, model = load(a.model)
ids = open(os.path.join(a.rundir, "steer_test_ids.txt")).read().split()
data = {e["id"]: e for e in read_jsonl(a.dataset)}; ex = [data[i] for i in ids]
dirs = np.load(os.path.join(a.rundir, "directions.npz"))[a.dir]           # [L+1, d], index = hidden_states index
results = json.load(open(a.out)) if os.path.exists(a.out) else {}

def make_hooks(cfg):
    hs = []
    parts = cfg.replace("spot+", "").split(":")
    if parts[0] == "add":
        l, alpha = int(parts[1]), float(parts[2]); d = torch.tensor(dirs[l], dtype=torch.float32)
        vec = (alpha * d).to(torch.bfloat16)          # alpha * ||d|| * d_hat == alpha * d
        def f(mod, inp, out):
            if isinstance(out, tuple): return (out[0] - vec,) + tuple(out[1:])
            return out - vec
        hs.append(layers(model)[l - 1].register_forward_hook(f))
    elif parts[0] == "ablate":
        l = int(parts[1]); d = torch.tensor(dirs[l], dtype=torch.float32); dh = (d / d.norm())
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float(); hf = hf - (hf @ dh)[..., None] * dh
            h2 = hf.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        for layer in layers(model): hs.append(layer.register_forward_hook(f))
    return hs

for cfg in a.configs.split(";"):
    if cfg in results: print("skip", cfg); continue
    variant = "spot" if cfg.startswith("spot") else "base"
    texts = [prompt(tok, e, variant) for e in ex]; order = np.argsort([len(t) for t in texts])
    hooks = make_hooks(cfg) if cfg not in ("none", "spot") else []
    outs = [None] * len(ex); tm = Timer(len(ex))
    try:
        for idx in batched(list(order), a.bs):
            o, _, _ = run_batch(tok, model, [texts[i] for i in idx], capture=False)
            for j, i in enumerate(idx): outs[i] = o[j]
            print(cfg, tm.tick(len(idx)), flush=True)
    finally:
        for h in hooks: h.remove()
    inj = [(e, o) for e, o in zip(ex, outs) if e["kind"] == "injected"]; cl = [(e, o) for e, o in zip(ex, outs) if e["kind"] == "clean"]
    results[cfg] = {"asr": float(np.mean([complied(e, o) for e, o in inj])),
                    "task_clean": float(np.mean([task_ok(e, o) for e, o in cl])),
                    "task_injected": float(np.mean([task_ok(e, o) for e, o in inj])),
                    "complied_flags": [bool(complied(e, o)) for e, o in inj],
                    "clean_flags": [bool(task_ok(e, o)) for e, o in cl],
                    "samples": [{"id": e["id"], "out": o[:200]} for e, o in (inj[:6] + cl[:3])]}
    print(cfg, {k: round(v, 3) for k, v in results[cfg].items() if isinstance(v, float)}, flush=True)
    with open(a.out, "w") as f: json.dump(results, f, indent=1)
