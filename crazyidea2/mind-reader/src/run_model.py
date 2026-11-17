"""Run a model over the dataset: greedy outputs + compliance labels + residual activations.

usage: python run_model.py MODEL DATASET OUTDIR [--variant base|spot] [--limit N] [--no-acts] [--kinds injected,clean]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import load, read_jsonl, prompt, run_batch, complied, task_ok, batched, Timer

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("dataset"); ap.add_argument("outdir")
ap.add_argument("--variant", default="base"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--no-acts", action="store_true"); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--ids", default="", help="file with one example id per line to restrict to")
a = ap.parse_args()

data = read_jsonl(a.dataset)
if a.ids: keep = set(open(a.ids).read().split()); data = [e for e in data if e["id"] in keep]
if a.limit: data = data[:a.limit]
os.makedirs(a.outdir, exist_ok=True)
tok, model = load(a.model)
texts = [prompt(tok, e, a.variant) for e in data]
order = np.argsort([len(t) for t in texts])          # length-sorted batches -> less padding
L = model.config.num_hidden_layers + 1; d = model.config.hidden_size
if not a.no_acts:
    last = np.zeros((len(data), L, d), np.float16); mean = np.zeros((len(data), L, d), np.float16)
outs = [None] * len(data); tm = Timer(len(data))
for idx in batched(list(order), a.bs):
    o, l, m = run_batch(tok, model, [texts[i] for i in idx], capture=not a.no_acts)
    for j, i in enumerate(idx):
        outs[i] = o[j]
        if not a.no_acts: last[i] = l[j]; mean[i] = m[j]
    print(tm.tick(len(idx)), flush=True)

with open(os.path.join(a.outdir, f"outputs_{a.variant}.jsonl"), "w") as f:
    for e, o in zip(data, outs):
        f.write(json.dumps({"id": e["id"], "kind": e["kind"], "family": e["family"], "goal": e["goal"],
                            "position": e["position"], "task": e["task"], "output": o,
                            "complied": complied(e, o), "task_ok": task_ok(e, o)}) + "\n")
if not a.no_acts:
    np.save(os.path.join(a.outdir, f"acts_last_{a.variant}.npy"), last)
    np.save(os.path.join(a.outdir, f"acts_mean_{a.variant}.npy"), mean)
print("done")
