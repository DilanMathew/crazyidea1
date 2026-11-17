"""Steering hooks (same maths as steer.py), importable.
cfg: 'none' | 'add:L:alpha' | 'ablate:L'   (L = hidden_states index; 0 = embeddings)"""
import torch
from common import layers

def make(model, dirs, cfg):
    hs = []; parts = cfg.replace("spot+", "").split(":")
    if parts[0] == "add":
        l, alpha = int(parts[1]), float(parts[2]); vec = (alpha * torch.tensor(dirs[l], dtype=torch.float32)).to(torch.bfloat16)
        def f(mod, inp, out):
            return (out[0] - vec,) + tuple(out[1:]) if isinstance(out, tuple) else out - vec
        hs.append(layers(model)[l - 1].register_forward_hook(f))
    elif parts[0] == "ablate":
        d = torch.tensor(dirs[int(parts[1])], dtype=torch.float32); dh = d / d.norm()
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float(); h2 = (hf - (hf @ dh)[..., None] * dh).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        for layer in layers(model): hs.append(layer.register_forward_hook(f))
    return hs
