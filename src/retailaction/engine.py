"""Shared train/eval plumbing: run model over a loader, format preds/GTs, score them.

combine_ideas_v3 additions:
- Rank modes: `cls`, `quality`, `product`, `obj`, `obj*q`, `obj*q*cls`, `iou`, `obj*iou`, `obj*q*iou`, `hybrid`
- Passes `hand_tracks` to `model(vids, hand_tracks=...)` when present in `tgt`
- Supports symmetric temporal boundary padding `pad_t` (`start -= pad_t, end += pad_t`, clamped to `[0, 1]`)
- Linear score averaging in `collect_fused` + multi-model list support + multi-scale TTA support.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import evaluate

RANK_MODES = (
    "cls",
    "quality",
    "product",
    "obj",
    "obj*q",
    "obj*q*cls",
    "iou",
    "obj*iou",
    "obj*q*iou",
    "hybrid",
)


def rank_score(out, mode="cls"):
    """-> (B,Q) numpy-ready torch score tensor from a forward dict."""
    logits = out["logits"].float()
    cls = logits.softmax(-1)[..., :-1].max(-1).values
    q = out["quality"].float().sigmoid()
    obj = out["obj"].float().sigmoid() if "obj" in out else None
    iou = out["iou"].float().sigmoid() if ("iou" in out and out["iou"] is not None) else None
    if mode == "cls":
        return cls
    if mode == "quality":
        return q
    if mode == "product":
        return cls * q
    if mode == "obj":
        return obj if obj is not None else cls
    if mode == "obj*q":
        return (obj * q) if obj is not None else cls * q
    if mode == "obj*q*cls":
        return (obj * q * cls) if obj is not None else cls * q
    if mode == "iou":
        return iou if iou is not None else q
    if mode == "obj*iou":
        base_o = obj if obj is not None else cls
        base_i = iou if iou is not None else q
        return (base_o ** 0.35) * (base_i ** 0.65)
    if mode == "obj*q*iou":
        base_o = obj if obj is not None else cls
        base_i = iou if iou is not None else q
        return base_o * q * base_i
    if mode == "hybrid":
        base_o = obj if obj is not None else cls
        base_i = iou if iou is not None else q
        return ((base_o * q) ** 0.5) * (base_i ** 0.5)
    raise ValueError(f"unknown rank mode {mode}")


def _pad_temporal(tm, pad_t=0.0):
    if pad_t == 0.0:
        return tm
    tm = tm.copy()
    tm[..., 0] = np.clip(tm[..., 0] - pad_t, 0.0, 1.0)
    tm[..., 1] = np.clip(tm[..., 1] + pad_t, 0.0, 1.0)
    tm[..., 1] = np.maximum(tm[..., 1], tm[..., 0])
    return tm


@torch.no_grad()
def collect(model, loader, device, amp=True, max_batches=None, rank="cls", pad_t=0.0):
    """-> (preds, gts): dict sid -> {temporal, spatial, scores|factors}"""
    model.eval()
    preds, gts = {}, {}
    for bi, (vids, tgt) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        B = vids.shape[0]
        ht = torch.stack(tgt["hand_tracks"]).to(device, non_blocking=True) if "hand_tracks" in tgt else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(vids.to(device, non_blocking=True), hand_tracks=ht)
        sc = rank_score(out, rank)
        sp = out["spatial"].float().cpu().numpy()
        tm = _pad_temporal(out["temporal"].float().cpu().numpy(), pad_t=pad_t)
        sc = sc.cpu().numpy()
        for b in range(B):
            sid = tgt["id"][b]
            preds[sid] = {"temporal": tm[b], "spatial": sp[b], "scores": sc[b]}
            gts[sid] = {
                "temporal": tgt["temporal"][b].numpy(),
                "spatial": tgt["spatial"][b].numpy(),
                "factors": tgt["factors"][b].numpy(),
            }
    return preds, gts


def score(model, loader, device, amp=True, max_batches=None, rank="cls",
          merge=None, dedup="nms", pad_t=0.0):
    """score() with decode-time dedup. merge=None -> v2 behaviour (no NMS)."""
    t0 = time.time()
    preds, gts = collect(model, loader, device, amp, max_batches, rank=rank, pad_t=pad_t)
    if merge:
        from .postprocess import merge_duplicates
        preds = merge_duplicates(preds, gts, **merge)
    res = evaluate(preds, gts)
    res["seconds"] = round(time.time() - t0, 1)
    return res


FUSE_PASSES = ("s0", "s1", "flip", "flip1")


def _resize_vids(vids, sz):
    """Resize (B,2,C,T,H,W) video batch on GPU to (B,2,C,T,sz,sz) if sz != H."""
    if sz is None or vids.shape[-1] == sz:
        return vids
    B, V, C, T, H, W = vids.shape
    x = vids.view(B * V * T, C, H, W)
    x = F.interpolate(x, size=(sz, sz), mode="bilinear", align_corners=False)
    return x.view(B, V, C, T, sz, sz)


def _flip_hand_tracks(ht):
    """Mirror horizontal x coordinate and velocity of hand_tracks (B,2,2,T,6)."""
    if ht is None:
        return None
    ht_f = ht.clone()
    ht_f[..., 0] = 1.0 - ht_f[..., 0]
    ht_f[..., 2] = -ht_f[..., 2]
    return ht_f


@torch.no_grad()
def collect_fused(model, dl0, dl1, device, amp=True, passes=("s0", "s1", "flip"),
                  max_batches=None, rank="obj*q", scales=(None,), pad_t=0.0):
    """Slot-aligned multi-pass / multi-model / multi-scale fusion.

    `model` can be a single `nn.Module` or a list/tuple of `nn.Module`s.
    `passes` is a tuple of subset of ("s0", "s1", "flip", "flip1").
    `scales` is a tuple of spatial resolutions (e.g. `(224,)` or `(224, 288)`).
    """
    from .fuse import fuse_sample
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    for m in models:
        m.eval()
    preds, gts = {}, {}
    for bi, ((v0, t0), (v1, t1)) in enumerate(zip(dl0, dl1)):
        if max_batches and bi >= max_batches:
            break
        assert list(t0["id"]) == list(t1["id"]), "slot alignment: loaders must share id order"
        B = v0.shape[0]
        per = [[] for _ in range(B)]
        v0_dev = v0.to(device, non_blocking=True)
        v1_dev = v1.to(device, non_blocking=True) if any(p.endswith("1") for p in passes) else None
        ht0 = torch.stack(t0["hand_tracks"]).to(device, non_blocking=True) if "hand_tracks" in t0 else None
        ht0_flip = _flip_hand_tracks(ht0) if any(p.startswith("flip") for p in passes) else None
        for m in models:
            for sz in scales:
                for tag in passes:
                    vids = v1_dev if tag.endswith("1") else v0_dev
                    ht = ht0_flip if tag.startswith("flip") else ht0
                    target_sz = sz if sz is not None else getattr(m, "native_size", None)
                    if target_sz is not None:
                        vids = _resize_vids(vids, target_sz)
                    if tag.startswith("flip"):
                        vids = vids.flip(-1)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        out = m(vids, hand_tracks=ht)
                    sp = out["spatial"].float().cpu().numpy()
                    if tag.startswith("flip"):
                        sp = sp.copy()
                        sp[..., 0::2] = 1.0 - sp[..., 0::2]
                    tm = out["temporal"].float().cpu().numpy()
                    sc = rank_score(out, rank).float().cpu().numpy()
                    for b in range(B):
                        per[b].append({"temporal": tm[b], "spatial": sp[b], "score": sc[b]})
        for b in range(B):
            sid = t0["id"][b]
            ft, fs, fsc = fuse_sample(per[b])
            ft = _pad_temporal(ft, pad_t=pad_t)
            preds[sid] = {"temporal": ft, "spatial": fs, "scores": fsc}
            gts[sid] = {"temporal": t0["temporal"][b].numpy(),
                        "spatial": t0["spatial"][b].numpy(),
                        "factors": t0["factors"][b].numpy()}
    return preds, gts


def dedup_preds(preds, gts, dedup="nms", merge=None):
    from .fuse import drop_shadow_dups
    if dedup == "shadow":
        out = {}
        for sid, p in preds.items():
            t, s, c = drop_shadow_dups(p["temporal"], p["spatial"], p["scores"], gts[sid]["factors"])
            out[sid] = {"temporal": t, "spatial": s, "scores": c}
        return out
    from .postprocess import merge_duplicates
    return merge_duplicates(preds, gts, **merge) if merge else preds


def score_fused(model, dl0, dl1, device, amp=True, passes=("s0", "s1", "flip"),
                max_batches=None, merge=None, dedup="nms", rank="obj*q", scales=(None,), pad_t=0.0):
    t0 = time.time()
    preds, gts = collect_fused(model, dl0, dl1, device, amp, passes, max_batches,
                                rank=rank, scales=scales, pad_t=pad_t)
    preds = dedup_preds(preds, gts, dedup, merge)
    res = evaluate(preds, gts)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def save_ckpt(path, model, optimizer, epoch, best, args, ema=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict() if optimizer else None,
                "epoch": epoch, "best": best, "args": vars(args) if args else {},
                **({"model_ema": (ema.state_dict() if hasattr(ema, "state_dict") else
                                  ema.module.state_dict())} if ema is not None else {})}, path)


def load_weights(model, state_dict):
    """Load the matching subset (lets a v1 checkpoint seed a model with new heads)."""
    own = model.state_dict()
    keep = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in keep]
    model.load_state_dict(keep, strict=False)
    return len(keep), missing


def load_ckpt(path, model, optimizer=None, use_ema_init=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_ema"] if (use_ema_init and "model_ema" in ck) else ck["model"]
    n, missing = load_weights(model, sd)
    src_tag = "model_ema" if (use_ema_init and "model_ema" in ck) else "model"
    print(f"loaded {n}/{len(sd)} tensors ({src_tag}) from {path}" + (f" (fresh: {missing})" if missing else ""))
    if optimizer and ck.get("opt"):
        optimizer.load_state_dict(ck["opt"])
    return ck.get("epoch", 0), ck.get("best", 0.0), ck.get("args", {})


def log(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


class ModelEma:
    """Exponential moving average of model weights (idea_07's proven recipe; timm-style)."""

    def __init__(self, model, decay=0.999):
        self.module = self._clone(model)
        self.decay = decay

    @staticmethod
    def _clone(model):
        import copy
        m = copy.deepcopy(model)
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        return m

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)
