"""Evaluate a checkpoint (or comma-separated ensemble) on a labelled split.

combine_ideas_v3 features:
- Auto-detects `obj_head`, `iou_head`, `use_pose_tokens`, `grid_hw` from checkpoint state_dict / args
- Supports `--pad-t` symmetric temporal boundary padding (e.g. `0.015`)
- Supports comma-separated `--ckpt a.pt:ema,a.pt:raw,b.pt:ema` for multi-checkpoint / dual-trajectory ensembling
- Supports `--scales 224,256` for multi-resolution TTA.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

from .data import build_loader
from .engine import RANK_MODES, collect, load_weights, rank_score, score
from .model import MultiViewActionDETR

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
SPLITS = {"train": "train/train", "val": "validation/validation", "test": "test/test",
          "challenge": "challenge_test_set/vam-dataset-challenge_test_no_labels/data"}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--ckpt", required=True, help="single path or comma-separated paths for ensemble")
    p.add_argument("--split", default="", choices=["", "train", "val", "test", "challenge"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--size", type=int, default=0, help="0 = use the checkpoint's value")
    p.add_argument("--frames", type=int, default=0, help="0 = use the checkpoint's value")
    p.add_argument("--max-queries", type=int, default=5)
    p.add_argument("--rank", default="obj*q", choices=RANK_MODES)
    p.add_argument("--use-ema", action="store_true", help="use the checkpoint's model_ema weights")
    p.add_argument("--nms-gate-t", type=float, default=0.0, help="0 = off; else tIoU gate (idea_07)")
    p.add_argument("--nms-gate-s", type=float, default=0.3, help="min-view distance gate in meters")
    p.add_argument("--vote", action="store_true", help="ActionFormer seg voting on kept seeds")
    p.add_argument("--vote-thresh", type=float, default=0.9)
    p.add_argument("--fuse", action="store_true", help="multi-pass TTA fusion (idea_08)")
    p.add_argument("--fuse-passes", default="s0,s1,flip,flip1", help="subset of s0,s1,flip,flip1")
    p.add_argument("--fuse-dedup", default="nms", choices=["nms", "shadow"])
    p.add_argument("--scales", default="", help="comma-separated resolutions for TTA, e.g. 224,256")
    p.add_argument("--pad-t", type=float, default=0.0, help="symmetric temporal boundary padding (e.g. 0.015)")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--out", default="outputs/submission.csv")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def to_prediction_json(temporal, spatial, scores, max_queries=5):
    acts = []
    for t, s, sc in zip(temporal, spatial, scores):
        acts.append({
            "start": float(t[0]), "end": float(t[1]),
            "spatial": {"action_cam": {
                "rank0": {"x": float(s[0]), "y": float(s[1])},
                "rank1": {"x": float(s[2]), "y": float(s[3])}}},
            "score": float(sc),
        })
    acts.sort(key=lambda a: -a["score"])
    return json.dumps({"actions": acts[:max_queries]})


def _load_one(path, device, use_ema=False):
    mode = None
    if path.endswith(":ema"):
        path, mode = path[:-4], "ema"
    elif path.endswith(":raw"):
        path, mode = path[:-4], "raw"
    want_ema = (mode == "ema") or (mode is None and use_ema)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ckargs = ck.get("args", {})
    sd = ck["model_ema"] if (want_ema and "model_ema" in ck) else ck["model"]
    model = MultiViewActionDETR(
        backbone=ckargs.get("backbone", "mvit_v2_s"),
        nqueries=ckargs.get("queries", 5),
        enc_layers=ckargs.get("enc_layers", 2),
        dec_layers=ckargs.get("dec_layers", 3),
        grid_t=ckargs.get("grid_t", 8),
        grid_hw=ckargs.get("grid_hw", 7),
        obj_head=any(k.startswith("obj_head") for k in sd),
        iou_head=any(k.startswith("iou_head") for k in sd),
        use_pose_tokens=any(k.startswith("pose_encoder") for k in sd),
        aux_layers=False,
        dense_aux=False,
        prior_aux=False,
    ).to(device)
    model.native_size = ckargs.get("size", 224)
    load_weights(model, sd)
    model.eval()
    return model, ck, ckargs


def main():
    args = get_args()
    paths = [p.strip() for p in args.ckpt.split(",") if p.strip()]
    models = []
    ckargs = {}
    for p in paths:
        m, ck, cka = _load_one(p, args.device, args.use_ema)
        models.append(m)
        if not ckargs:
            ckargs = cka
        print(f"loaded {p} ({'EMA' if (args.use_ema and 'model_ema' in ck) else 'raw'}, "
              f"epoch {ck.get('epoch')}, best {ck.get('best')})")
    model_or_list = models[0] if len(models) == 1 else models
    args.size = args.size or max(getattr(m, "native_size", 224) for m in models)
    args.frames = args.frames or ckargs.get("frames", 16)
    scales = tuple(int(s) for s in args.scales.split(",") if s.strip()) if args.scales else (None,)

    if args.split:
        d0 = os.path.join(args.root, SPLITS[args.split])
        common = dict(frames=args.frames, size=args.size, train=False, batch=args.batch,
                      workers=args.workers, limit=args.limit or None)
        if args.fuse or len(models) > 1 or scales != (None,):
            from .engine import score_fused
            passes = tuple(args.fuse_passes.split(",")) if args.fuse else ("s0",)
            dl0 = build_loader(d0, frame_shift=0, **common)
            dl1 = build_loader(d0, frame_shift=1, **common)
            res = score_fused(model_or_list, dl0, dl1, args.device,
                              passes=passes, dedup=args.fuse_dedup, rank=args.rank, scales=scales,
                              pad_t=args.pad_t,
                              merge=(None if (args.nms_gate_t <= 0 or args.fuse_dedup != "nms") else
                                     {"gate_t": args.nms_gate_t, "gate_s": args.nms_gate_s,
                                      "vote": args.vote, "vote_thresh": args.vote_thresh}))
        else:
            dl = build_loader(d0, **common)
            res = score(models[0], dl, args.device, rank=args.rank, pad_t=args.pad_t,
                        merge=(None if args.nms_gate_t <= 0 else
                               {"gate_t": args.nms_gate_t, "gate_s": args.nms_gate_s,
                                "vote": args.vote, "vote_thresh": args.vote_thresh}))
        print(f"{args.split} [{args.rank}{' EMA' if args.use_ema else ''}"
              f"{' fuse(' + args.fuse_passes + ')' if args.fuse else ''}"
              f"{' scales=' + args.scales if args.scales else ''}"
              f"{' pad_t=' + str(args.pad_t) if args.pad_t else ''}]: "
              f"mAP={res['mAP']:.4f} mAPs={res['mAPs']:.4f} mAPt={res['mAPt']:.4f} "
              f"(n_gt={res['n_gt']}, {res['seconds']}s)")
        base_p = paths[0].split(":")[0]
        np.save(base_p.replace(".pt", f"_{args.split}_ap.npy"), res["AP"])

    if args.submit:
        from .engine import collect, collect_fused, dedup_preds
        common = dict(frames=args.frames, size=args.size, train=False,
                      batch=args.batch, workers=args.workers)
        d0 = os.path.join(args.root, SPLITS["challenge"])
        if args.fuse or len(models) > 1 or scales != (None,):
            passes = tuple(args.fuse_passes.split(",")) if args.fuse else ("s0",)
            dl0 = build_loader(d0, frame_shift=0, **common)
            dl1 = build_loader(d0, frame_shift=1, **common)
            preds, gts = collect_fused(model_or_list, dl0, dl1, args.device,
                                       passes=passes, rank=args.rank, scales=scales, pad_t=args.pad_t)
            preds = dedup_preds(preds, gts, args.fuse_dedup,
                                (None if (args.nms_gate_t <= 0 or args.fuse_dedup != "nms") else
                                 {"gate_t": args.nms_gate_t, "gate_s": args.nms_gate_s,
                                  "vote": args.vote, "vote_thresh": args.vote_thresh}))
        else:
            dl = build_loader(d0, **common)
            preds, gts = collect(models[0], dl, args.device, rank=args.rank, pad_t=args.pad_t)
            if args.nms_gate_t > 0:
                preds = dedup_preds(preds, gts, "nms",
                                    {"gate_t": args.nms_gate_t, "gate_s": args.nms_gate_s,
                                     "vote": args.vote, "vote_thresh": args.vote_thresh})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "prediction"])
            for sid in sorted(preds):
                p = preds[sid]
                w.writerow([sid, to_prediction_json(p["temporal"], p["spatial"], p["scores"],
                                                    args.max_queries)])
        print(f"wrote {args.out} ({len(preds)} rows)")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
