"""Train the DETR model with combine_ideas_v3 additions:
- STMixer existence head + matched-only CE + per-layer fresh re-matching (with decoder.norm on aux layers)
- ActionFormer temporal DIoU + EMA + wd-hygiene
- PointTAD dense per-frame multi-label BCE aux head (--dense-weight)
- MultiTSF per-frame pose-proximity prior soft-CE aux head (--prior-weight)
- Multi-HMR 2 / Camera-Pose-AE metric-space spatial loss & matcher cost (--meter-spatial)
- TubeR matched-query quality upweighting (--up-matched-q)
- All-Query True Continuous IoU Head + Pairwise Intra-Clip Ranking (--iou-weight, --iou-rank-weight)
- MV-TAP 32-frame 2-view hand trajectory tokens (--use-pose-tokens)
- DiGIT log-duration smooth-L1 loss (--log-w-weight)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from .data import build_loader
from .engine import RANK_MODES, ModelEma, load_ckpt, log, save_ckpt, score
from .loss import Criterion
from .model import MultiViewActionDETR

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
SPLITS = {"train": "train/train", "val": "validation/validation", "test": "test/test"}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--out", default="outputs/run1")
    p.add_argument("--backbone", default="mvit_v2_s", choices=["mvit_v2_s", "mvit_v1_b"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--grid-hw", type=int, default=7)
    p.add_argument("--queries", type=int, default=5)
    p.add_argument("--enc-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="cap train samples (0=all)")
    p.add_argument("--val-limit", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--init", default="", help="checkpoint to initialize weights from (fresh schedule)")
    p.add_argument("--use-ema-init", action="store_true", help="initialize from checkpoint's model_ema weights")
    p.add_argument("--resume", default="", help="checkpoint to continue training from (keeps epoch/schedule)")
    p.add_argument("--device", default="cuda")
    # ---- idea_11 / combine_ideas_v2 / combine_ideas_v3 ----
    p.add_argument("--obj-weight", type=float, default=2.0)
    p.add_argument("--aux-weight", type=float, default=0.4)
    p.add_argument("--no-aux", action="store_true", help="disable per-layer supervision")
    p.add_argument("--all-cls", action="store_true", help="v2 CE over all queries (disable matched-only)")
    p.add_argument("--no-obj", action="store_true", help="disable the existence head (pure v2 loss)")
    p.add_argument("--new-head-lr", type=float, default=0.0,
                   help="higher LR for fresh heads (obj_head, dense_head, prior_head, iou_head, pose_encoder)")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--layer-decay", type=float, default=0.0,
                   help="layer-wise LR decay gamma (Idea 5); 0 = off")
    p.add_argument("--wd-hygiene", action="store_true", help="no wd on biases/norms/1-D params")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diou", action="store_true", help="temporal DIoU loss (idea_07)")
    p.add_argument("--meter-spatial", action="store_true",
                   help="scale spatial L1 loss and matcher cost by factors/2.74 (idea_20/24)")
    p.add_argument("--up-matched-q", type=float, default=1.0,
                   help="upweight matched queries in quality BCE (idea_12 TubeR)")
    p.add_argument("--dense-weight", type=float, default=0.0,
                   help="PointTAD dense per-frame multi-label BCE weight (idea_05, e.g. 5.0)")
    p.add_argument("--prior-weight", type=float, default=0.0,
                   help="MultiTSF per-frame pose-proximity prior weight (idea_23, e.g. 1.0)")
    p.add_argument("--iou-weight", type=float, default=0.0,
                   help="All-query true continuous IoU head weight (combine_ideas_v3, e.g. 3.0)")
    p.add_argument("--iou-rank-weight", type=float, default=0.0,
                   help="Intra-clip pairwise ranking loss weight on iou_head (e.g. 1.0)")
    p.add_argument("--use-pose-tokens", action="store_true",
                   help="Encode 32-frame 2-view hand trajectories into decoder memory (idea_19)")
    p.add_argument("--log-w-weight", type=float, default=0.0,
                   help="DiGIT log-duration smooth-L1 loss weight (idea_14, e.g. 0.2)")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="enable activation checkpointing in MViT backbone to allow larger batch/resolution")
    p.add_argument("--val-rank", default="obj*q", choices=RANK_MODES,
                   help="ranking mode used for validation and best.pt checkpoint selection")
    return p.parse_args()


def make_optimizer(args, model, fresh_keys=()):
    """Param groups: backbone | decoder (layer-decayed) | heads; wd hygiene on 1-D params."""
    names = dict(model.named_parameters())
    fresh_set = set(fresh_keys)

    def is_fresh_head(n):
        return (n in fresh_set) or any(h in n for h in ("dense_head", "prior_head", "iou_head", "pose_encoder"))

    def is_backbone(n):
        return n.startswith("backbone.")

    def lr_of(n):
        if args.new_head_lr > 0 and is_fresh_head(n):
            return args.new_head_lr
        return args.backbone_lr if is_backbone(n) else args.lr

    def scale_of(n):
        if args.layer_decay <= 0 or is_fresh_head(n) or not n.startswith("decoder."):
            return 1.0
        li = int(n.split(".")[2]) if n.split(".")[2].isdigit() else 0
        return args.layer_decay ** (len(model.decoder.layers) - 1 - li)

    def wd_of(n, p):
        return 0.0 if (args.wd_hygiene and p.dim() <= 1) else args.wd

    sig = {}
    for n, p in names.items():
        key = (lr_of(n) * scale_of(n), wd_of(n, p))
        sig.setdefault(key, []).append(p)
    groups = [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in sig.items()]
    seen = [p for g in groups for p in g["params"]]
    assert len(seen) == len(names) and len(set(id(p) for p in seen)) == len(names), \
        "param groups must cover every parameter exactly once"
    return torch.optim.AdamW(groups)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    model = MultiViewActionDETR(
        backbone=args.backbone, nqueries=args.queries,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        grid_hw=args.grid_hw,
        obj_head=not args.no_obj,
        iou_head=args.iou_weight > 0,
        use_pose_tokens=args.use_pose_tokens,
        aux_layers=not args.no_aux,
        dense_aux=args.dense_weight > 0,
        prior_aux=args.prior_weight > 0, grad_ckpt=args.grad_ckpt,
    ).to(dev)
    crit = Criterion(
        weight_obj=0.0 if args.no_obj else args.obj_weight,
        weight_aux=0.0 if args.no_aux else args.aux_weight,
        aux_layers=not args.no_aux,
        matched_cls_only=not args.all_cls,
        temporal_loss="diou" if args.diou else "giou",
        meter_spatial=args.meter_spatial,
        up_matched_q=args.up_matched_q,
        weight_dense=args.dense_weight,
        weight_prior=args.prior_weight,
        weight_iou=args.iou_weight,
        weight_iou_rank=args.iou_rank_weight,
        weight_log_w=args.log_w_weight,
    )
    crit.attach_heads(model)

    train_dl = build_loader(os.path.join(args.root, SPLITS["train"]), frames=args.frames, size=args.size,
                            train=True, batch=args.batch, workers=args.workers,
                            limit=args.limit or None)
    val_dl = build_loader(os.path.join(args.root, SPLITS["val"]), frames=args.frames, size=args.size,
                          train=False, batch=args.batch, workers=args.workers, limit=args.val_limit or None)

    start, best = 0, 0.0
    fresh_keys = []
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        sd = ck["model_ema"] if (args.use_ema_init and "model_ema" in ck) else ck["model"]
        own = model.state_dict()
        fresh_keys = [k for k in own if k not in sd or own[k].shape != sd[k].shape]
        load_ckpt(args.init, model, use_ema_init=args.use_ema_init)
    if args.resume:
        start, best, _ = load_ckpt(args.resume, model)
        print(f"resumed from {args.resume} epoch={start} best={best:.4f}")

    opt = make_optimizer(args, model, fresh_keys=fresh_keys)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * ((len(train_dl) + args.accum - 1) // args.accum)))
    ema = ModelEma(model, decay=args.ema_decay) if args.ema else None

    val_rank = "cls" if args.no_obj else args.val_rank
    print(f"train samples: {len(train_dl.dataset)}  val samples: {len(val_dl.dataset)}  "
          f"val_rank: {val_rank}  fresh_keys: {fresh_keys}", flush=True)
    for epoch in range(start, args.epochs):
        model.train()
        t0, run = time.time(), {}
        opt.zero_grad(set_to_none=True)
        for it, (vids, tgt) in enumerate(train_dl):
            vids = vids.to(dev, non_blocking=True)
            ht = torch.stack(tgt["hand_tracks"]).to(dev, non_blocking=True) if "hand_tracks" in tgt else None
            tgt = {k: ([t.to(dev) for t in v] if k in ("labels", "spatial", "temporal") else v)
                   for k, v in tgt.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(vids, hand_tracks=ht)
                loss = crit(out, tgt)
            if not torch.isfinite(loss["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch={epoch + 1} batch={it}: {loss}")
            # Flush the last partial accumulation group with the correct mean scale.
            group_start = (it // args.accum) * args.accum
            group_size = min(args.accum, len(train_dl) - group_start)
            (loss["loss"] / group_size).backward()
            for k, v in loss.items():
                run[k] = run.get(k, 0.0) + float(v.detach())
            if (it + 1) % args.accum == 0 or it + 1 == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update(model)
        n = max(1, len(train_dl))
        msg = f"epoch {epoch+1}/{args.epochs} " + " ".join(f"{k}={v/n:.4f}" for k, v in run.items())
        msg += f" {time.time()-t0:.0f}s"
        if (epoch + 1) % args.eval_every == 0:
            res = score(model, val_dl, dev, rank=val_rank)
            msg += f" | val({val_rank}) mAP={res['mAP']:.4f} mAPs={res['mAPs']:.4f} mAPt={res['mAPt']:.4f} ({res['seconds']}s)"
            ema_res = None
            if ema is not None:
                ema_res = score(ema.module, val_dl, dev, rank=val_rank)
                msg += f" | EMA val({val_rank}) mAP={ema_res['mAP']:.4f}"
            # Select using the configured deployed ranking only, never an unrelated hybrid score.
            sel_score = max(res["mAP"], ema_res["mAP"] if ema_res is not None else res["mAP"])
            if sel_score > best:
                best = sel_score
                save_ckpt(f"{args.out}/best.pt", model, opt, epoch + 1, best, args, ema=ema)
            log(f"{args.out}/log.jsonl", {"epoch": epoch + 1,
                                          **{k: v / n for k, v in run.items()},
                                          "val_mAP": res["mAP"], "val_mAPs": res["mAPs"],
                                          "val_mAPt": res["mAPt"],
                                          **({"ema_mAP": ema_res["mAP"], "ema_mAPs": ema_res["mAPs"],
                                              "ema_mAPt": ema_res["mAPt"]} if ema_res else {}),
                                          "sel_mAP": sel_score,
                                          "lr": sched.get_last_lr()[0]})
        print(msg, flush=True)
        save_ckpt(f"{args.out}/last.pt", model, opt, epoch + 1, best, args, ema=ema)
    print(f"best val({val_rank}) mAP={best:.4f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
