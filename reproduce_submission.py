#!/usr/bin/env python3
"""
Standalone inference & evaluation entrypoint for the RetailAction Challenge
(PRAW Workshop @ WACV 2027) — 0.7942 Public Leaderboard Submission (V4).

Authors:
  Kaveh Safavigerdini, Kannappan Palaniappan
  EECS Department, University of Missouri

Usage:
  # 1. Reproduce the 2,500-row challenge submission.csv (0.7942 Public LB)
  python reproduce_submission.py --data-root data/raw --out submission.csv

  # 2. Evaluate on full validation (1,277 clips) or local test (2,501 clips)
  python reproduce_submission.py --data-root data/raw --split validation
  python reproduce_submission.py --data-root data/raw --split test

  # 3. Verify a generated submission.csv against sample_submission.csv and reference CSV
  python reproduce_submission.py --verify-only --out submission.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_v4 import evaluate_all_clips, fuse, padded
from retailaction.data import build_loader
from retailaction.engine import collect_fused
from retailaction.metrics import evaluate
from retailaction.predict import SPLITS as PREDICT_SPLITS, _load_one, to_prediction_json

SPLIT_MAP = {
    "train": PREDICT_SPLITS["train"],
    "val": PREDICT_SPLITS["val"],
    "validation": PREDICT_SPLITS["val"],
    "test": PREDICT_SPLITS["test"],
    "challenge": PREDICT_SPLITS["challenge"],
}
REFERENCE_SUBMISSION_SHA256 = "b7636dec15f3aa37b996e9b6a3fcc81c954487429ae1e67722be8a187bb1b561"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint(preferred: pathlib.Path, fallbacks: List[pathlib.Path]) -> pathlib.Path:
    for p in [preferred] + fallbacks:
        if p.exists():
            return p.resolve()
    tried = ", ".join(str(p) for p in [preferred] + fallbacks)
    raise FileNotFoundError(f"Could not find checkpoint. Tried: {tried}")


def collect_member_predictions(
    ckpt_path: pathlib.Path,
    mode: str,
    dl0,
    dl1,
    passes: Tuple[str, ...],
    rank: str,
    device: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    model, _, _ = _load_one(f"{ckpt_path}:{mode}", device)
    try:
        preds, gts = collect_fused(
            model,
            dl0,
            dl1,
            device,
            passes=passes,
            rank=rank,
            pad_t=0.0,  # Applied exactly once after ensemble averaging via padded()
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return preds, gts


def write_submission_in_sample_order(
    sample_csv: pathlib.Path,
    preds: Dict[str, Dict[str, np.ndarray]],
    out_csv: pathlib.Path,
    max_queries: int = 5,
) -> Dict[str, object]:
    with sample_csv.open("r", newline="") as f:
        sample_ids = [r["id"] for r in csv.DictReader(f)]
    if len(sample_ids) != 2500 or len(set(sample_ids)) != 2500 or set(sample_ids) != set(preds):
        missing = sorted(set(sample_ids) - set(preds))[:5]
        extra = sorted(set(preds) - set(sample_ids))[:5]
        raise RuntimeError(f"Challenge predictions do not match 2500 sample IDs: missing={missing}, extra={extra}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_actions = 0
    for sid in sample_ids:
        p = preds[sid]
        payload = to_prediction_json(p["temporal"], p["spatial"], p["scores"], max_queries)
        acts = json.loads(payload)["actions"]
        total_actions += len(acts)
        rows.append((sid, payload))

    tmp = out_csv.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        w.writerows(rows)
    tmp.replace(out_csv)

    return {
        "rows": len(rows),
        "actions": total_actions,
        "sha256": sha256_file(out_csv),
        "path": str(out_csv.resolve()),
    }


def verify_submission_csv(
    csv_path: pathlib.Path,
    sample_csv: pathlib.Path,
    reference_csv: Optional[pathlib.Path] = None,
) -> Dict[str, object]:
    with sample_csv.open("r", newline="") as f:
        sample_ids = [r["id"] for r in csv.DictReader(f)]
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["id", "prediction"]:
            raise RuntimeError(f"Invalid header {r.fieldnames}")
        rows = list(r)
    if len(rows) != 2500:
        raise RuntimeError(f"Expected 2,500 rows, got {len(rows)}")
    ids = [row["id"] for row in rows]
    if ids != sample_ids:
        raise RuntimeError("Row IDs do not match sample_submission.csv order")

    total_actions = 0
    for row in rows:
        payload = json.loads(row["prediction"])
        acts = payload["actions"]
        if not (1 <= len(acts) <= 5):
            raise RuntimeError(f"Invalid action count {len(acts)} for {row['id']}")
        total_actions += len(acts)
        scores = [float(a["score"]) for a in acts]
        if scores != sorted(scores, reverse=True):
            raise RuntimeError(f"Scores not sorted descending for {row['id']}")
        for a in acts:
            s, e = float(a["start"]), float(a["end"])
            if not (0.0 <= s <= e <= 1.0):
                raise RuntimeError(f"Invalid interval [{s}, {e}] for {row['id']}")
            for rk in ("rank0", "rank1"):
                pt = a["spatial"]["action_cam"][rk]
                x, y = float(pt["x"]), float(pt["y"])
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise RuntimeError(f"Invalid {rk} coordinates ({x}, {y}) for {row['id']}")

    sha = sha256_file(csv_path)
    result: Dict[str, object] = {
        "valid_format": True,
        "rows": len(rows),
        "actions": total_actions,
        "sample_order_exact": True,
        "sha256": sha,
        "matches_07942_reference_sha256": (sha == REFERENCE_SUBMISSION_SHA256),
    }

    if reference_csv is not None and reference_csv.exists():
        with reference_csv.open("r", newline="") as f:
            ref_rows = list(csv.DictReader(f))
        max_score_diff = 0.0
        max_temp_diff = 0.0
        max_spat_diff = 0.0
        for r_new, r_ref in zip(rows, ref_rows):
            a_new = json.loads(r_new["prediction"])["actions"]
            a_ref = json.loads(r_ref["prediction"])["actions"]
            for x, y in zip(a_new, a_ref):
                max_score_diff = max(max_score_diff, abs(float(x["score"]) - float(y["score"])))
                max_temp_diff = max(
                    max_temp_diff,
                    abs(float(x["start"]) - float(y["start"])),
                    abs(float(x["end"]) - float(y["end"])),
                )
                for rk in ("rank0", "rank1"):
                    px = x["spatial"]["action_cam"][rk]
                    py = y["spatial"]["action_cam"][rk]
                    max_spat_diff = max(
                        max_spat_diff,
                        abs(float(px["x"]) - float(py["x"])),
                        abs(float(px["y"]) - float(py["y"])),
                    )
        result["max_abs_diff_vs_reference"] = {
            "score": max_score_diff,
            "temporal": max_temp_diff,
            "spatial": max_spat_diff,
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the 0.7942 RetailAction V4 submission or split evaluation.")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"), help="Path to data/raw containing videos, annotations, sample_submission.csv")
    ap.add_argument("--split", choices=["challenge", "val", "validation", "test"], default="challenge", help="Dataset split to run")
    ap.add_argument("--ckpt-256", default=str(ROOT / "weights" / "v4_control256_best.pt"), help="256x256 (grid_hw=8) V4 checkpoint")
    ap.add_argument("--ckpt-288", default=str(ROOT / "weights" / "v4_high288_best.pt"), help="288x288 (grid_hw=9) V4 checkpoint")
    ap.add_argument("--common-size", type=int, default=288, help="Unified input frame resolution before per-member resize")
    ap.add_argument("--frames", type=int, default=16, help="Sampled frames per clip")
    ap.add_argument("--passes", default="s0,s1,flip,flip1", help="Comma-separated TTA passes")
    ap.add_argument("--rank", default="obj*q", help="Per-pass query ranking mode")
    ap.add_argument("--pad-t", type=float, default=0.015, help="Symmetric temporal boundary padding applied once after ensemble averaging")
    ap.add_argument("--batch", type=int, default=16, help="Inference batch size")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers per stream")
    ap.add_argument("--limit", type=int, default=0, help="Optional clip limit for quick smoke tests (0 = full split)")
    ap.add_argument("--out", default=str(ROOT / "submission.csv"), help="Output CSV path (when --split challenge)")
    ap.add_argument("--verify-only", action="store_true", help="Only verify an existing --out CSV without running GPU inference")
    args = ap.parse_args()

    data_root = pathlib.Path(args.data_root)
    sample_csv = data_root / "sample_submission.csv"
    out_csv = pathlib.Path(args.out)
    ref_csv = ROOT / "evidence" / "reference_submission_v4_0.7942.csv"

    if args.verify_only:
        ver = verify_submission_csv(out_csv, sample_csv, ref_csv)
        print(json.dumps(ver, indent=2))
        return

    ckpt_256 = resolve_checkpoint(
        pathlib.Path(args.ckpt_256),
        [
            ROOT / "weights" / "v4_control256_inference.pt",
            ROOT / "src" / "outputs" / "v4_control256" / "best.pt",
        ],
    )
    ckpt_288 = resolve_checkpoint(
        pathlib.Path(args.ckpt_288),
        [
            ROOT / "weights" / "v4_high288_inference.pt",
            ROOT / "src" / "outputs" / "v4_high288" / "best.pt",
        ],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    passes = tuple(p.strip() for p in args.passes.split(",") if p.strip())
    split_dir = data_root / SPLIT_MAP[args.split]

    print(f"[V4 Inference] split={args.split} ({split_dir}) device={device} common_size={args.common_size} passes={passes} rank={args.rank} pad_t={args.pad_t}")
    print(f"  ckpt_256: {ckpt_256} (sha256={sha256_file(ckpt_256)[:16]}...)")
    print(f"  ckpt_288: {ckpt_288} (sha256={sha256_file(ckpt_288)[:16]}...)")

    common = dict(
        frames=args.frames,
        size=args.common_size,
        batch=args.batch,
        workers=args.workers,
        train=False,
        limit=(args.limit or None),
    )
    dl0 = build_loader(str(split_dir), frame_shift=0, **common)
    dl1 = build_loader(str(split_dir), frame_shift=1, **common) if ("s1" in passes or "flip1" in passes) else None
    assert dl1 is None or dl0.dataset.ids == dl1.dataset.ids, "Loader clip ID order mismatch"

    members = [
        ("c_ema", ckpt_256, "ema"),
        ("c_raw", ckpt_256, "raw"),
        ("h_ema", ckpt_288, "ema"),
        ("h_raw", ckpt_288, "raw"),
    ]
    weights = {"c_ema": 1, "c_raw": 1, "h_ema": 1, "h_raw": 1}

    t0 = time.time()
    member_preds: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    gts: Dict[str, Dict[str, np.ndarray]] = {}
    for name, cpath, mode in members:
        mt0 = time.time()
        preds, gts = collect_member_predictions(
            cpath,
            mode=mode,
            dl0=dl0,
            dl1=dl1,
            passes=passes,
            rank=args.rank,
            device=device,
        )
        member_preds[name] = preds
        print(f"  collected {name} ({mode.upper()}) on {len(preds)} clips in {time.time() - mt0:.1f}s", flush=True)

    fused = fuse(member_preds, weights)
    if abs(args.pad_t - 0.015) < 1e-9:
        final = padded(fused)
    else:
        final = {}
        for sid, p in fused.items():
            tm = p["temporal"].copy()
            tm[:, 0] = (tm[:, 0] - args.pad_t).clip(0.0, 1.0)
            tm[:, 1] = (tm[:, 1] + args.pad_t).clip(0.0, 1.0)
            final[sid] = {**p, "temporal": tm}
    elapsed = time.time() - t0

    if args.split == "challenge" and not args.limit:
        meta = write_submission_in_sample_order(sample_csv, final, out_csv)
        ver = verify_submission_csv(out_csv, sample_csv, ref_csv)
        print(json.dumps({"status": "ok", "elapsed_sec": round(elapsed, 1), "submission": meta, "verification": ver}, indent=2))
    elif args.split == "challenge":
        print(json.dumps({"status": "smoke_ok", "clips": len(final), "elapsed_sec": round(elapsed, 1)}, indent=2))
    else:
        res = evaluate(final, gts)
        res_all = evaluate_all_clips(final, gts)
        print(
            json.dumps(
                {
                    "split": args.split,
                    "clips": len(final),
                    "elapsed_sec": round(elapsed, 1),
                    "mAP": float(res["mAP"]),
                    "mAPs": float(res["mAPs"]),
                    "mAPt": float(res["mAPt"]),
                    "all_clips_mAP": float(res_all["mAP"]),
                    "AP_s01": float(res["AP"][:, 0].mean()),
                    "AP_s02": float(res["AP"][:, 1].mean()),
                    "AP_t50": float(res["AP"][-1, :].mean()),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
