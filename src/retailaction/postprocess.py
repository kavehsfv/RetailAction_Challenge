"""Decode-time de-duplication (metric-grid NMS) + ActionFormer segment voting.

Idea 1 of idea_07_actionformer_retail_challenge.md, ported from the ActionFormer release
(related_papers/07_actionformer/actionformer_release):

- suppression: greedy NMS where "duplicate" is defined in the *evaluator's* geometry
  (tIoU >= gate_t AND min-over-views distance <= gate_s meters) — the metric matches each
  GT at most once (metrics.py `used` mask), so a second prediction of an already-matched
  action is always a false positive (~2.78 preds/GT measured on v2, val-400).
- voting: seg_voting (libs/utils/nms.py:L67-104) — refine each kept segment as an
  iou-weighted, score-weighted mean over ALL segments with tIoU >= vote_thresh.
  Configs use vote_thresh 0.9-0.95 (NOT the suppression gate): voting must average only
  near-identical segments, which is why the loose-gate variant measured worse than pure
  suppression in the idea file (+0.031 vs +0.038).
"""
import numpy as np


def _tiou(p, g):
    inter = min(p[1], g[1]) - max(p[0], g[0])
    union = max(p[1], g[1]) - min(p[0], g[0])
    return max(inter, 0.0) / max(union, 1e-6)


def _dmin(ps, gs, f):  # the metric's min-over-views distance in meters
    return min(np.hypot(ps[2 * v] - gs[2 * v], ps[2 * v + 1] - gs[2 * v + 1]) * f[v] for v in range(2))


def merge_duplicates(preds, gts, gate_t=0.3, gate_s=0.3, vote=False, vote_thresh=0.9):
    """Greedy NMS in metric space; optional ActionFormer seg_voting (refines kept seeds).

    preds/gts: dict sid -> {'temporal'(q,2), 'spatial'(q,4), 'scores'(q,)} / {'factors'(2,)}.
    gate_t = 0 disables everything (v2 behaviour).
    """
    if gate_t <= 0:
        return preds
    out = {}
    for sid, p in preds.items():
        pt, ps, sc = p["temporal"].copy(), p["spatial"].copy(), p["scores"].copy()
        f = gts[sid]["factors"]
        order = np.argsort(-sc, kind="stable")
        kept, done = [], np.zeros(len(sc), dtype=bool)
        for i in order:                          # --- suppression (metric-grid NMS) ---
            if done[i]:
                continue
            clus = [j for j in order if not done[j]
                    and _tiou(pt[i], pt[j]) >= gate_t and _dmin(ps[i], ps[j], f) <= gate_s]
            done[clus] = True
            kept.append(i)
        if vote:                                 # --- ActionFormer seg_voting ---
            for i in kept:                       # nms.py:L100: w = mask * score * iou
                votes = [j for j in range(len(sc))
                         if _tiou(pt[i], pt[j]) >= vote_thresh and _dmin(ps[i], ps[j], f) <= gate_s]
                if len(votes) > 1:
                    w = np.array([sc[j] * _tiou(pt[i], pt[j]) for j in votes])
                    if w.sum() > 0:              # seed itself has iou=1 and the max score
                        w = w / w.sum()
                        pt[i] = (w[:, None] * pt[votes]).sum(0)
                        ps[i] = (w[:, None] * ps[votes]).sum(0)
        out[sid] = {"temporal": pt[kept], "spatial": ps[kept], "scores": sc[kept]}
    return out
