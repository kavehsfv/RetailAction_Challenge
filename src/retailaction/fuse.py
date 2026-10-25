"""idea_08 (Query-Matching-for-STAD): cross-rendering fusion + greedy mutual-exclusion linking.

combine_ideas_v2 fix:
`fuse_sample` supports direct `p["score"]` linear averaging across passes/models in addition to
legacy `p["cls"] * p["q"]`, preventing unintended squaring (`sc^2`) when `collect_fused`
pre-computes `rank_score(out, rank)`.
"""
import numpy as np


def tiou(t1, t2):
    inter = min(t1[1], t2[1]) - max(t1[0], t2[0])
    union = max(t1[1], t2[1]) - min(t1[0], t2[0])
    return max(inter, 0.0) / max(union, 1e-6)


def drop_shadow_dups(t, s, sc, factors, t_th=0.5, d_th=0.15):
    """Greedy mutual-exclusion linking (paper card __07), output-space form (idea file Idea 1):
    keep highest-scored, drop any pred with tIoU > t_th AND < d_th meters in BOTH views to a
    kept one. Pre-registered rule from idea_19 (0.15 m / tIoU 0.5) — deliberately NOT tuned.
    Returns filtered (temporal, spatial, scores)."""
    keep = np.ones(len(sc), dtype=bool)
    for a in np.argsort(-sc, kind="stable"):
        if not keep[a]:
            continue
        for b in range(len(sc)):
            if b == a or not keep[b] or sc[b] > sc[a]:
                continue
            if tiou(t[a], t[b]) <= t_th:
                continue
            near = all(float(np.hypot(s[a][2 * v] - s[b][2 * v], s[a][2 * v + 1] - s[b][2 * v + 1])
                             * factors[v]) < d_th for v in (0, 1))
            if near:
                keep[b] = False
    return t[keep], s[keep], sc[keep]


def fuse_sample(passes):
    """Slot-aligned fusion across rendering passes and/or ensemble members.
    passes: list of per-pass dicts {temporal(Q,2), spatial(Q,4), score(Q,) | (cls(Q,), q(Q,))}.
    """
    t = np.mean([p["temporal"] for p in passes], axis=0)
    s = np.mean([p["spatial"] for p in passes], axis=0)
    sc = np.mean([p["score"] if "score" in p else (p["cls"] * p["q"]) for p in passes], axis=0)
    return t, s, sc
