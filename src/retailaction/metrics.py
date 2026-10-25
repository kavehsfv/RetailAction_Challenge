"""Paper metric: mAP over a 10x10 grid of temporal-IoU x spatial-distance(m) thresholds.

mAP   = mean AP over all 100 (tau_t, tau_s) configs
mAPs  = mean AP over distance thresholds at the lowest temporal IoU threshold
mAPt  = mean AP over temporal thresholds at the highest distance threshold
"""
import numpy as np

TAU_T = np.round(np.arange(0.05, 0.501, 0.05), 3)   # 0.05 ... 0.50
TAU_S = np.round(np.arange(0.1, 1.001, 0.1), 3)     # 0.1 ... 1.0 meters


def _tiou(p, g):
    inter = np.minimum(p[..., 1], g[..., 1]) - np.maximum(p[..., 0], g[..., 0])
    union = np.maximum(p[..., 1], g[..., 1]) - np.minimum(p[..., 0], g[..., 0])
    return np.clip(inter, 0, None) / np.maximum(union, 1e-6)


def _dists(p_sp, g_sp, factors):
    """meters in each view: (q,n,2)"""
    d = np.empty((p_sp.shape[0], g_sp.shape[0], 2))
    for v in range(2):
        dx = p_sp[:, None, 2 * v] - g_sp[None, :, 2 * v]
        dy = p_sp[:, None, 2 * v + 1] - g_sp[None, :, 2 * v + 1]
        d[:, :, v] = np.hypot(dx, dy) * factors[v]
    return d


def _voc_ap(tp, fp, ngt):
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / max(ngt, 1)
    prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(preds, gts, tau_t=TAU_T, tau_s=TAU_S, verbose=False):
    """preds/gts: dict sid -> dict(temporal(q,2), spatial(q,4) [, scores(q,)], factors(2,))"""
    sids = sorted(gts)
    per_sample = []
    ngt_total = 0
    for sid in sids:
        g = gts[sid]
        p = preds.get(sid)
        gt_t = np.asarray(g["temporal"], dtype=np.float64).reshape(-1, 2)
        gt_s = np.asarray(g["spatial"], dtype=np.float64).reshape(-1, 4)
        f = np.asarray(g["factors"], dtype=np.float64)
        ngt_total += len(gt_t)
        if p is None or len(p["scores"]) == 0 or len(gt_t) == 0:
            per_sample.append(None)
            continue
        pt = np.asarray(p["temporal"], dtype=np.float64).reshape(-1, 2)
        ps = np.asarray(p["spatial"], dtype=np.float64).reshape(-1, 4)
        sc = np.asarray(p["scores"], dtype=np.float64).reshape(-1)
        per_sample.append({
            "tiou": _tiou(pt[:, None, :], gt_t[None, :, :]),
            "dmin": _dists(ps, gt_s, f).min(axis=2),
            "score": sc,
            "n_gt": len(gt_t),
            "n_pred": len(sc),
        })

    g_sample = np.concatenate([np.full(s["n_pred"], i) for i, s in enumerate(per_sample) if s is not None]) \
        if any(s is not None for s in per_sample) else np.zeros(0, dtype=int)
    g_local = np.concatenate([np.arange(s["n_pred"]) for s in per_sample if s is not None]) \
        if len(g_sample) else np.zeros(0, dtype=int)
    scores = np.concatenate([s["score"] for s in per_sample if s is not None]) if len(g_sample) else np.zeros(0)
    o = np.argsort(-scores, kind="stable")
    g_sample, g_local = g_sample[o], g_local[o]

    ap = np.zeros((len(tau_t), len(tau_s)))
    for i, tt in enumerate(tau_t):
        for j, ts in enumerate(tau_s):
            tp = np.zeros(len(g_sample), dtype=bool)
            fp = np.zeros(len(g_sample), dtype=bool)
            used = [np.zeros(s["n_gt"], dtype=bool) if s is not None else None for s in per_sample]
            for k in range(len(g_sample)):
                si, li = int(g_sample[k]), int(g_local[k])
                s = per_sample[si]
                m = (s["tiou"][li] > tt) & (s["dmin"][li] < ts) & ~used[si]
                if m.any():
                    used[si][int(np.argmax(m))] = True
                    tp[k] = True
                else:
                    fp[k] = True
            ap[i, j] = _voc_ap(tp, fp, ngt_total)

    res = {"mAP": float(ap.mean()), "mAPs": float(ap[0].mean()), "mAPt": float(ap[:, -1].mean()),
           "n_gt": int(ngt_total), "AP": ap}
    if verbose:
        res["tau_t"], res["tau_s"] = list(tau_t), list(tau_s)
    return res
