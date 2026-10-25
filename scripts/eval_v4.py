"""Paired validation-selected v4 evaluation, using the SAME production path for all members.

Fixed f4/obj*q/pad=.015/no NMS. Cache predictions with code/data/checkpoint identities.
Local test confirmation requires an already frozen outputs/selected_config.json; it never sweeps.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))
from retailaction.data import build_loader
from retailaction.engine import collect_fused
from retailaction.metrics import evaluate, _tiou, _dists, _voc_ap, TAU_T, TAU_S
from retailaction.predict import _load_one
from v4_provenance import (INFERENCE, checkpoint_identity, code_identity, dataset_identity,
                           gts_identity, verify_frozen)

SPLITS = {'val': 'validation/validation', 'test': 'test/test'}
MEMBERS = {
    'a_ema': ('combine_v3a', 'ema'), 'a_raw': ('combine_v3a', 'raw'),
    'b_ema': ('combine_v3b_256', 'ema'), 'b_raw': ('combine_v3b_256', 'raw'),
    'c_ema': ('v4_control256', 'ema'), 'c_raw': ('v4_control256', 'raw'),
    'h_ema': ('v4_high288', 'ema'), 'h_raw': ('v4_high288', 'raw'),
}
V3 = {'a_ema': 1, 'a_raw': 1, 'b_ema': 2, 'b_raw': 2}


def evaluate_all_clips(preds, gts):
    """Same greedy matcher, strict thresholds, stable order as historical evaluator; include empty GT FPs.

    Unlike eval_combine_v3.evaluate_keep_0gt this changes ONLY the empty-GT behavior, not matching
    tie breaks or threshold inequalities. Keep old evaluator intact for historical comparability.
    """
    samples = []
    ngt = 0
    for sid in sorted(gts):
        g = gts[sid]
        gt = np.asarray(g['temporal'], dtype=np.float64).reshape(-1, 2)
        gs = np.asarray(g['spatial'], dtype=np.float64).reshape(-1, 4)
        ngt += len(gt)
        p = preds.get(sid)
        if p is None or not len(p['scores']):
            continue
        pt = np.asarray(p['temporal'], dtype=np.float64).reshape(-1, 2)
        ps = np.asarray(p['spatial'], dtype=np.float64).reshape(-1, 4)
        samples.append(dict(n_gt=len(gt), score=np.asarray(p['scores'], dtype=np.float64),
                            tiou=_tiou(pt[:, None], gt[None]), dmin=_dists(ps, gs, g['factors']).min(2)))
    if not samples:
        return {'mAP': 0., 'mAPs': 0., 'mAPt': 0., 'AP': np.zeros((10, 10))}
    si = np.concatenate([np.full(len(s['score']), i) for i, s in enumerate(samples)])
    li = np.concatenate([np.arange(len(s['score'])) for s in samples])
    order = np.argsort(-np.concatenate([s['score'] for s in samples]), kind='stable')
    si, li = si[order], li[order]
    ap = np.zeros((10, 10))
    for i, tt in enumerate(TAU_T):
        for j, ts in enumerate(TAU_S):
            used = [np.zeros(s['n_gt'], dtype=bool) for s in samples]
            tp = np.zeros(len(si), dtype=bool)
            for k, (sidx, lidx) in enumerate(zip(si, li)):
                s = samples[sidx]
                ok = (s['tiou'][lidx] > tt) & (s['dmin'][lidx] < ts) & ~used[sidx]
                if ok.any():
                    used[sidx][int(np.argmax(ok))] = True
                    tp[k] = True
            ap[i, j] = _voc_ap(tp, ~tp, ngt)
    return {'mAP': float(ap.mean()), 'mAPs': float(ap[0].mean()), 'mAPt': float(ap[:, -1].mean()), 'AP': ap}


def fuse(preds, weights):
    keys = list(weights)
    w = np.array([weights[k] for k in keys], dtype=np.float64)
    w /= w.sum()
    sids = list(preds[keys[0]])
    assert all(set(preds[k]) == set(sids) for k in keys)
    return {sid: {field: np.sum(np.stack([preds[k][sid][field] for k in keys]) * w.reshape((-1,) + (1,) * np.asarray(preds[keys[0]][sid][field]).ndim), axis=0).astype(np.float32)
                  for field in ('temporal', 'spatial', 'scores')} for sid in sids}

def padded(preds):
    out = {}
    for sid, p in preds.items():
        tm = p["temporal"].copy()
        tm[:, 0] = (tm[:, 0] - .015).clip(0, 1)
        tm[:, 1] = (tm[:, 1] + .015).clip(0, 1)
        out[sid] = {**p, "temporal": tm}
    return out


def stats(name, pred, gts):
    pred = padded(pred)
    r = evaluate(pred, gts)
    r0 = evaluate_all_clips(pred, gts)
    row = dict(name=name, mAP=r['mAP'], mAPs=r['mAPs'], mAPt=r['mAPt'], all_clips_mAP=r0['mAP'],
               AP_s01=float(r['AP'][:, 0].mean()), AP_s02=float(r['AP'][:, 1].mean()),
               AP_t50=float(r['AP'][-1, :].mean()),
               AP_t05_s01=float(r['AP'][0, 0]), AP_t05_s02=float(r['AP'][0, 1]),
               AP_t50_s10=float(r['AP'][-1, -1]))
    print(json.dumps(row), flush=True)
    return row


def paired_bootstrap(p0, p1, gts, nboot=40):
    """Paired clip-level bootstrap (NOT independent-prediction sampling). Bounded, descriptive CI."""
    p0, p1 = padded(p0), padded(p1)
    rng = np.random.default_rng(20261023)
    ids = np.array(sorted(gts))
    diffs = []
    for _ in range(nboot):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        g = {str(i): gts[s] for i, s in enumerate(chosen)}
        a = {str(i): p0[s] for i, s in enumerate(chosen)}
        b = {str(i): p1[s] for i, s in enumerate(chosen)}
        diffs.append(evaluate(b, g)['mAP'] - evaluate(a, g)['mAP'])
    return dict(nboot=nboot, mean=float(np.mean(diffs)), ci95=np.quantile(diffs, [.025, .975]).tolist())


def cached_member(key, split, common_size, dl0, dl1, out, collect, code, dataset):
    tag, mode = MEMBERS[key]
    path = BASE / f'src/outputs/{tag}/best.pt'
    inference = {**INFERENCE, 'common_size': common_size, 'pad_t': 0.}
    sig = dict(checkpoint=checkpoint_identity(path, mode), inference=inference,
               code=code, dataset=dataset)
    cache = out / f'{split}_{key}_common{common_size}.pt'
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        if c['signature'] != sig:
            raise RuntimeError(f'Stale cache: {cache}; remove explicitly before recollection')
    elif collect:
        m, _, _ = _load_one(str(path) + ':' + mode, 'cuda')
        p, g = collect_fused(m, dl0, dl1, 'cuda', passes=tuple(INFERENCE['passes']),
                             rank=INFERENCE['rank'], pad_t=0.)
        c = dict(preds=p, gts=g, gt_sha256=gts_identity(g), signature=sig)
        temporary = cache.with_suffix('.tmp')
        torch.save(c, temporary)
        temporary.replace(cache)
        del m
        torch.cuda.empty_cache()
    else:
        raise FileNotFoundError(cache)
    if set(c['preds']) != set(c['gts']) or len(c['gts']) != dataset['clips']:
        raise RuntimeError(f'Incomplete IDs in cache: {cache}')
    if c['gt_sha256'] != gts_identity(c['gts']):
        raise RuntimeError(f'Ground-truth identity mismatch in cache: {cache}')
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=SPLITS, default='val')
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--workers', type=int, default=2, help='per loader; two persistent loaders = 4 total')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--bootstrap', type=int, default=0)
    args = ap.parse_args()
    out = BASE / 'outputs'
    out.mkdir(exist_ok=True)
    code = code_identity()
    if args.split == 'test':
        config = json.loads((out / 'selected_config.json').read_text())
        verify_frozen(config, MEMBERS)
        required = set(V3) | set(config['weights'])
    else:
        config = None
        required = set(MEMBERS)
    root = BASE / 'data/raw' / SPLITS[args.split]
    dataset = dataset_identity(root)
    def loaders(size):
        common = dict(frames=16, size=size, batch=args.batch, workers=args.workers, train=False)
        return build_loader(str(root), frame_shift=0, **common), build_loader(str(root), frame_shift=1, **common)
    dl0, dl1 = loaders(288)
    preds, identities, member_rows = {}, {}, []
    gts, gt_digest = None, None
    for key in MEMBERS:
        if key not in required:
            continue
        c = cached_member(key, args.split, 288, dl0, dl1, out, args.collect, code, dataset)
        if gt_digest is not None and c['gt_sha256'] != gt_digest:
            raise RuntimeError(f'Ground truth differs between member caches: {key}')
        gt_digest = c['gt_sha256']
        preds[key], gts = c['preds'], c['gts']
        identities[key] = c['signature']['checkpoint']
        member_rows.append(stats(key, preds[key], gts))
    del dl0, dl1
    base = fuse(preds, V3)
    rows = [stats('v3_common288', base, gts)]

    # Attribution control: reproduce the ACTUAL .7882 submission preprocessing,
    # not direct-native caches and not the common288 matched baseline.
    dl0, dl1 = loaders(256)
    original = {}
    for key in V3:
        c = cached_member(key, args.split, 256, dl0, dl1, out, args.collect, code, dataset)
        if c['gt_sha256'] != gt_digest:
            raise RuntimeError(f'Ground truth differs for original preprocessing: {key}')
        original[key] = c['preds']
    del dl0, dl1
    original = fuse(original, V3)
    rows.append(stats('v3_original_common256', original, gts))
    if config is not None:
        final = fuse(preds, config['weights'])
        rows.append(stats('FROZEN_v4', final, gts))
        torch.save(dict(preds=padded(final), baseline=padded(base), original_baseline=padded(original),
                        gts=gts, config=config, dataset=dataset), out / 'test_frozen.pt')
        (out / 'test_results.json').write_text(json.dumps(rows, indent=2))
    else:
        # Small prespecified family: matched control, high resolution, and simple equal mixtures.
        specs = {
            'control_dual': {'c_ema': 1, 'c_raw': 1},
            'high_dual': {'h_ema': 1, 'h_raw': 1},
            'high_plus_control': {'c_ema': 1, 'c_raw': 1, 'h_ema': 1, 'h_raw': 1},
            'v3_plus_high': {**V3, 'h_ema': 3, 'h_raw': 3},
            'v3_plus_control': {**V3, 'c_ema': 3, 'c_raw': 3},
        }
        for name, w in specs.items():
            rows.append(stats(name, fuse(preds, w), gts))
        result = dict(split=args.split, inference=INFERENCE, results=rows, specs=specs,
                      member_results=member_rows, checkpoints=identities, code=code,
                      dataset=dataset, gt_sha256=gt_digest)
        if args.bootstrap:
            # Descriptive only: no adjustment for validation model selection.
            best = max((r for r in rows if r['name'] in specs), key=lambda x: x['mAP'])
            result['bootstrap'] = paired_bootstrap(base, fuse(preds, specs[best['name']]), gts, args.bootstrap)
            result['bootstrap']['candidate'] = best['name']
            print(json.dumps(result['bootstrap']), flush=True)
        temporary = out / 'validation_results.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(out / 'validation_results.json')


if __name__ == '__main__':
    main()
