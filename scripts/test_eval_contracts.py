"""CPU-only regression tests for v4's evaluation and deployment arithmetic.

No dataset labels, trained models, or GPU are needed. Run with the pt Python.
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'scripts'))
from retailaction.metrics import evaluate, TAU_T, TAU_S
from eval_v4 import evaluate_all_clips, fuse, padded, stats


def gt(t, s):
    return {'temporal': np.asarray(t, dtype=np.float32).reshape(-1, 2),
            'spatial': np.asarray(s, dtype=np.float32).reshape(-1, 4),
            'factors': np.array([2.74, 3.1])}


def pred(t, s, c):
    return {'temporal': np.asarray(t, dtype=np.float32).reshape(-1, 2),
            'spatial': np.asarray(s, dtype=np.float32).reshape(-1, 4),
            'scores': np.asarray(c, dtype=np.float32)}


def main():
    rng = np.random.default_rng(44)
    # Exact agreement with the historical greedy matcher when every clip has GT,
    # including score ties, missing predictions, and spatial/temporal misses.
    for trial in range(10):
        g, p = {}, {}
        for i in range(9):
            n = int(rng.integers(1, 4))
            t = np.sort(rng.random((n, 2)), axis=-1)
            s = rng.random((n, 4))
            g[str(i)] = gt(t, s)
            if i == 8:
                continue
            q = 5
            pt = np.sort(rng.random((q, 2)), axis=-1)
            ps = rng.random((q, 4))
            pt[:n], ps[:n] = t, s
            p[str(i)] = pred(pt, ps, rng.choice([.1, .4, .9], size=q))
        a, b = evaluate(p, g), evaluate_all_clips(p, g)
        np.testing.assert_array_equal(a['AP'], b['AP'])
    print('Historical matcher equivalence (10 randomized fixtures, including ties): PASS')

    g = {'action': gt([[.1, .9]], [[.5, .5, .5, .5]]), 'empty': gt([], [])}
    p = {'action': pred([[.1, .9]], [[.5, .5, .5, .5]], [.8]),
         'empty': pred([[.1, .9]], [[.5, .5, .5, .5]], [.9])}
    assert evaluate(p, g)['mAP'] == 1.
    assert evaluate_all_clips(p, g)['mAP'] == .5
    assert evaluate_all_clips({}, g)['mAP'] == 0.
    assert evaluate_all_clips({'empty': p['empty']}, {'empty': g['empty']})['mAP'] == 0.
    print('Empty-GT high-confidence false positives are penalized: PASS')

    # Fixed strict > / < thresholds must NOT accidentally become >= / <=.
    g = {'x': gt([[0., 1.]], [[0., 0., 0., 0.]])}
    p = {'x': pred([[0., 1.]], [[.5, 0., .5, 0.]], [.8])}
    g['x']['factors'][:] = 2.
    r = evaluate_all_clips(p, g)
    assert r['mAP'] == 0.  # distance == 1.0 fails even the loosest spatial threshold
    p['x']['spatial'][:] = 0.
    p['x']['temporal'][:] = [0., .5]
    r = evaluate_all_clips(p, g)
    assert np.all(r['AP'][-1] == 0.) and np.all(r['AP'][:-1] == 1.)
    print('Threshold inequalities: PASS')

    # A spatial threshold is a COLUMN averaged over all temporal thresholds;
    # temporal .50 is a ROW averaged over spatial thresholds, not a corner.
    from unittest.mock import patch
    matrix = np.arange(100, dtype=float).reshape(10, 10) / 100.
    fixture = dict(mAP=matrix.mean(), mAPs=matrix[0].mean(),
                   mAPt=matrix[:, -1].mean(), AP=matrix)
    with patch('eval_v4.evaluate', return_value=fixture), patch('eval_v4.evaluate_all_clips', return_value=fixture):
        summary = stats('fixture', p, g)
    assert summary['AP_s01'] == matrix[:, 0].mean()
    assert summary['AP_s02'] == matrix[:, 1].mean()
    assert summary['AP_t50'] == matrix[-1].mean()
    assert summary['AP_t05_s01'] == matrix[0, 0]
    assert summary['AP_t50_s10'] == matrix[-1, -1]
    print('Strict metric summaries use complete threshold rows/columns: PASS')

    # Weighted average of f4 per-member means equals the production flat list
    # of model/pass renderings (up to fp32 arithmetic), not mean(obj)*mean(q).
    parts = {}
    components = {}
    for k in ['a', 'b']:
        components[k] = [pred(np.sort(rng.random((5, 2)), -1), rng.random((5, 4)), rng.random(5)) for _ in range(4)]
        parts[k] = {'x': {f: np.mean([v[f] for v in components[k]], axis=0) for f in ['temporal', 'spatial', 'scores']}}
    z = fuse(parts, {'a': 1, 'b': 2})
    flat = components['a'] + components['b'] * 2
    for f in ['temporal', 'spatial', 'scores']:
        np.testing.assert_allclose(z['x'][f], np.mean([v[f] for v in flat], axis=0), atol=1e-7)
    assert all(np.array_equal(fuse(parts, {'a': 1})['x'][f], parts['a']['x'][f]) for f in z['x'])
    print('Weighted ensemble vs production rendering mean, identity: PASS')

    # Clipping makes boundary-padding and averaging noncommutative. Protect
    # v4's choice: pad exactly ONCE after f4 and member fusion.
    parts = {'a': {'x': pred([[0., .5]], [[0., 0., 0., 0.]], [.8])},
             'b': {'x': pred([[.04, 1.]], [[0., 0., 0., 0.]], [.7])}}
    original = parts['a']['x']['temporal'].copy()
    after = padded(fuse(parts, {'a': 1, 'b': 1}))['x']['temporal']
    before = fuse({k: padded(v) for k, v in parts.items()}, {'a': 1, 'b': 1})['x']['temporal']
    assert not np.allclose(after, before)
    np.testing.assert_array_equal(parts['a']['x']['temporal'], original)
    assert np.all(after >= 0.) and np.all(after <= 1.) and np.all(after[:, 0] <= after[:, 1])
    print('Temporal boundary padding order and no mutation: PASS')
    print('ALL CPU EVALUATION CONTRACT TESTS PASSED')


if __name__ == '__main__':
    main()
