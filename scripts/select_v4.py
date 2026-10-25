"""Freeze one validation-only choice before any v4 test confirmation.

No test or challenge labels are read. Small gains below .002 are not promoted;
all-clips AP must not regress against either baseline. This is a conservative
shipping gate, not proof of statistical significance.
"""
import argparse
import json
from datetime import datetime, timezone

from eval_v4 import BASE, MEMBERS, V3, SPLITS
from v4_provenance import (INFERENCE, code_identity, dataset_identity, file_hash,
                           verify_frozen)


def select(result, min_gain=.002):
    rows = {r['name']: r for r in result['results']}
    baseline_names = ('v3_common288', 'v3_original_common256')
    baselines = [rows[n] for n in baseline_names]
    candidates = [rows[n] for n in result['specs']]
    best = max(candidates, key=lambda r: r['mAP'])
    gains = {b['name']: best['mAP'] - b['mAP'] for b in baselines}
    safe_empty = all(best['all_clips_mAP'] >= b['all_clips_mAP'] for b in baselines)
    promote = min(gains.values()) >= min_gain and safe_empty
    return best, gains, safe_empty, promote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-gain', type=float, default=.002)
    args = ap.parse_args()
    out = BASE / 'outputs'
    frozen = out / 'selected_config.json'
    if frozen.exists() or (out / 'test_results.json').exists():
        raise RuntimeError('Freeze/test artifacts already exist; refusing post-test reselection')
    path = out / 'validation_results.json'
    r = json.loads(path.read_text())
    if r['split'] != 'val' or r['inference'] != INFERENCE:
        raise RuntimeError('Not a compatible validation result')
    if r['code'] != code_identity():
        raise RuntimeError('Code changed after validation collection')
    if r['dataset'] != dataset_identity(BASE / 'data/raw' / SPLITS['val']):
        raise RuntimeError('Validation data changed after collection')
    best, gains, safe_empty, promote = select(r, args.min_gain)
    decision = dict(candidate=best, gain_vs_baselines=gains, min_gain=args.min_gain,
                    no_all_clips_regression=safe_empty, promote=promote,
                    note='Validation-selected; no public or local-test improvement claim yet')
    (out / 'selection_decision.json').write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)
    if not promote:
        print('RETAIN ORIGINAL v3 submission; promotion gate not met', flush=True)
        return
    weights = r['specs'][best['name']]
    config = dict(schema=1, name=best['name'], weights=weights, inference=INFERENCE,
                  checkpoints=r['checkpoints'], code=r['code'],
                  validation=dict(dataset=r['dataset'], gt_sha256=r['gt_sha256'],
                                  result_sha256=file_hash(path), selection=decision),
                  frozen_utc=datetime.now(timezone.utc).isoformat())
    verify_frozen(config, MEMBERS)
    # Exclusive creation: a single validation-frozen deployment choice.
    with frozen.open('x') as f:
        json.dump(config, f, indent=2)
    print(f'FROZEN {frozen}', flush=True)


if __name__ == '__main__':
    main()
