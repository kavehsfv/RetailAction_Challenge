"""Generate a challenge CSV from the ONE validation-frozen, test-confirmed v4 config.

Collect models sequentially (bounded VRAM), fuse cached f4 means with the frozen
integer weights, then pad exactly once. Equivalent to the flat rendering mean.
No challenge labels are used for selection, ranking, or coordinate refinement.
"""
import argparse
import csv
import json

from eval_v4 import BASE, MEMBERS, V3, cached_member, fuse, padded
from retailaction.data import build_loader
from retailaction.predict import SPLITS, to_prediction_json
from v4_provenance import (INFERENCE, code_identity, dataset_identity, file_hash,
                           verify_frozen)


def confirm_test(rows):
    r = {x['name']: x for x in rows}
    final = r['FROZEN_v4']
    for key in ('v3_common288', 'v3_original_common256'):
        if final['mAP'] <= r[key]['mAP'] or final['all_clips_mAP'] < r[key]['all_clips_mAP']:
            raise RuntimeError(f'Test confirmation did not support promotion against {key}; retain v3')


def write_csv(preds, sample_path, path):
    with open(sample_path) as f:
        ids = [r['id'] for r in csv.DictReader(f)]
    if len(ids) != 2500 or len(set(ids)) != 2500 or set(ids) != set(preds):
        raise RuntimeError('Challenge predictions do not match all 2500 unique sample IDs')
    rows = []
    for sid in ids:
        p = preds[sid]
        payload = to_prediction_json(p['temporal'], p['spatial'], p['scores'], INFERENCE['max_queries'])
        actions = json.loads(payload)['actions']
        if not 0 < len(actions) <= 5:
            raise RuntimeError(f'Invalid number of queries for {sid}')
        prev = 1.
        for a in actions:
            if not 0 <= a['start'] <= a['end'] <= 1 or not 0 <= a['score'] <= prev:
                raise RuntimeError(f'Invalid interval/confidence for {sid}')
            prev = a['score']
            for v in ('rank0', 'rank1'):
                if any(not 0 <= a['spatial']['action_cam'][v][xy] <= 1 for xy in ('x', 'y')):
                    raise RuntimeError(f'Invalid coordinates for {sid}')
        rows.append((sid, payload))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'prediction'])
        w.writerows(rows)
    temporary.replace(path)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2, help='two loaders: <=4 workers total')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out', type=str, default='submission.csv')
    args = ap.parse_args()
    out = BASE / 'outputs'
    config = json.loads((out / 'selected_config.json').read_text())
    verify_frozen(config, MEMBERS)
    confirm_test(json.loads((out / 'test_results.json').read_text()))
    root = BASE / 'data/raw' / SPLITS['challenge']
    dataset, code = dataset_identity(root), code_identity()
    common = dict(frames=INFERENCE['frames'], size=INFERENCE['common_size'],
                  batch=args.batch, workers=args.workers, train=False)
    dl0 = build_loader(str(root), frame_shift=0, **common)
    dl1 = build_loader(str(root), frame_shift=1, **common)
    preds = {}
    for key in config['weights']:
        c = cached_member(key, 'challenge', INFERENCE['common_size'], dl0, dl1, out,
                          True, code, dataset)
        preds[key] = c['preds']
    final = padded(fuse(preds, config['weights']))
    path = BASE / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    count = write_csv(final, BASE / 'data/raw/sample_submission.csv', path)
    manifest = dict(rows=count, csv=str(path), sha256=file_hash(path), dataset=dataset,
                    frozen_config_sha256=file_hash(out / 'selected_config.json'),
                    test_result_sha256=file_hash(out / 'test_results.json'), config=config)
    (out / 'submission_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Wrote and verified {path}: {count} rows; sha256={manifest["sha256"]}', flush=True)


if __name__ == '__main__':
    main()
