"""Independent, CPU-only serialized CSV and frozen-manifest verification."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def unit(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1


def verify(base=BASE):
    manifest_path = base / 'outputs/submission_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    path = Path(manifest['csv'])
    if path.resolve() != (base / 'submission.csv').resolve():
        raise ValueError('Expected the final submission.csv, not a historical artifact')
    if sha256(path) != manifest['sha256']:
        raise ValueError('Submission SHA256 mismatch')
    config_path = base / 'outputs/selected_config.json'
    if sha256(config_path) != manifest['frozen_config_sha256']:
        raise ValueError('Frozen configuration SHA256 mismatch')
    config = json.loads(config_path.read_text())
    if config != manifest['config']:
        raise ValueError('Manifest config differs from frozen validation choice')
    if sha256(base / 'outputs/test_results.json') != manifest['test_result_sha256']:
        raise ValueError('Frozen test result SHA256 mismatch')
    with (base / 'data/raw/sample_submission.csv').open() as f:
        sample_ids = [r['id'] for r in csv.DictReader(f)]
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'prediction']:
            raise ValueError('CSV columns differ from challenge format')
        rows = list(reader)
    ids = [r['id'] for r in rows]
    if len(rows) != manifest['rows'] or len(rows) != 2500 or len(set(ids)) != 2500 or ids != sample_ids:
        raise ValueError('Rows/unique IDs/sample order mismatch')
    total = 0
    for r in rows:
        actions = json.loads(r['prediction'])['actions']
        if len(actions) != config['inference']['max_queries']:
            raise ValueError(f"Unexpected query count: {r['id']}")
        previous = 1.
        for action in actions:
            start, end, score = (action[k] for k in ('start', 'end', 'score'))
            if not all(unit(x) for x in (start, end, score)) or start > end or score > previous:
                raise ValueError(f"Invalid interval/score: {r['id']}")
            previous = score
            for view in ('rank0', 'rank1'):
                point = action['spatial']['action_cam'][view]
                if not all(unit(point[k]) for k in ('x', 'y')):
                    raise ValueError(f"Invalid coordinates: {r['id']}")
            total += 1
    return dict(rows=len(rows), actions=total, sample_order=True, sha256=manifest['sha256'],
                frozen_choice=config['name'], csv=str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=BASE)
    args = ap.parse_args()
    result = verify(args.base)
    (args.base / 'outputs/submission_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
