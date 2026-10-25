"""CPU-only tests for fail-closed cache, selection and submission contracts."""
import copy
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import eval_v4
import v4_provenance as provenance
from select_v4 import select
from submit_v4 import confirm_test, write_csv


def must_fail(fn, text):
    try:
        fn()
    except RuntimeError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f'Expected RuntimeError containing {text}')


def main():
    code = provenance.code_identity()
    assert len(code['sha256']) == 64
    cp = {'path': '/fixture', 'bytes': 10, 'sha256': 'fixture', 'mode': 'ema'}
    config = dict(schema=1, inference=copy.deepcopy(provenance.INFERENCE),
                  code=code, checkpoints={'a_ema': cp}, weights={'a_ema': 1})
    with patch.object(provenance, 'checkpoint_identity', return_value=cp):
        provenance.verify_frozen(config, eval_v4.MEMBERS)
        for field, value, msg in [('weights', {'a_ema': -1}, 'positive integer'),
                                  ('weights', {'unknown': 1}, 'without identity'),
                                  ('code', {}, 'code changed'),
                                  ('inference', {}, 'inference contract')]:
            broken = {**config, field: value}
            must_fail(lambda: provenance.verify_frozen(broken, eval_v4.MEMBERS), msg)
    print('Frozen code/checkpoint/inference/weights checks: PASS')

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / '000000').mkdir()
        j = root / '000000/meta.json'
        j.write_text('{"v": 1}')
        (root / '000000/rank0_video.mp4').write_bytes(b'fixture-video')
        a = provenance.dataset_identity(root)
        j.write_text('{"v": 2}')
        b = provenance.dataset_identity(root)
        assert a['clips'] == b['clips'] == 1 and a['sha256'] != b['sha256']

        gt = {'temporal': np.array([[.1, .9]]), 'spatial': np.full((1, 4), .5),
              'factors': np.array([2.74, 2.74])}
        gts = {'000000': gt}
        pred = {'temporal': np.array([[.1, .9]]), 'spatial': np.full((1, 4), .5),
                'scores': np.array([.8])}
        sig = dict(checkpoint=cp, inference={**provenance.INFERENCE, 'common_size': 288, 'pad_t': 0.},
                   code=code, dataset=b)
        cache = root / 'val_a_ema_common288.pt'
        good = dict(signature=sig, preds={'000000': pred}, gts=gts,
                    gt_sha256=provenance.gts_identity(gts))
        torch.save(good, cache)
        with patch.object(eval_v4, 'checkpoint_identity', return_value=cp):
            load = lambda: eval_v4.cached_member('a_ema', 'val', 288, None, None, root, False, code, b)
            assert load()['gt_sha256'] == good['gt_sha256']
            bad = copy.deepcopy(good)
            bad['signature']['inference']['rank'] = 'quality'
            torch.save(bad, cache)
            must_fail(load, 'Stale cache')
            bad = copy.deepcopy(good)
            bad['gts']['000000']['temporal'][0, 0] = .2
            torch.save(bad, cache)
            must_fail(load, 'Ground-truth identity')
        print('Dataset and GT identities; stale-cache rejection: PASS')

        # Keep sample ordering and require exact unique IDs; reject NaNs.
        sample = root / 'sample.csv'
        ids = [f'{i:06d}' for i in range(2499, -1, -1)]
        with sample.open('w') as f:
            f.write('id,prediction\n' + ''.join(s + ',{}\n' for s in ids))
        pp = {sid: copy.deepcopy(pred) for sid in ids}
        csv_path = root / 'submission.csv'
        assert write_csv(pp, sample, csv_path) == 2500
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert [r['id'] for r in rows] == ids
        assert len(json.loads(rows[0]['prediction'])['actions']) == 1
        pp[ids[0]]['scores'][0] = np.nan
        must_fail(lambda: write_csv(pp, sample, csv_path), 'interval/confidence')
        print('CSV sample order, completeness, NaN rejection: PASS')

    rows = [dict(name='v3_common288', mAP=.85, all_clips_mAP=.84),
            dict(name='v3_original_common256', mAP=.849, all_clips_mAP=.839),
            dict(name='high_dual', mAP=.855, all_clips_mAP=.845)]
    r = dict(results=rows, specs={'high_dual': {}})
    assert select(r)[-1]
    rows[-1]['mAP'] = .851
    assert not select(r)[-1]
    rows[-1]['mAP'], rows[-1]['all_clips_mAP'] = .855, .83
    assert not select(r)[-1]
    rows[-1] = dict(name='FROZEN_v4', mAP=.855, all_clips_mAP=.845)
    confirm_test(rows)
    rows[-1]['mAP'] = .849
    must_fail(lambda: confirm_test(rows), 'did not support promotion')
    print('Validation and one-shot test promotion gates: PASS')
    print('ALL PROVENANCE/SELECTION/CSV TESTS PASSED')


if __name__ == '__main__':
    main()
