"""Fail-closed identities for v4 inference, caches, and validation-frozen configs.

Video identities use relative paths/size/mtime (not full content hashes); metadata,
pose JSON, checkpoints, and inference code are SHA256 content-addressed.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
INFERENCE = dict(common_size=288, frames=16, passes=['s0', 's1', 'flip', 'flip1'],
                 rank='obj*q', pad_t=.015, padding_stage='after_ensemble',
                 scales=None, dedup=None, autocast='bfloat16', max_queries=5)
CODE_FILES = [f'src/retailaction/{x}.py' for x in
              ('data', 'model', 'util', 'engine', 'fuse', 'postprocess', 'metrics', 'predict')]
CODE_FILES += ['scripts/eval_v4.py', 'scripts/v4_provenance.py',
               'scripts/select_v4.py', 'scripts/submit_v4.py']


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=64)
def _file_hash(path, size, mtime):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    st = Path(path).stat()
    if (st.st_size, st.st_mtime_ns) != (size, mtime):
        raise RuntimeError(f'File changed while hashing: {path}')
    return h.hexdigest()


def file_hash(path):
    path = Path(path).resolve()
    st = path.stat()
    return _file_hash(str(path), st.st_size, st.st_mtime_ns)


def checkpoint_identity(path, mode):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_hash(path), mode=mode)


def code_identity():
    files = {p: file_hash(BASE / p) for p in CODE_FILES}
    return dict(sha256=json_digest(files), files=files)


def dataset_identity(root):
    root = Path(root).resolve()
    entries = []
    sids = sorted(p.name for p in root.iterdir() if p.is_dir())
    for sid in sids:
        for p in sorted((root / sid).iterdir()):
            if not p.is_file() or p.suffix not in ('.mp4', '.json'):
                continue
            st = p.stat()
            row = dict(path=f'{sid}/{p.name}', bytes=st.st_size)
            if p.suffix == '.json':
                row['sha256'] = file_hash(p)
            else:
                row['mtime_ns'] = st.st_mtime_ns
            entries.append(row)
    return dict(root=str(root), clips=len(sids), files=len(entries), sha256=json_digest(entries),
                policy='JSON content SHA256; video size+mtime; sorted relative paths')


def gts_identity(gts):
    h = hashlib.sha256()
    for sid in sorted(gts):
        h.update(sid.encode() + b'\0')
        for field in ('temporal', 'spatial', 'factors'):
            a = np.asarray(gts[sid][field], dtype=np.float64)
            h.update(field.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def verify_frozen(config, members):
    if config.get('schema') != 1 or config.get('inference') != INFERENCE:
        raise RuntimeError('Frozen inference contract missing or different')
    if config['code'] != code_identity():
        raise RuntimeError('Inference/evaluation code changed after validation freeze')
    for key, expected in config['checkpoints'].items():
        tag, mode = members[key]
        actual = checkpoint_identity(BASE / f'src/outputs/{tag}/best.pt', mode)
        if actual != expected:
            raise RuntimeError(f'Frozen checkpoint changed: {key}')
    if any(k not in config['checkpoints'] for k in config['weights']):
        raise RuntimeError('Frozen weights reference a checkpoint without identity')
    weights = list(config['weights'].values())
    if not weights or any(not isinstance(w, int) or w <= 0 for w in weights):
        raise RuntimeError('Deployment requires positive integer rendering weights')
