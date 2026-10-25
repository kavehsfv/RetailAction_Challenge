"""Dataset + decoding for RetailAction (two top-view clips + spatio-temporal action labels).

combine_ideas_v3 additions:
- dense per-frame multi-label targets (idea_05 PointTAD Idea 1)
- pose-proximity per-frame prior targets (idea_23 MultiTSF Idea 1, window-free hand proximity)
- explicit 32-frame pose hand-track extraction `(2, 2, 32, 6)` (idea_19 MV-TAP Idea 1: `[x, y, vx, vy, score, obs]`)
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from .metrics import TAU_S
from .util import CLASSES, FALLBACK_M_PER_UNIT, meters_per_unit

MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)
VIEWS = ("rank0", "rank1")
HANDS = ("left_hand", "right_hand", "left_wrist", "right_wrist")
HAND_PAIRS = (("left_hand", "left_wrist"), ("right_hand", "right_wrist"))


def load_sample(sample_dir):
    """-> dict(actions=[{start,end,label,spatial:{view:{x,y}}}], poses, has_labels, seg)"""
    md = json.load(open(os.path.join(sample_dir, "metadata.json")))
    c = md.get("content", md)
    out = {"actions": [], "poses": {}, "has_labels": "labels" in c, "segment_info": c.get("segment_info")}
    if out["has_labels"]:
        out["actions"] = c["labels"].get("action") or []
    for v in VIEWS:
        cam = (c.get("action_cam") or {}).get(v) or {}
        poses = cam.get("poses")
        if not poses:
            pf = os.path.join(sample_dir, f"{v}_video_poses.json")
            if os.path.exists(pf):
                poses = json.load(open(pf))
        out["poses"][v] = poses or []
    return out


def extract_hand_tracks(poses_by_view, flip=None, T=32, min_score=0.3):
    """Extract 32-frame hand trajectories per view (idea_19 MV-TAP Idea 1).

    Returns tensor of shape (2, 2, T, 6) for (view, hand={left,right}, time, [x, y, vx, vy, score, obs]).
    Unobserved frames are linearly interpolated in (x, y) and marked with obs=0.
    """
    tracks = np.zeros((2, 2, T, 6), dtype=np.float32)
    for k, v in enumerate(VIEWS):
        poses_v = poses_by_view.get(v) or []
        is_flip = bool(flip[k]) if flip is not None else False
        for h_idx, (kp_primary, kp_fallback) in enumerate(HAND_PAIRS):
            # If horizontally flipped, left/right image coordinates mirror x -> 1 - x
            xs = np.full(T, np.nan, dtype=np.float32)
            ys = np.full(T, np.nan, dtype=np.float32)
            scs = np.zeros(T, dtype=np.float32)
            obs = np.zeros(T, dtype=np.float32)
            for t in range(min(T, len(poses_v))):
                fr = poses_v[t]
                if not isinstance(fr, dict):
                    continue
                p_dict = fr.get("pose") or {}
                sc_dict = fr.get("joint_scores") or {}
                chosen_xy, chosen_sc = None, 0.0
                for kp in (kp_primary, kp_fallback):
                    xy = p_dict.get(kp)
                    sc = float(sc_dict.get(kp, 0.0) or 0.0)
                    if xy is not None and len(xy) >= 2 and sc >= min_score:
                        chosen_xy = (float(xy[0]), float(xy[1]))
                        chosen_sc = sc
                        break
                if chosen_xy is not None:
                    x_val = 1.0 - chosen_xy[0] if is_flip else chosen_xy[0]
                    xs[t] = np.clip(x_val, 0.0, 1.0)
                    ys[t] = np.clip(chosen_xy[1], 0.0, 1.0)
                    scs[t] = np.clip(chosen_sc, 0.0, 1.0)
                    obs[t] = 1.0
            valid = np.where(obs > 0.5)[0]
            if len(valid) == 0:
                xs[:] = 0.5
                ys[:] = 0.5
            elif len(valid) == 1:
                xs[:] = xs[valid[0]]
                ys[:] = ys[valid[0]]
            else:
                t_grid = np.arange(T, dtype=np.float32)
                xs = np.interp(t_grid, valid.astype(np.float32), xs[valid]).astype(np.float32)
                ys = np.interp(t_grid, valid.astype(np.float32), ys[valid]).astype(np.float32)
            vx = np.zeros(T, dtype=np.float32)
            vy = np.zeros(T, dtype=np.float32)
            vx[1:] = xs[1:] - xs[:-1]
            vy[1:] = ys[1:] - ys[:-1]
            tracks[k, h_idx, :, 0] = xs
            tracks[k, h_idx, :, 1] = ys
            tracks[k, h_idx, :, 2] = np.clip(vx * 5.0, -1.0, 1.0)
            tracks[k, h_idx, :, 3] = np.clip(vy * 5.0, -1.0, 1.0)
            tracks[k, h_idx, :, 4] = scs
            tracks[k, h_idx, :, 5] = obs
    return torch.from_numpy(tracks)


class RetailActionDataset(torch.utils.data.Dataset):
    """Yields (video[2,C,T,H,W], target) with normalized [0,1] coordinates."""

    def __init__(self, root, frames=16, size=224, train=True, limit=None, ids=None, frame_shift=0):
        self.root, self.frames, self.size, self.train = root, frames, size, train
        self.frame_shift = frame_shift   # idea_08: complementary sampling pass (grid+1 covers
                                         # the frames the production grid skips)
        self.ids = ids if ids is not None else sorted(os.listdir(root))
        self.ids = [i for i in self.ids if os.path.isdir(os.path.join(root, i))]
        if limit:
            self.ids = self.ids[:limit]

    def __len__(self):
        return len(self.ids)

    def _read_clip(self, path):
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        n = len(vr)
        idx = np.linspace(0, max(n - 1, 1), self.frames).round().astype(int)  # repeats if clip < frames
        if self.frame_shift:
            idx = np.clip(idx + self.frame_shift, 0, max(n - 1, 0))   # idea_08 pass B: grid+1
        buf = vr.get_batch(idx).asnumpy()  # T,H,W,3 uint8
        x = torch.from_numpy(buf).permute(0, 3, 1, 2).float().div(255.0)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
        return x.permute(1, 0, 2, 3).contiguous(), idx  # C,T,H,W and sampled frame indices

    def __getitem__(self, i):
        sid = self.ids[i]
        d = os.path.join(self.root, sid)
        s = load_sample(d)
        clips, idxs = zip(*[self._read_clip(os.path.join(d, f"{v}_video.mp4")) for v in VIEWS])
        vid = torch.stack(clips)  # 2,C,T,H,W

        flip = torch.zeros(2, dtype=torch.bool)
        if self.train:
            flip = torch.rand(2) < 0.5
            for k in range(2):
                if flip[k]:
                    vid[k] = vid[k].flip(-1)

        n = len(s["actions"])
        spatial = torch.zeros(n, 4)
        temporal = torch.zeros(n, 2)
        labels = torch.zeros(n, dtype=torch.long)
        for j, a in enumerate(s["actions"]):
            for k, v in enumerate(VIEWS):
                p = a["spatial"]["action_cam"][v]
                x = p["x"]
                if flip[k]:
                    x = 1.0 - x
                spatial[j, 2 * k], spatial[j, 2 * k + 1] = x, p["y"]
            temporal[j, 0], temporal[j, 1] = a["start"], a["end"]
            labels[j] = CLASSES.index(a["label"]) if a.get("label") in CLASSES else 0

        factors = torch.tensor(
            [meters_per_unit(s["poses"][v]) or FALLBACK_M_PER_UNIT for v in VIEWS], dtype=torch.float32
        )

        # ---- idea_05 PointTAD Idea 1: dense per-frame multi-label target (T=frames//2=8, C=3) ----
        T = self.frames // 2
        dense = torch.zeros(T, len(CLASSES))
        for j in range(n):
            s0 = max(int(float(temporal[j, 0]) * T), 0)
            e0 = min(int(float(temporal[j, 1]) * T), T - 1)
            dense[s0:e0 + 1, labels[j]] = 1.0

        # ---- idea_23 MultiTSF Idea 1: window-free pose-proximity prior (V, T=8) ----
        T32 = 32
        kap = np.zeros((2, T32), dtype=np.float32)
        for k, v in enumerate(VIEWS):
            fac = float(factors[k])
            poses_v = s["poses"][v][:T32]
            for a in s["actions"]:
                gt = a["spatial"]["action_cam"][v]
                gx, gy = float(gt["x"]), float(gt["y"])
                for t_idx, fr in enumerate(poses_v):
                    if not isinstance(fr, dict):
                        continue
                    p_dict = fr.get("pose") or {}
                    sc_dict = fr.get("joint_scores") or {}
                    dists = [
                        np.hypot(p_dict[h][0] - gx, p_dict[h][1] - gy) * fac
                        for h in HANDS
                        if h in p_dict and p_dict[h] is not None and sc_dict.get(h, 0) > 0.3
                    ]
                    if dists:
                        kap[k, t_idx] = max(kap[k, t_idx], float((TAU_S > min(dists)).mean()))
        prior_list = []
        for k in range(2):
            idx_k = np.clip(idxs[k], 0, T32 - 1)
            pr_k = kap[k, idx_k].reshape(T, 2).mean(-1)
            prior_list.append(pr_k)
        prior_target = torch.from_numpy(np.stack(prior_list, axis=0))  # (2, 8)

        # ---- idea_19 MV-TAP Idea 1: full 32-frame hand trajectories (2, 2, 32, 6) ----
        hand_tracks = extract_hand_tracks(s["poses"], flip=flip, T=T32)

        return vid, {
            "labels": labels,
            "spatial": spatial,
            "temporal": temporal,
            "factors": factors,
            "dense": dense,
            "prior_target": prior_target,
            "hand_tracks": hand_tracks,
            "id": sid,
            "flip": flip,
        }


def collate(batch):
    vids = torch.stack([b[0] for b in batch])
    keys = ("labels", "spatial", "temporal", "factors", "dense", "prior_target", "hand_tracks", "id", "flip")
    tgt = {k: [b[1][k] for b in batch] for k in keys}
    return vids, tgt


def build_loader(root, frames=16, size=224, train=True, batch=4, workers=8, limit=None, ids=None,
                 shuffle=None, frame_shift=0):
    ds = RetailActionDataset(root, frames=frames, size=size, train=train, limit=limit, ids=ids,
                             frame_shift=frame_shift)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=(train if shuffle is None else shuffle),
        num_workers=workers, collate_fn=collate, pin_memory=True, drop_last=train,
        persistent_workers=workers > 0,
    )
