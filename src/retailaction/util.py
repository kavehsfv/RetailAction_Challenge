"""Shared constants and helpers (meters-per-normalized-unit factor, IOU, misc)."""
import numpy as np
import torch

# Average real-world bone lengths in meters (dataset card / paper).
BONE_LENGTH_MEANS = {
    ("neck", "top_of_head"): 0.19354,
    ("left_shoulder", "left_elbow"): 0.27096,
    ("left_elbow", "left_wrist"): 0.21228,
    ("right_shoulder", "right_elbow"): 0.27210,
    ("right_elbow", "right_wrist"): 0.21316,
    ("left_hip", "left_knee"): 0.39204,
    ("right_hip", "right_knee"): 0.39266,
    ("left_shoulder", "right_shoulder"): 0.35484,
    ("left_hip", "right_hip"): 0.17150,
    ("neck", "left_shoulder"): 0.18136,
    ("neck", "right_shoulder"): 0.18081,
    ("left_shoulder", "left_hip"): 0.51375,
    ("right_shoulder", "right_hip"): 0.51226,
}
M_PX_FACTOR_AVG = 3.07 / 1000.0  # meters per pixel (dataset card fallback)
FRAME_PX = 600  # all videos are 600x600; only used by the fallback factor
FALLBACK_M_PER_UNIT = M_PX_FACTOR_AVG * FRAME_PX

CLASSES = ["take", "put", "touch"]


def meters_per_unit(poses, min_score=0.3):
    """median over bones of (mean_real_length_m / observed_length_in_normalized_units)."""
    ratios = []
    for fr in poses or []:
        if not isinstance(fr, dict):  # some pose files use a different (non-dict) layout
            continue
        p, s = fr.get("pose", {}), fr.get("joint_scores", {})
        for (a, b), m in BONE_LENGTH_MEANS.items():
            if a in p and b in p and s.get(a, 0) > min_score and s.get(b, 0) > min_score:
                d = float(np.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1]))
                if d > 1e-4:
                    ratios.append(m / d)
    if not ratios:
        return None
    med = float(np.median(ratios))
    return med if (0.5 < med < 10.0) else None


def temporal_iou(p, g):
    """p, g: (..., 2) [start, end]."""
    inter = (torch.min(p[..., 1], g[..., 1]) - torch.max(p[..., 0], g[..., 0])).clamp(min=0)
    union = (torch.max(p[..., 1], g[..., 1]) - torch.min(p[..., 0], g[..., 0])).clamp(min=1e-6)
    return inter / union


def temporal_diou(p, g, eps=1e-8):
    """Interval form of ActionFormer's ctr_diou_loss_1d (idea_07 Idea 2, ported for the
    idea_11 A/B control: cont_ft was trained with diou=True). p, g: (..., 2) [start, end];
    returns DIoU (higher = better), fp32. Center-alignment penalty rho^2/c^2 keeps a
    non-vanishing gradient at IoU=0."""
    p, g = p.float(), g.float()
    inter = (torch.min(p[..., 1], g[..., 1]) - torch.max(p[..., 0], g[..., 0])).clamp(min=0)
    union = (torch.max(p[..., 1], g[..., 1]) - torch.min(p[..., 0], g[..., 0])).clamp(min=eps)
    iou = inter / union
    enclose = (torch.max(p[..., 1], g[..., 1]) - torch.min(p[..., 0], g[..., 0])).clamp(min=eps)
    rho = 0.5 * (p[..., 1] + p[..., 0] - g[..., 1] - g[..., 0])
    return iou - rho.square() / enclose.square()

def temporal_giou(p, g):
    iou = temporal_iou(p, g)
    inter = (torch.min(p[..., 1], g[..., 1]) - torch.max(p[..., 0], g[..., 0])).clamp(min=0)
    union = (torch.max(p[..., 1], g[..., 1]) - torch.min(p[..., 0], g[..., 0])).clamp(min=1e-6)
    enclose = (torch.max(p[..., 1], g[..., 1]) - torch.min(p[..., 0], g[..., 0])).clamp(min=1e-6)
    return iou - (enclose - union) / enclose


def sinusoidal_3d(dim, t, h, w):
    """(1, t*h*w, dim) fixed 3D positional encoding (DETR-style, no learned params)."""
    d = dim // 6
    pos_t = torch.arange(t, dtype=torch.float32)[:, None] / max(t - 1, 1)
    pos_h = torch.arange(h, dtype=torch.float32)[:, None] / max(h - 1, 1)
    pos_w = torch.arange(w, dtype=torch.float32)[:, None] / max(w - 1, 1)
    freq = torch.exp(torch.arange(0, d, dtype=torch.float32) * (-np.log(10000.0) / d))
    enc = lambda pos: torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], dim=-1)
    pe = torch.cat([
        enc(pos_t)[:, None, None, :].expand(t, h, w, -1),
        enc(pos_h)[None, :, None, :].expand(t, h, w, -1),
        enc(pos_w)[None, None, :, :].expand(t, h, w, -1),
    ], dim=-1).reshape(t * h * w, -1)
    pad = dim - pe.shape[-1]
    if pad > 0:
        pe = torch.cat([pe, torch.zeros(pe.shape[0], pad)], -1)
    return pe[..., :dim].unsqueeze(0)
