"""STMixer + PointTAD + MultiTSF + MV-TAP + Camera-Pose-AE additions for RetailAction (combine_ideas_v3).

Implements:
1. idea_11 Idea 1: binary existence head `obj_head` + matched-only CE.
2. idea_11 Idea 4: per-layer decoder supervision with FRESH Hungarian re-matching,
   AND `decoder.norm` applied to intermediate layers so shared heads see consistent scale.
3. idea_12/17/22: runtime `thw = tuple(c.shape[2:])` in `VideoBackbone.forward`
   (unlocks multi-scale training/TTA at 256/288 px and arbitrary frame counts).
4. idea_05 Idea 1: `dense_head` per-frame multi-label auxiliary supervision on encoder tokens.
5. idea_23 Idea 1: `prior_head` per-frame pose-proximity distribution supervision on encoder tokens.
6. idea_24 Idea 1 + All-Query IoU Head (`iou_head`): predicts continuous 10x10 metric quality of EACH
   query slot's own `(temporal, spatial)` box conditioned on `[h, posenc(start, end, x0, y0, x1, y1)]`.
7. idea_19 Idea 1 (`use_pose_tokens`): encodes 32-frame 2-view hand trajectories `(B, 2, 2, 32, 6)` into
   128 pose tokens appended to decoder memory with a zero-initialized residual gate so warm-start is exact.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import (
    MViT_V1_B_Weights,
    MViT_V2_S_Weights,
    mvit_v1_b,
    mvit_v2_s,
)

from .util import sinusoidal_3d

BACKBONES = {"mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights), "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights)}


def tuple_posenc(est6, n_freq=8):
    """Fourier positional encoding of 6-D normalized tuple [start, end, x0, y0, x1, y1] (idea_24 card __01).

    est6: (..., 6) in [0, 1] -> (..., 6 * (1 + 2 * n_freq)) = (..., 102) for n_freq=8.
    """
    freqs = (2.0 ** torch.arange(n_freq, device=est6.device, dtype=est6.dtype)) * math.pi
    ang = est6.unsqueeze(-1) * freqs  # (..., 6, n_freq)
    pe = torch.cat([est6.unsqueeze(-1), ang.sin(), ang.cos()], dim=-1)
    return pe.flatten(-2)


class HandTrackEncoder(nn.Module):
    """Encodes 32-frame per-view left/right hand trajectories (B, 2, 2, 32, 6) and fuses into query features (idea_19).

    Uses cross-attention from decoder queries `h` (B, Q, D) to the 128 pose tokens (B, 128, D)
    with a zero-initialized residual gate (`self.scale = 0.0`) so at step 0 of fine-tuning,
    the warm-started decoder outputs are preserved to 1e-7!
    """

    def __init__(self, d_model=256, nhead=8, max_t=32, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(54, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.hand_emb = nn.Embedding(2, d_model)
        self.time_emb = nn.Embedding(max_t, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def encode_tokens(self, tracks, view_emb_weight):
        # tracks: (B, 2, 2, T, 6)
        B, V, H, T, _ = tracks.shape
        freqs = (2.0 ** torch.arange(4, device=tracks.device, dtype=tracks.dtype)) * math.pi
        ang = tracks.unsqueeze(-1) * freqs  # (..., 6, 4)
        feat = torch.cat([tracks, ang.sin().flatten(-2), ang.cos().flatten(-2)], dim=-1)  # (..., 54)
        tok = self.mlp(feat)  # (B, 2, 2, T, D)
        t_idx = torch.arange(T, device=tracks.device)
        h_idx = torch.arange(H, device=tracks.device)
        tok = (
            tok
            + view_emb_weight.view(1, V, 1, 1, -1)
            + self.hand_emb(h_idx).view(1, 1, H, 1, -1)
            + self.time_emb(t_idx).view(1, 1, 1, T, -1)
        )
        return tok.reshape(B, V * H * T, -1)

    def forward(self, h, pose_tokens):
        # h: (B, Q, D), pose_tokens: (B, 128, D)
        attn_out, _ = self.attn(h, pose_tokens, pose_tokens, need_weights=False)
        delta = attn_out + self.ffn(h + attn_out)
        return h + self.scale * delta

class VideoBackbone(nn.Module):
    """Pretrained video encoder -> token sequence (B, N, D) + grid (t, h, w).

    combine_ideas_v2/v3: derives `thw = tuple(c.shape[2:])` from the runtime conv_proj output
    (idea_12/17/22 fix) so multi-resolution inputs (e.g. 256^2, 288^2) run cleanly.
    """

    def __init__(self, name="mvit_v2_s", pretrained=True):
        super().__init__()
        ctor, weights = BACKBONES[name]
        self.net = ctor(weights=weights.DEFAULT if pretrained else None)
        self.dim = self.net.norm.normalized_shape[0]

    def forward(self, x, return_fine=False, grad_ckpt=False):
        from torch.utils.checkpoint import checkpoint
        net = self.net
        c = net.conv_proj(x)
        thw = tuple(c.shape[2:])
        x = c.flatten(2).transpose(1, 2)
        x = net.pos_encoding(x)
        levels = {}
        fine, fine_thw = None, None
        for blk in net.blocks:
            if grad_ckpt and self.training and x.requires_grad:
                x, thw = checkpoint(blk, x, thw, use_reentrant=False)
            else:
                x, thw = blk(x, thw)
            if return_fine:
                levels[thw] = x
        if return_fine:
            keys = list(levels)
            if len(keys) >= 2:
                fine_thw = keys[-2]
                fine = levels[fine_thw]
        return net.norm(x), thw, fine, fine_thw

class MultiViewActionDETR(nn.Module):
    """Two clips -> N action queries: (class, [start,end], (x,y) per view).

    combine_ideas_v3 additions:
    - obj_head: Binary existence head (B,Q) over all queries (idea_11 Idea 1).
    - iou_head: Tuple-conditioned all-query continuous quality head (B,Q) (idea_24 + all-query true IoU).
    - use_pose_tokens: 32-frame 2-view hand trajectory tokens appended to decoder memory (idea_19 Idea 1).
    - aux_layers: return per-decoder-layer outputs (normalized via decoder.norm) for re-matched aux losses.
    - dense_aux: PointTAD per-frame multi-label aux head on spatial-mean-pooled encoder tokens (idea_05 Idea 1).
    - prior_aux: MultiTSF per-frame pose-proximity prior head on spatial-mean-pooled encoder tokens (idea_23 Idea 1).
    """

    def __init__(self, backbone="mvit_v2_s", pretrained=True, d_model=256, nhead=8,
                 enc_layers=2, dec_layers=3, nqueries=5, nclasses=4,
                 grid_t=8, grid_hw=7, dropout=0.1, chunk=16,
                 obj_head=True, iou_head=False, use_pose_tokens=False,
                 aux_layers=True, fine_lateral=False,
                 dense_aux=False, prior_aux=False, grad_ckpt=False):
        super().__init__()
        self.backbone = VideoBackbone(backbone, pretrained)
        self.proj = nn.Linear(self.backbone.dim, d_model)
        self.grid_t, self.grid_hw = grid_t, grid_hw
        self.chunk = chunk
        self.grad_ckpt = grad_ckpt
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout,
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)
        self.view_emb = nn.Embedding(2, d_model)
        self.query = nn.Embedding(nqueries, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, 4 * d_model, dropout,
                                               batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.nqueries = nqueries
        self.cls_head = nn.Linear(d_model, nclasses)
        self.spatial_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
        self.time_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))
        self.quality_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.obj_head = nn.Linear(d_model, 1) if obj_head else None
        self.iou_head = (
            nn.Sequential(
                nn.Linear(d_model + 102, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            if iou_head else None
        )
        self.pose_encoder = HandTrackEncoder(d_model=d_model) if use_pose_tokens else None
        self.aux_layers = aux_layers
        self.fine_lateral = fine_lateral
        if fine_lateral:
            self.lat_fine = nn.Linear(384, d_model)
            self.lat_fine_norm = nn.LayerNorm(d_model)
        self.dense_head = (
            nn.Sequential(
                nn.Conv1d(d_model, d_model // 2, 3, padding=1),
                nn.Dropout(dropout),
                nn.Conv1d(d_model // 2, nclasses - 1, 1),
            )
            if dense_aux else None
        )
        self.prior_head = nn.Conv1d(d_model, 1, 1) if prior_aux else None

    def compute_iou_from_h(self, h, temporal=None, spatial=None):
        """Predict all-query continuous metric quality conditioned on query feature `h` and detached 6-D box."""
        if self.iou_head is None:
            return None
        if temporal is None:
            sd = self.time_head(h).sigmoid()
            start, dur = sd[..., 0], sd[..., 1]
            temporal = torch.stack([start, start + (1 - start) * dur], dim=-1)
        if spatial is None:
            spatial = self.spatial_head(h).sigmoid()
        est6 = torch.cat([temporal.detach(), spatial.detach()], dim=-1).float()
        pe6 = tuple_posenc(est6, n_freq=8).to(h.dtype)
        return self.iou_head(torch.cat([h, pe6], dim=-1)).squeeze(-1)

    def _encode(self, clip, fine=False):  # (B,C,T,H,W) -> (B,N,D)
        chunks = clip.split(self.chunk, dim=2) if clip.shape[2] > self.chunk else [clip]
        clss, grids, t_total, fines = [], [], 0, []
        for ch in chunks:
            tok, thw, fine_tok, fine_thw = self.backbone(ch, return_fine=self.fine_lateral, grad_ckpt=self.grad_ckpt)
            t, h, w = thw
            clss.append(tok[:, :1])
            g = tok[:, 1:].view(tok.shape[0], t, h, w, -1).permute(0, 4, 1, 2, 3)  # B,D,t,h,w
            g = F.adaptive_avg_pool3d(g, (t, self.grid_hw, self.grid_hw))
            grids.append(g)
            t_total += t
            if self.fine_lateral and fine_tok is not None:
                tf, hf, wf = fine_thw
                gf = fine_tok.view(tok.shape[0], tf, hf, wf, -1).permute(0, 4, 1, 2, 3)
                gf = F.adaptive_avg_pool3d(gf, (tf, self.grid_hw * 2, self.grid_hw * 2))
                fines.append(gf)
        grid = torch.cat(grids, dim=2)
        cls = torch.stack(clss, dim=1).mean(1)
        toks = [cls, grid.flatten(2).permute(0, 2, 1)]
        if self.fine_lateral and fines:
            gf = torch.cat(fines, dim=2)
            fine_tok = self.lat_fine_norm(self.lat_fine(gf.flatten(2).permute(0, 2, 1)))
            pe_f = sinusoidal_3d(fine_tok.shape[-1], gf.shape[2], self.grid_hw * 2, self.grid_hw * 2).to(fine_tok.device)
            toks.append(fine_tok + pe_f)
        tok = self.proj(torch.cat(toks[:2], dim=1))         # B, 1+N, d_model
        pe = sinusoidal_3d(tok.shape[-1], t_total, self.grid_hw, self.grid_hw).to(tok.device)
        tok = torch.cat([tok[:, :1], tok[:, 1:] + pe], dim=1)
        out = self.encoder(tok)
        if self.fine_lateral and fines:
            return torch.cat([out, torch.cat(toks[2:], dim=1)], dim=1)
        return out

    def forward(self, clips, hand_tracks=None):
        # clips: (B,2,C,T,H,W), optional hand_tracks: (B,2,2,32,6)
        B = clips.shape[0]
        feats = [self._encode(clips[:, k]) + self.view_emb.weight[k] for k in range(2)]
        memory = torch.cat(feats, dim=1)
        pose_mem = None
        if self.pose_encoder is not None and hand_tracks is not None:
            pose_mem = self.pose_encoder.encode_tokens(
                hand_tracks.to(clips.device, dtype=clips.dtype), self.view_emb.weight
            )
        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)
        if self.aux_layers:
            hs = []
            h = q
            for layer in self.decoder.layers:
                h = layer(h, memory)
                h_norm = self.decoder.norm(h) if self.decoder.norm is not None else h
                if pose_mem is not None:
                    h_norm = self.pose_encoder(h_norm, pose_mem)
                hs.append(h_norm)
            h = hs[-1]
        else:
            hs = None
            h = self.decoder(q, memory)
            if pose_mem is not None:
                h = self.pose_encoder(h, pose_mem)
        start_dur = self.time_head(h).sigmoid()
        start, dur = start_dur[..., 0], start_dur[..., 1]
        temporal = torch.stack([start, start + (1 - start) * dur], dim=-1)
        spatial = self.spatial_head(h).sigmoid()
        out = {
            "logits": self.cls_head(h),                       # B,Q,C+1
            "spatial": spatial,                               # B,Q,4
            "temporal": temporal,                             # B,Q,2
            "quality": self.quality_head(h).squeeze(-1),      # B,Q
        }
        if self.obj_head is not None:
            out["obj"] = self.obj_head(h).squeeze(-1)         # B,Q existence logits
        if self.iou_head is not None:
            out["iou"] = self.compute_iou_from_h(h, temporal=temporal, spatial=spatial)
        if hs is not None:
            out["aux"] = hs                                   # list of B,Q,D per decoder layer
        if self.dense_head is not None or self.prior_head is not None:
            hw = self.grid_hw * self.grid_hw
            n_coarse = 1 + self.grid_t * hw
            # spatial mean pool over coarse grid (exclude cls token at index 0 and any fine lateral tokens)
            t_feats = [f[:, 1:n_coarse].view(B, -1, hw, f.shape[-1]).mean(2).transpose(1, 2) for f in feats]  # list of (B, D, T)
            if self.dense_head is not None:
                out["dense"] = torch.stack([self.dense_head(tf).transpose(1, 2) for tf in t_feats], dim=1)    # (B, 2, T, C-1)
            if self.prior_head is not None:
                out["prior"] = torch.stack([self.prior_head(tf).squeeze(1) for tf in t_feats], dim=1)         # (B, 2, T)
        return out
