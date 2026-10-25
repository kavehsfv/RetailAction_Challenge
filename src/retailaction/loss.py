"""Hungarian matching + DETR-style losses for spatio-temporal action queries (combine_ideas_v3).

Combines:
1. idea_11 (STMixer):
   - Binary existence head `obj` with full-weight BCE over ALL queries.
   - Action CE restricted to Hungarian-matched queries ONLY (`matched_cls_only=True`).
   - Matcher existence cost (`cost_obj=2.0`) added to class cost.
   - Per-layer decoder supervision with FRESH Hungarian re-matching per layer (`weight_aux=0.4`).
2. idea_07 (ActionFormer):
   - Temporal DIoU loss (`temporal_loss="diou"`).
3. idea_20/24 (Multi-HMR 2 / Camera-Pose-AE):
   - Metric-space spatial loss & matcher cost scaled by `factors / 2.74` (`meter_spatial=True`),
     aligning spatial supervision with the physical-meter evaluation thresholds `TAU_S`.
4. idea_12 (TubeR):
   - Upweighted quality BCE on matched queries (`up_matched_q > 1.0`) to counter 1:4 query imbalance.
5. idea_05 (PointTAD):
   - Dense per-frame multi-label BCE supervision (`weight_dense=5.0`) on encoder temporal tokens,
     active even on zero-GT (`n==0`) batches.
6. idea_23 (MultiTSF):
   - Per-frame pose-proximity prior soft-CE supervision (`weight_prior=1.0`) on encoder temporal tokens.
7. combine_ideas_v3 (All-Query True Continuous IoU Head + Log-Extent Smooth-L1):
   - `iou_head`: supervised on ALL queries against `max_{m in GT} [mean(tau_t < tIoU(q, m)) * mean(tau_s > d_min(q, m))]`,
     plus pairwise intra-sample margin ranking loss (`weight_iou_rank`) so the best-localized slot within each clip
     ranks highest!
   - `weight_log_w`: DiGIT log-duration smooth-L1 loss (`idea_14`) for strict temporal thresholds.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .metrics import TAU_S, TAU_T
from .util import FALLBACK_M_PER_UNIT, temporal_diou, temporal_giou, temporal_iou


class HungarianMatcher:
    def __init__(self, cost_class=1.0, cost_spatial=5.0, cost_temporal=5.0, cost_obj=2.0,
                 meter_spatial=False):
        self.w = (cost_class, cost_spatial, cost_temporal)
        self.w_obj = cost_obj
        self.meter_spatial = meter_spatial

    @torch.no_grad()
    def __call__(self, pred, tgt):
        """pred: logits(B,Q,C+1) spatial(B,Q,4) temporal(B,Q,2) [obj(B,Q)]; tgt lists per batch."""
        wc, ws, wt = self.w
        out = []
        use_obj = "obj" in pred and self.w_obj > 0
        for b in range(pred["logits"].shape[0]):
            n = len(tgt["labels"][b])
            if n == 0:
                out.append(([], []))
                continue
            prob = pred["logits"][b].softmax(-1).float()
            c_cls = -prob[:, tgt["labels"][b]]
            if use_obj:
                c_obj = -pred["obj"][b].float().sigmoid()[:, None].expand(-1, n)
                c_cls = c_cls + self.w_obj * c_obj
            p_sp = pred["spatial"][b].float()
            g_sp = tgt["spatial"][b].float()
            if self.meter_spatial and "factors" in tgt:
                fac = tgt["factors"][b].to(p_sp.device).float() / FALLBACK_M_PER_UNIT
                scale = torch.stack([fac[0], fac[0], fac[1], fac[1]])
                c_sp = torch.cdist(p_sp * scale, g_sp * scale, p=1)
            else:
                c_sp = torch.cdist(p_sp, g_sp, p=1)
            c_t = torch.cdist(pred["temporal"][b].float(), tgt["temporal"][b].float(), p=1) - \
                temporal_giou(pred["temporal"][b].float()[:, None], tgt["temporal"][b].float()[None])
            cost = wc * c_cls + ws * c_sp + wt * c_t
            cost = cost.float()
            i, j = linear_sum_assignment(cost.cpu())
            out.append((i, j))
        return out


class Criterion:
    def __init__(self, nclasses=4, weight_class=1.0, weight_spatial=5.0, weight_temporal=5.0,
                 weight_giou=2.0, weight_quality=2.0, empty_weight=0.1,
                 weight_obj=2.0, weight_aux=0.4, aux_layers=True, matched_cls_only=True,
                 temporal_loss="giou", matcher_diou=False,
                 meter_spatial=False, up_matched_q=1.0,
                 weight_dense=0.0, weight_prior=0.0,
                 weight_iou=0.0, weight_iou_rank=0.0, weight_log_w=0.0):
        self.matcher = HungarianMatcher(weight_class, weight_spatial, weight_temporal,
                                        cost_obj=weight_obj, meter_spatial=meter_spatial)
        self.temporal_loss = temporal_loss
        self.wc, self.ws, self.wt, self.wg, self.wq = (weight_class, weight_spatial,
                                                       weight_temporal, weight_giou, weight_quality)
        self.nclasses = nclasses
        self.empty_weight = empty_weight
        self.w_obj = weight_obj
        self.w_aux = weight_aux
        self.aux_layers = aux_layers
        self.matched_cls_only = matched_cls_only
        self.meter_spatial = meter_spatial
        self.up_matched_q = up_matched_q
        self.w_dense = weight_dense
        self.w_prior = weight_prior
        self.w_iou = weight_iou
        self.w_iou_rank = weight_iou_rank
        self.w_log_w = weight_log_w
        self.tau_t = torch.tensor(TAU_T, dtype=torch.float32)
        self.tau_s = torch.tensor(TAU_S, dtype=torch.float32)

    def _aux_encoder_losses(self, pred, tgt, device):
        """Encoder-level auxiliary losses (dense per-frame BCE + pose-proximity prior)."""
        extra = {}
        extra_loss = torch.zeros((), device=device)
        if self.w_dense > 0 and "dense" in pred and "dense" in tgt:
            tgt_dense = torch.stack(tgt["dense"]).to(device).unsqueeze(1).expand_as(pred["dense"])
            loss_dense = F.binary_cross_entropy_with_logits(pred["dense"].float(), tgt_dense.float())
            extra["dense"] = loss_dense
            extra_loss = extra_loss + self.w_dense * loss_dense
        if self.w_prior > 0 and "prior" in pred and "prior_target" in tgt:
            raw = torch.stack(tgt["prior_target"]).to(device).float()  # (B, 2, T)
            tot = raw.sum(-1, keepdim=True)
            valid = (tot.squeeze(-1) > 1e-4)
            if valid.any():
                y = (raw / tot.clamp_min(1e-6))[valid]
                p = pred["prior"].float()[valid]
                loss_prior = -(y * F.log_softmax(p, dim=-1)).sum(-1).mean()
            else:
                loss_prior = pred["prior"].float().sum() * 0.0
            extra["prior"] = loss_prior
            extra_loss = extra_loss + self.w_prior * loss_prior
        return extra_loss, extra

    @torch.no_grad()
    def _compute_all_query_true_q(self, pred, tgt, device):
        """Compute true 10x10 metric quality for EVERY query (B, Q) against the best GT in the clip."""
        B, Q, _ = pred["temporal"].shape
        target_iou = torch.zeros((B, Q), device=device, dtype=torch.float32)
        tt = self.tau_t.to(device)
        ts = self.tau_s.to(device)
        for b in range(B):
            m = len(tgt["temporal"][b])
            if m == 0:
                continue
            pt = pred["temporal"][b].float()      # (Q, 2)
            ps = pred["spatial"][b].float()       # (Q, 4)
            gt = tgt["temporal"][b].to(device).float()  # (M, 2)
            gs = tgt["spatial"][b].to(device).float()   # (M, 4)
            fac = tgt["factors"][b].to(device).float()  # (2,)
            ti = temporal_iou(pt[:, None, :], gt[None, :, :])  # (Q, M)
            d0 = (ps[:, None, 0:2] - gs[None, :, 0:2]).norm(dim=-1) * fac[0]
            d1 = (ps[:, None, 2:4] - gs[None, :, 2:4]).norm(dim=-1) * fac[1]
            dm = torch.minimum(d0, d1)  # (Q, M)
            qt = (ti[:, :, None] > tt[None, None, :]).float().mean(-1)  # (Q, M)
            qs = (dm[:, :, None] < ts[None, None, :]).float().mean(-1)  # (Q, M)
            target_iou[b] = (qt * qs).max(dim=1).values
        return target_iou

    def _iou_losses(self, pred, tgt, device):
        """All-query true quality BCE + intra-clip pairwise ranking loss."""
        target_iou = self._compute_all_query_true_q(pred, tgt, device)
        pred_iou_logits = pred["iou"].float()
        loss_iou = F.binary_cross_entropy_with_logits(pred_iou_logits, target_iou)
        if self.w_iou_rank > 0:
            # Intra-clip pairwise ranking: when slot i has higher true quality than slot j by > 0.02,
            # push pred_iou_logits[b, i] - pred_iou_logits[b, j] > 0 weighted by |target_i - target_j|
            diff_t = target_iou[:, :, None] - target_iou[:, None, :]   # (B, Q, Q)
            diff_p = pred_iou_logits[:, :, None] - pred_iou_logits[:, None, :]
            mask = (diff_t > 0.02).float()
            if mask.sum() > 0:
                # Smooth logistic pairwise ranking loss weighted by quality gap diff_t
                rank_mat = F.softplus(-diff_p) * diff_t * mask
                loss_rank = rank_mat.sum() / mask.sum().clamp_min(1.0)
            else:
                loss_rank = pred_iou_logits.sum() * 0.0
            loss_iou = loss_iou + self.w_iou_rank * loss_rank
        return loss_iou

    def _layer_loss(self, pred, tgt, matched_cls_only=True):
        """Losses for ONE decoder layer's outputs."""
        B, Q, _ = pred["logits"].shape
        device = pred["logits"].device
        w = torch.full((self.nclasses,), self.empty_weight, device=device)
        w[:-1] = 1.0
        match = self.matcher(pred, tgt)
        idx = [(torch.as_tensor(i, dtype=torch.int64, device=device),
                torch.as_tensor(j, dtype=torch.int64, device=device)) for i, j in match]

        has_obj = "obj" in pred
        if has_obj:
            tgt_obj = torch.zeros((B, Q), device=device)
            for b, (i, _) in enumerate(idx):
                if len(i):
                    tgt_obj[b, i] = 1.0
            loss_obj = F.binary_cross_entropy_with_logits(pred["obj"].float(), tgt_obj)

        has_iou = "iou" in pred and pred["iou"] is not None and self.w_iou > 0
        if has_iou:
            loss_iou = self._iou_losses(pred, tgt, device)

        if matched_cls_only:
            rows = [b for b, (i, _) in enumerate(idx) for _ in i]
            sidx = torch.cat([i for i, _ in idx]) if any(len(i) for i, _ in idx) else None
            if sidx is not None and len(sidx):
                bi = torch.as_tensor(rows, dtype=torch.int64, device=device)
                lbl = torch.cat([tgt["labels"][b][j] for b, (_, j) in enumerate(idx) if len(j)]).to(device)
                loss_cls = F.cross_entropy(pred["logits"][bi, sidx], lbl)
            else:
                loss_cls = torch.zeros((), device=device)
        else:
            target_cls = torch.full((B, Q), self.nclasses - 1, dtype=torch.long, device=device)
            for b, (i, j) in enumerate(idx):
                if len(i):
                    target_cls[b, i] = tgt["labels"][b][j]
            loss_cls = F.cross_entropy(pred["logits"].flatten(0, 1), target_cls.flatten(), weight=w)

        n = sum(len(i) for i, _ in idx)
        target_q = torch.zeros((B, Q), device=device)
        if n == 0:
            loss_q = F.binary_cross_entropy_with_logits(pred["quality"].float(), target_q)
            out = {"loss": loss_cls + self.wq * loss_q, "cls": loss_cls, "quality": loss_q,
                   "spatial": torch.zeros((), device=device),
                   "temporal": torch.zeros((), device=device),
                   "giou": torch.zeros((), device=device)}
            if has_obj:
                out["obj"] = loss_obj
                out["loss"] = out["loss"] + self.w_obj * loss_obj
            if has_iou:
                out["iou"] = loss_iou
                out["loss"] = out["loss"] + self.w_iou * loss_iou
            return out

        bi, si, p_sp, p_t, g_sp, g_t, fac = [], [], [], [], [], [], []
        for b, (i, j) in enumerate(idx):
            if len(i) == 0:
                continue
            i = i.to(device)
            bi.append(torch.full_like(i, b))
            si.append(i)
            p_sp.append(pred["spatial"][b, i])
            p_t.append(pred["temporal"][b, i])
            g_sp.append(tgt["spatial"][b][j])
            g_t.append(tgt["temporal"][b][j])
            fac.append(tgt["factors"][b].to(device).expand(len(i), 2))
        bi, si = torch.cat(bi), torch.cat(si)
        p_sp, g_sp, p_t, g_t = torch.cat(p_sp), torch.cat(g_sp), torch.cat(p_t), torch.cat(g_t)
        fac = torch.cat(fac)
        if self.meter_spatial:
            scale = torch.stack([fac[:, 0], fac[:, 0], fac[:, 1], fac[:, 1]], dim=-1) / FALLBACK_M_PER_UNIT
            loss_sp = F.l1_loss(p_sp * scale, g_sp * scale)
        else:
            loss_sp = F.l1_loss(p_sp, g_sp)
        loss_l1 = F.l1_loss(p_t, g_t)
        if self.w_log_w > 0:
            wp = (p_t[:, 1] - p_t[:, 0]).clamp_min(1e-4)
            wg = (g_t[:, 1] - g_t[:, 0]).clamp_min(1e-4)
            loss_l1 = loss_l1 + self.w_log_w * F.smooth_l1_loss(wp.log(), wg.log(), beta=0.2)
        giou_fn = temporal_diou if self.temporal_loss == "diou" else temporal_giou
        loss_giou = 1.0 - giou_fn(p_t, g_t).mean()

        with torch.no_grad():
            d = torch.stack([(p_sp[:, 2 * v:2 * v + 2] - g_sp[:, 2 * v:2 * v + 2]).norm(dim=-1) * fac[:, v]
                             for v in range(2)], dim=1).min(dim=1).values
            t_iou = temporal_iou(p_t, g_t)
            q = ((self.tau_t.to(device)[None] < t_iou[:, None]).float().mean(1) *
                 (self.tau_s.to(device)[None] > d[:, None]).float().mean(1))
        target_q[bi, si] = q
        if self.up_matched_q != 1.0:
            wq_mat = torch.ones_like(target_q)
            wq_mat[bi, si] = self.up_matched_q
            loss_q = F.binary_cross_entropy_with_logits(pred["quality"].float(), target_q, weight=wq_mat)
        else:
            loss_q = F.binary_cross_entropy_with_logits(pred["quality"].float(), target_q)

        loss = (self.wc * loss_cls + self.ws * loss_sp + self.wt * loss_l1 +
                self.wg * loss_giou + self.wq * loss_q)
        if has_obj:
            loss = loss + self.w_obj * loss_obj
        if has_iou:
            loss = loss + self.w_iou * loss_iou
        return {"loss": loss, "cls": loss_cls, "spatial": loss_sp, "temporal": loss_l1,
                "giou": loss_giou, "quality": loss_q,
                **({"obj": loss_obj} if has_obj else {}),
                **({"iou": loss_iou} if has_iou else {})}

    def __call__(self, pred, tgt):
        device = pred["logits"].device
        extra_loss, extra_dict = self._aux_encoder_losses(pred, tgt, device)
        if not self.aux_layers or "aux" not in pred:
            out = self._layer_loss(pred, tgt, matched_cls_only=self.matched_cls_only)
            out["loss"] = out["loss"] + extra_loss
            for k, v in extra_dict.items():
                out[k] = v.detach()
            return out

        total = None
        agg = {}
        for li, h in enumerate(pred["aux"]):
            sp_l = self.heads["spatial"](h).sigmoid()
            tm_l = self.heads["temporal"](h)
            layer_pred = {
                "logits": self.heads["logits"](h), "spatial": sp_l,
                "temporal": tm_l, "quality": self.heads["quality"](h),
            }
            if "obj" in pred and self.heads.get("obj") is not None:
                layer_pred["obj"] = self.heads["obj"](h)
            if "iou" in pred and self.heads.get("iou") is not None:
                layer_pred["iou"] = self.heads["iou"](h, tm_l, sp_l)
            w = 1.0 if li == len(pred["aux"]) - 1 else self.w_aux
            sub = self._layer_loss(layer_pred, tgt, matched_cls_only=self.matched_cls_only)
            total = w * sub["loss"] if total is None else total + w * sub["loss"]
            for k, v in sub.items():
                if k == "loss":
                    continue
                agg[f"{k}_l{li}"] = v.detach()
        for k in ("cls", "spatial", "temporal", "giou", "quality", "obj", "iou"):
            key = f"{k}_l{len(pred['aux']) - 1}"
            if key in agg:
                agg[k] = agg[key]
        for k, v in extra_dict.items():
            agg[k] = v.detach()
        agg["loss"] = total + extra_loss
        return agg

    def attach_heads(self, model):
        def temporal_from_h(h):
            sd = model.time_head(h).sigmoid()
            start, dur = sd[..., 0], sd[..., 1]
            return torch.stack([start, start + (1 - start) * dur], dim=-1)

        self.heads = {
            "logits": model.cls_head,
            "spatial": model.spatial_head,
            "temporal": temporal_from_h,
            "quality": lambda h: model.quality_head(h).squeeze(-1),
            "obj": (lambda h: model.obj_head(h).squeeze(-1)) if model.obj_head is not None else None,
            "iou": (lambda h, tm, sp: model.compute_iou_from_h(h, temporal=tm, spatial=sp))
                   if model.iou_head is not None else None,
        }
        return self
