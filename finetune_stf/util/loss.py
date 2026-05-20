import torch
import torch.nn.functional as F
from torch import nn


class SigLoss(nn.Module):
    def __init__(self, warm_up=False, warm_iter=100, eps=1e-3):
        super().__init__()
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0
        self.eps = eps

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        pred = pred[valid_mask]
        target = target[valid_mask]
        if pred.numel() == 0:
            return pred.sum() * 0.0

        g = torch.log(pred + self.eps) - torch.log(target + self.eps)
        if self.warm_up and self.warm_up_counter < self.warm_iter:
            self.warm_up_counter += 1
            loss = 0.15 * torch.pow(torch.mean(g), 2)
        else:
            loss = 0.15 * torch.pow(torch.mean(g), 2) + torch.var(g)

        return torch.sqrt(torch.clamp(loss, min=0.0))


def build_inverse_depth(depth, valid_mask, eps=1e-6):
    inv_depth = torch.zeros_like(depth)
    safe_depth = depth.clamp_min(eps)
    inv_depth[valid_mask] = 1.0 / safe_depth[valid_mask]
    return inv_depth


def build_training_target(depth, valid_mask, *, target_space="metric_depth", eps=1e-6):
    if target_space == "metric_depth":
        return build_inverse_depth(depth, valid_mask, eps=eps)
    if target_space == "inverse_relative":
        inv_target = torch.zeros_like(depth)
        inv_target[valid_mask] = depth[valid_mask].clamp_min(eps)
        return inv_target
    raise ValueError(f"Unsupported target_space={target_space!r}")


def robust_normalize_target_per_sample(
    target,
    valid_mask,
    min_valid_pixels=128,
    min_scale=1e-3,
):
    """Per-image target normalization using median centering and mean absolute deviation.

    This matches the intended MiDaS-style per-image normalization while keeping the
    statistics detached from the backward graph.
    """
    if target.shape != valid_mask.shape:
        raise ValueError(f"Shape mismatch: target={target.shape}, mask={valid_mask.shape}")

    target_norm = target.clone()
    B = target.shape[0]
    device = target.device
    dtype = target.dtype

    centers = torch.zeros(B, device=device, dtype=dtype)
    scales = torch.ones(B, device=device, dtype=dtype)
    normalized = torch.zeros(B, dtype=torch.bool, device=device)

    with torch.no_grad():
        for b in range(B):
            mb = valid_mask[b]
            if int(mb.sum().item()) < min_valid_pixels:
                continue

            vals = target[b][mb]
            if vals.numel() == 0:
                continue

            center = vals.median()
            scale = (vals - center).abs().mean().clamp_min(min_scale)
            centers[b] = center
            scales[b] = scale
            target_norm[b] = (target[b] - center) / scale
            normalized[b] = True

    return target_norm, {
        "norm_centers": centers,
        "norm_scales": scales,
        "normalized_mask": normalized,
        "normalized_samples": int(normalized.sum().item()),
        "unnormalized_samples": int((~normalized).sum().item()),
    }


def align_prediction_to_inverse_gt(pred_disp, inv_gt, valid_mask, eps=1e-6):
    with torch.no_grad():
        x = pred_disp[valid_mask].reshape(-1, 1)
        y = inv_gt[valid_mask].reshape(-1, 1)
        A = torch.cat([x, torch.ones_like(x)], dim=1)
        solution = torch.linalg.lstsq(A, y).solution
        scale = solution[0, 0]
        shift = solution[1, 0]

        if not torch.isfinite(scale) or not torch.isfinite(shift) or scale.abs() < eps:
            return None

        # Pull GT into prediction space so the gradient does not inherit scale.
        aligned_gt = (inv_gt - shift) / scale

    aligned_mask = valid_mask & torch.isfinite(aligned_gt) & (aligned_gt > eps)
    return pred_disp, aligned_gt, aligned_mask, scale, shift


class AlignedInverseSigLoss(nn.Module):
    def __init__(self, min_valid_pixels_per_sample=128, eps=1e-6):
        super().__init__()
        self.min_valid_pixels_per_sample = int(min_valid_pixels_per_sample)
        self.eps = eps
        self.sigloss = SigLoss(warm_up=False, eps=1e-3)

    def forward(self, pred_disp, depth_gt, valid_mask, target_space="metric_depth"):
        inv_gt = build_training_target(depth_gt, valid_mask, target_space=target_space, eps=self.eps)

        losses = []
        used_samples = 0
        skipped_samples = 0

        for pred_sample, inv_gt_sample, mask_sample in zip(pred_disp, inv_gt, valid_mask):
            if int(mask_sample.sum().item()) < self.min_valid_pixels_per_sample:
                skipped_samples += 1
                continue

            try:
                aligned = align_prediction_to_inverse_gt(
                    pred_sample,
                    inv_gt_sample,
                    mask_sample,
                    eps=self.eps,
                )
            except RuntimeError:
                skipped_samples += 1
                continue

            if aligned is None:
                skipped_samples += 1
                continue

            pred_aligned, aligned_gt, aligned_mask, _, _ = aligned
            if int(aligned_mask.sum().item()) < self.min_valid_pixels_per_sample:
                skipped_samples += 1
                continue

            losses.append(self.sigloss(pred_aligned, aligned_gt, aligned_mask))
            used_samples += 1

        if not losses:
            return pred_disp.sum() * 0.0, {
                "used_samples": 0,
                "skipped_samples": skipped_samples,
            }

        return torch.stack(losses).mean(), {
            "used_samples": used_samples,
            "skipped_samples": skipped_samples,
        }


# =====================================================================
# DAv2 / MiDaS-style SSI + gradient matching loss (decomposed for ablation)
#
# Reference: dav2_loss_notes_v2.md
#   L = L_ssi + lambda * L_grad
#   L_ssi  = (1 / 2M) * sum_i m_i (s d_i + t - d*_i)^2       (MSE, no log)
#   L_grad = sum_k (1 / Mk) [ sum |dx r_k| + sum |dy r_k| ]  (K scales, L1)
#
# Alignment is per-image, wrapped in torch.no_grad() around the lstsq so
# (s, t) are detached. Gradient still flows through pred via `s * pred`.
# =====================================================================


def solve_scale_shift_per_sample(
    pred,
    inv_gt,
    valid_mask,
    min_valid_pixels=128,
    eps=1e-6,
):
    """Per-sample lstsq solving (s * pred + t) ~= inv_gt.

    Returns:
        pred_aligned: (B, H, W) = s * pred + t, broadcasted per sample.
                      (s, t) are detached; gradients flow into `pred` only.
        effective_mask: (B, H, W) bool. False for samples that were skipped
                        (too few valid pixels, lstsq failure, degenerate s).
        stats: dict with used_samples / skipped_samples counts.
    """
    if pred.shape != inv_gt.shape or pred.shape != valid_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred.shape}, inv_gt={inv_gt.shape}, "
            f"mask={valid_mask.shape}"
        )

    B = pred.shape[0]
    device = pred.device
    dtype = pred.dtype
    scales = torch.zeros(B, device=device, dtype=dtype)
    shifts = torch.zeros(B, device=device, dtype=dtype)
    ok = torch.zeros(B, dtype=torch.bool, device=device)

    with torch.no_grad():
        for b in range(B):
            mb = valid_mask[b]
            if int(mb.sum().item()) < min_valid_pixels:
                continue
            x = pred[b][mb].reshape(-1, 1)
            y = inv_gt[b][mb].reshape(-1, 1)
            A = torch.cat([x, torch.ones_like(x)], dim=1)
            try:
                sol = torch.linalg.lstsq(A, y).solution
            except RuntimeError:
                continue
            s = sol[0, 0]
            t = sol[1, 0]
            if not torch.isfinite(s) or not torch.isfinite(t) or s.abs() < eps:
                continue
            scales[b] = s
            shifts[b] = t
            ok[b] = True

    pred_aligned = scales.view(B, 1, 1) * pred + shifts.view(B, 1, 1)
    effective_mask = valid_mask & ok.view(B, 1, 1)
    used = int(ok.sum().item())
    return pred_aligned, effective_mask, {
        "used_samples": used,
        "skipped_samples": B - used,
    }


def _ssi_mse_from_aligned(pred_aligned, inv_gt, effective_mask, min_valid):
    """Per-sample (1/2M) MSE in GT inverse-depth space, mean across valid samples."""
    losses = []
    for b in range(pred_aligned.shape[0]):
        mb = effective_mask[b]
        m = int(mb.sum().item())
        if m < min_valid:
            continue
        diff = pred_aligned[b][mb] - inv_gt[b][mb]
        losses.append(diff.pow(2).sum() / (2.0 * m))
    if not losses:
        return pred_aligned.sum() * 0.0
    return torch.stack(losses).mean()


def _grad_matching_from_aligned(
    pred_aligned,
    inv_gt,
    effective_mask,
    n_scales=4,
    mask_downsample="strict",
    min_valid=1,
):
    """Multi-scale L1 on residual gradients.

    Returns the batch mean of per-sample sums over scales. Each scale term is
    normalized by that sample's own number of valid edge pairs.
    """
    if mask_downsample not in {"strict", "loose"}:
        raise ValueError(f"Unsupported mask_downsample={mask_downsample!r}")

    r = pred_aligned - inv_gt
    mask_f = effective_mask.to(pred_aligned.dtype)
    block_area = 1

    sample_losses = []
    for b in range(pred_aligned.shape[0]):
        if int(effective_mask[b].sum().item()) < min_valid:
            continue

        sample_total = pred_aligned[b].sum() * 0.0
        sample_r = r[b]
        sample_mask = effective_mask[b]
        sample_mask_f = mask_f[b]

        for k in range(n_scales):
            if k == 0:
                rk = sample_r
                mk = sample_mask
            else:
                ksz = 2 ** k
                block_area = ksz * ksz
                if sample_r.shape[-2] < ksz or sample_r.shape[-1] < ksz:
                    break
                pooled_sum = F.avg_pool2d(
                    (sample_r * sample_mask_f).unsqueeze(0).unsqueeze(0),
                    kernel_size=ksz,
                    stride=ksz,
                ).squeeze(0).squeeze(0) * block_area
                pooled_count = F.avg_pool2d(
                    sample_mask_f.unsqueeze(0).unsqueeze(0),
                    kernel_size=ksz,
                    stride=ksz,
                ).squeeze(0).squeeze(0) * block_area
                if mask_downsample == "strict":
                    mk = pooled_count >= (block_area - 0.5)
                else:
                    mk = pooled_count > 0.0
                rk = torch.where(
                    mk,
                    pooled_sum / pooled_count.clamp_min(1.0),
                    torch.zeros_like(pooled_sum),
                )

            diff_x = rk[:, 1:] - rk[:, :-1]
            mask_x = mk[:, 1:] & mk[:, :-1]
            if mask_x.any():
                sample_total = sample_total + (
                    diff_x.abs() * mask_x.to(diff_x.dtype)
                ).sum() / mask_x.sum().to(diff_x.dtype)

            diff_y = rk[1:, :] - rk[:-1, :]
            mask_y = mk[1:, :] & mk[:-1, :]
            if mask_y.any():
                sample_total = sample_total + (
                    diff_y.abs() * mask_y.to(diff_y.dtype)
                ).sum() / mask_y.sum().to(diff_y.dtype)

        sample_losses.append(sample_total)

    if not sample_losses:
        return pred_aligned.sum() * 0.0
    return torch.stack(sample_losses).mean()


class ScaleShiftInvariantLoss(nn.Module):
    """DAv2 / MiDaS SSI data term: per-image lstsq alignment then MSE in inverse-depth space.

    forward(pred_disp, depth_gt, valid_mask) -> (loss, info).
    """

    def __init__(
        self,
        min_valid_pixels_per_sample=128,
        eps=1e-6,
        use_target_normalization=True,
        norm_min_scale=1e-3,
    ):
        super().__init__()
        self.min_valid_pixels_per_sample = int(min_valid_pixels_per_sample)
        self.eps = eps
        self.use_target_normalization = bool(use_target_normalization)
        self.norm_min_scale = float(norm_min_scale)

    def forward(self, pred_disp, depth_gt, valid_mask, target_space="metric_depth"):
        inv_gt = build_training_target(depth_gt, valid_mask, target_space=target_space, eps=self.eps)
        info = {}
        if self.use_target_normalization:
            inv_gt, norm_stats = robust_normalize_target_per_sample(
                inv_gt,
                valid_mask,
                min_valid_pixels=self.min_valid_pixels_per_sample,
                min_scale=self.norm_min_scale,
            )
            normalized_mask = norm_stats["normalized_mask"]
            if normalized_mask.any():
                mean_scale = norm_stats["norm_scales"][normalized_mask].mean()
            else:
                mean_scale = norm_stats["norm_scales"].new_tensor(1.0)
            info["norm_scale_mean"] = float(mean_scale.detach().item())
            info["normalized_samples"] = norm_stats["normalized_samples"]
            info["unnormalized_samples"] = norm_stats["unnormalized_samples"]
        pred_aligned, effective_mask, stats = solve_scale_shift_per_sample(
            pred_disp,
            inv_gt,
            valid_mask,
            min_valid_pixels=self.min_valid_pixels_per_sample,
            eps=self.eps,
        )
        info.update(stats)
        if stats["used_samples"] == 0:
            return pred_disp.sum() * 0.0, info
        loss = _ssi_mse_from_aligned(
            pred_aligned, inv_gt, effective_mask, self.min_valid_pixels_per_sample
        )
        return loss, info


class GradientMatchingLoss(nn.Module):
    """DAv2 / MiDaS gradient matching loss on aligned residual, K scales.

    Does its own per-image alignment so it can be used standalone. For efficiency
    when paired with ScaleShiftInvariantLoss, use DAv2RelativeLoss which shares
    the alignment step.
    """

    def __init__(
        self,
        n_scales=4,
        min_valid_pixels_per_sample=128,
        mask_downsample="strict",
        eps=1e-6,
        use_target_normalization=True,
        norm_min_scale=1e-3,
    ):
        super().__init__()
        self.n_scales = int(n_scales)
        self.min_valid_pixels_per_sample = int(min_valid_pixels_per_sample)
        self.mask_downsample = mask_downsample
        self.eps = eps
        self.use_target_normalization = bool(use_target_normalization)
        self.norm_min_scale = float(norm_min_scale)

    def forward(self, pred_disp, depth_gt, valid_mask, target_space="metric_depth"):
        inv_gt = build_training_target(depth_gt, valid_mask, target_space=target_space, eps=self.eps)
        info = {}
        if self.use_target_normalization:
            inv_gt, norm_stats = robust_normalize_target_per_sample(
                inv_gt,
                valid_mask,
                min_valid_pixels=self.min_valid_pixels_per_sample,
                min_scale=self.norm_min_scale,
            )
            normalized_mask = norm_stats["normalized_mask"]
            if normalized_mask.any():
                mean_scale = norm_stats["norm_scales"][normalized_mask].mean()
            else:
                mean_scale = norm_stats["norm_scales"].new_tensor(1.0)
            info["norm_scale_mean"] = float(mean_scale.detach().item())
            info["normalized_samples"] = norm_stats["normalized_samples"]
            info["unnormalized_samples"] = norm_stats["unnormalized_samples"]
        pred_aligned, effective_mask, stats = solve_scale_shift_per_sample(
            pred_disp,
            inv_gt,
            valid_mask,
            min_valid_pixels=self.min_valid_pixels_per_sample,
            eps=self.eps,
        )
        info.update(stats)
        if stats["used_samples"] == 0:
            return pred_disp.sum() * 0.0, info
        loss = _grad_matching_from_aligned(
            pred_aligned,
            inv_gt,
            effective_mask,
            n_scales=self.n_scales,
            mask_downsample=self.mask_downsample,
            min_valid=self.min_valid_pixels_per_sample,
        )
        return loss, info


class DAv2RelativeLoss(nn.Module):
    """Combined SSI + lambda * gradient-matching loss with shared per-image alignment.

    Toggles `use_ssi` / `use_grad` for ablation. `lambda_grad` follows the DAv2
    paper recommendation of 2.0.
    """

    def __init__(
        self,
        lambda_grad=2.0,
        n_scales=4,
        use_ssi=True,
        use_grad=True,
        min_valid_pixels_per_sample=128,
        mask_downsample="strict",
        eps=1e-6,
        use_target_normalization=True,
        norm_min_scale=1e-3,
    ):
        super().__init__()
        if not (use_ssi or use_grad):
            raise ValueError("DAv2RelativeLoss requires at least one of use_ssi/use_grad to be enabled.")
        if mask_downsample not in {"strict", "loose"}:
            raise ValueError(f"Unsupported mask_downsample={mask_downsample!r}")
        self.lambda_grad = float(lambda_grad)
        self.n_scales = int(n_scales)
        self.use_ssi = bool(use_ssi)
        self.use_grad = bool(use_grad)
        self.min_valid_pixels_per_sample = int(min_valid_pixels_per_sample)
        self.mask_downsample = mask_downsample
        self.eps = eps
        self.use_target_normalization = bool(use_target_normalization)
        self.norm_min_scale = float(norm_min_scale)

    def forward(self, pred_disp, depth_gt, valid_mask, target_space="metric_depth"):
        inv_gt = build_training_target(depth_gt, valid_mask, target_space=target_space, eps=self.eps)
        info = {}
        if self.use_target_normalization:
            inv_gt, norm_stats = robust_normalize_target_per_sample(
                inv_gt,
                valid_mask,
                min_valid_pixels=self.min_valid_pixels_per_sample,
                min_scale=self.norm_min_scale,
            )
            normalized_mask = norm_stats["normalized_mask"]
            if normalized_mask.any():
                mean_scale = norm_stats["norm_scales"][normalized_mask].mean()
            else:
                mean_scale = norm_stats["norm_scales"].new_tensor(1.0)
            info["norm_scale_mean"] = float(mean_scale.detach().item())
            info["normalized_samples"] = norm_stats["normalized_samples"]
            info["unnormalized_samples"] = norm_stats["unnormalized_samples"]
        pred_aligned, effective_mask, stats = solve_scale_shift_per_sample(
            pred_disp,
            inv_gt,
            valid_mask,
            min_valid_pixels=self.min_valid_pixels_per_sample,
            eps=self.eps,
        )
        info.update(stats)
        if stats["used_samples"] == 0:
            return pred_disp.sum() * 0.0, info

        loss = pred_disp.sum() * 0.0
        if self.use_ssi:
            l_ssi = _ssi_mse_from_aligned(
                pred_aligned, inv_gt, effective_mask, self.min_valid_pixels_per_sample
            )
            info["loss_ssi"] = float(l_ssi.detach().item())
            loss = loss + l_ssi
        if self.use_grad:
            l_grad = _grad_matching_from_aligned(
                pred_aligned,
                inv_gt,
                effective_mask,
                n_scales=self.n_scales,
                mask_downsample=self.mask_downsample,
                min_valid=self.min_valid_pixels_per_sample,
            )
            info["loss_grad"] = float(l_grad.detach().item())
            loss = loss + self.lambda_grad * l_grad
        return loss, info
