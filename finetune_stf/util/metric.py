import torch
import numpy as np


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25**2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25**3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {
        "d1": d1.item(),
        "d2": d2.item(),
        "d3": d3.item(),
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "log10": log10.item(),
        "silog": silog.item(),
    }


__all__ = ["eval_depth"]


def affine_align_to_inverse_target(pred_inverse, target_inverse, valid_mask, min_positive=1e-6):
    pred = np.asarray(pred_inverse, dtype=np.float64)
    target = np.asarray(target_inverse, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if pred.shape != target.shape or pred.shape != valid.shape:
        raise ValueError(
            f"Expected pred/target/mask shape match, got pred={pred.shape} target={target.shape} mask={valid.shape}"
        )

    valid = valid & np.isfinite(pred) & np.isfinite(target) & (target > 0)
    if int(valid.sum()) < 2:
        aligned = np.full(pred.shape, np.nan, dtype=np.float64)
        return aligned, {"scale": 0.0, "shift": 0.0, "valid_pixels": int(valid.sum())}

    x = pred[valid]
    y = target[valid]
    design = np.stack([x, np.ones_like(x)], axis=-1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    scale, shift = float(coef[0]), float(coef[1])

    aligned = pred * scale + shift
    aligned = np.where(np.isfinite(aligned), aligned, np.nan)
    aligned = np.where(aligned > 0, aligned, min_positive)
    return aligned, {"scale": scale, "shift": shift, "valid_pixels": int(valid.sum())}


def compute_inverse_relative_metrics(pred_inverse, target_inverse, valid_mask, min_positive=1e-6):
    pred = np.asarray(pred_inverse, dtype=np.float64)
    target = np.asarray(target_inverse, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    valid = valid & np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
    if int(valid.sum()) == 0:
        return None

    pred = np.clip(pred[valid], min_positive, None)
    target = np.clip(target[valid], min_positive, None)
    thresh = np.maximum(target / pred, pred / target)
    diff = pred - target
    diff_log = np.log(pred) - np.log(target)
    silog_var = np.mean(diff_log**2) - 0.5 * np.mean(diff_log) ** 2
    return {
        "abs_rel": float(np.mean(np.abs(diff) / target)),
        "sq_rel": float(np.mean(diff**2 / target)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(np.sqrt(np.mean(diff_log**2))),
        "log10": float(np.mean(np.abs(np.log10(pred) - np.log10(target)))),
        "silog": float(np.sqrt(max(silog_var, 0.0))),
        "silog_x100": float(100.0 * np.sqrt(max(silog_var, 0.0))),
        "d1": float(np.mean(thresh < 1.25)),
        "d2": float(np.mean(thresh < 1.25**2)),
        "d3": float(np.mean(thresh < 1.25**3)),
    }


__all__ = ["eval_depth", "affine_align_to_inverse_target", "compute_inverse_relative_metrics"]
