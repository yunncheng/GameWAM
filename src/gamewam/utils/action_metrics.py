import json
from typing import Optional, Sequence

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_action_btd(raw_action, action_name: str) -> torch.Tensor:
    if not isinstance(raw_action, torch.Tensor):
        raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
    if raw_action.ndim == 2:
        action_btd = raw_action.unsqueeze(0)
    elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
        action_btd = raw_action
    else:
        raise ValueError(
            f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
        )
    return action_btd.detach().to(device="cpu", dtype=torch.float32)


def tensor_stats_1d(x: torch.Tensor) -> dict:
    return {
        "mean": x.mean(dim=0).tolist(),
        "std": x.std(dim=0, unbiased=False).tolist(),
        "min": x.min(dim=0).values.tolist(),
        "max": x.max(dim=0).values.tolist(),
    }


def detect_binary_like_dims(gt_action: torch.Tensor) -> list[int]:
    gt = gt_action.reshape(-1, gt_action.shape[-1])
    if gt.numel() == 0:
        return []
    near_binary = torch.minimum(gt.abs(), (gt - 1.0).abs()).mean(dim=0) <= 0.05
    in_range = (gt.min(dim=0).values >= -0.05) & (gt.max(dim=0).values <= 1.05)
    return (near_binary & in_range).nonzero().flatten().tolist()


def resolve_action_dims(
    gt_action: torch.Tensor,
    dim_is_continuous: Optional[Sequence[bool]] = None,
) -> tuple[list[int], list[int]]:
    dim_count = int(gt_action.shape[-1])
    if dim_is_continuous is None:
        discrete_dims = detect_binary_like_dims(gt_action)
        discrete_set = set(discrete_dims)
        continuous_dims = [i for i in range(dim_count) if i not in discrete_set]
        return continuous_dims, discrete_dims

    flags = [bool(value) for value in dim_is_continuous]
    if len(flags) != dim_count:
        raise ValueError(
            "dim_is_continuous length must match action dim: "
            f"got {len(flags)} vs {dim_count}"
        )
    continuous_dims = [i for i, value in enumerate(flags) if value]
    discrete_dims = [i for i, value in enumerate(flags) if not value]
    return continuous_dims, discrete_dims


def action_metric_summary(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    *,
    dim_is_continuous: Optional[Sequence[bool]] = None,
) -> dict:
    if pred_action.shape != gt_action.shape:
        raise ValueError(
            "Predicted action/GT action shape mismatch: "
            f"pred={tuple(pred_action.shape)} vs gt={tuple(gt_action.shape)}"
        )
    pred = pred_action.reshape(-1, pred_action.shape[-1])
    gt = gt_action.reshape(-1, gt_action.shape[-1])
    diff = pred - gt
    abs_diff = diff.abs()
    sq_diff = diff.pow(2)
    continuous_dims, discrete_dims = resolve_action_dims(
        gt_action, dim_is_continuous
    )

    summary = {
        "shape": list(pred_action.shape),
        "action_l1": float(abs_diff.mean().item()),
        "action_l2": float(sq_diff.mean().item()),
        "action_rmse": float(sq_diff.mean().sqrt().item()),
        "pred": tensor_stats_1d(pred),
        "gt": tensor_stats_1d(gt),
        "l1_per_dim": abs_diff.mean(dim=0).tolist(),
        "l2_per_dim": sq_diff.mean(dim=0).tolist(),
        "continuous_dims": [int(i) for i in continuous_dims],
        "discrete_dims": [int(i) for i in discrete_dims],
        # ``binary_like_dims`` mirrors the configured discrete dimensions in
        # the metrics schema.
        "binary_like_dims": [int(i) for i in discrete_dims],
    }

    if discrete_dims:
        pred_pos = pred[:, discrete_dims] >= 0.5
        gt_pos = gt[:, discrete_dims] >= 0.5
        tp = (pred_pos & gt_pos).sum(dim=0).to(torch.float32)
        pred_count = pred_pos.sum(dim=0).to(torch.float32)
        gt_count = gt_pos.sum(dim=0).to(torch.float32)
        precision = tp / pred_count.clamp(min=1.0)
        recall = tp / gt_count.clamp(min=1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        summary["binary_like"] = {
            "dims": [int(i) for i in discrete_dims],
            "tp": tp.tolist(),
            "fp": (pred_count - tp).tolist(),
            "fn": (gt_count - tp).tolist(),
            "tn": (torch.tensor(float(pred.shape[0])) - pred_count - gt_count + tp).tolist(),
            "pred_pos_ratio": pred_pos.to(torch.float32).mean(dim=0).tolist(),
            "gt_pos_ratio": gt_pos.to(torch.float32).mean(dim=0).tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "mean_precision": float(precision.mean().item()),
            "mean_recall": float(recall.mean().item()),
            "mean_f1": float(f1.mean().item()),
        }
    else:
        summary["binary_like"] = None
    return summary


def _as_action_td(action: torch.Tensor) -> np.ndarray:
    arr = action.detach().to(device="cpu", dtype=torch.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected single-batch action [1,T,D], got {tuple(arr.shape)}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected action [T,D] or [1,T,D], got {tuple(arr.shape)}")
    return arr.numpy().astype(np.float32)


def _add_time_lines(
    ax,
    num_steps: int,
    *,
    block_size: Optional[int],
    action_video_freq_ratio: Optional[int],
) -> None:
    ratio = int(action_video_freq_ratio or 0)
    if ratio > 0:
        for x in range(ratio, num_steps, ratio):
            ax.axvline(x - 0.5, color="black", linewidth=0.45, alpha=0.2)
    if block_size is not None and block_size > 0:
        for x in range(block_size, num_steps, block_size):
            ax.axvline(x - 0.5, color="black", linewidth=1.25, alpha=0.8)


def _set_dim_ticks(
    ax,
    dims: Sequence[int],
    reference_scales: Optional[Sequence[float]] = None,
) -> None:
    dim_count = len(dims)
    if dim_count <= 32:
        ticks = np.arange(dim_count, dtype=int)
    else:
        ticks = np.linspace(0, dim_count - 1, 16, dtype=int)
    ax.set_yticks(ticks)
    labels = []
    for tick in ticks:
        position = int(tick)
        label = f"d{int(dims[position])}"
        if reference_scales is not None:
            label += f" (±{float(reference_scales[position]):.3g})"
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=8)


def _validate_continuous_reference_scale(
    reference_scale: Optional[Sequence[float]],
    dim_count: int,
) -> Optional[np.ndarray]:
    if reference_scale is None:
        return None
    scale = np.asarray(reference_scale, dtype=np.float64)
    if scale.shape != (dim_count,):
        raise ValueError(
            "continuous_reference_scale must have one value per action dim: "
            f"got {scale.shape}, expected ({dim_count},)"
        )
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("continuous_reference_scale values must be finite and positive")
    return scale


def _prepare_action_arrays(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    dim_is_continuous: Optional[Sequence[bool]] = None,
):
    pred = _as_action_td(pred_action)
    gt = _as_action_td(gt_action)
    if pred.shape != gt.shape:
        raise ValueError(f"Predicted action/GT action shape mismatch: {pred.shape} vs {gt.shape}")
    err = np.abs(pred - gt)
    dim_l1 = err.mean(axis=0)
    continuous_dims, discrete_dims = resolve_action_dims(
        torch.from_numpy(gt).unsqueeze(0), dim_is_continuous
    )
    top_cont_dims = sorted(continuous_dims, key=lambda i: float(dim_l1[i]), reverse=True)[: min(6, len(continuous_dims))]
    return pred, gt, err, dim_l1, continuous_dims, discrete_dims, top_cont_dims


def save_action_heatmap(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    path: str,
    block_size: Optional[int] = None,
    *,
    dim_is_continuous: Optional[Sequence[bool]] = None,
    continuous_reference_scale: Optional[Sequence[float]] = None,
    continuous_value_floor: Optional[float] = None,
    action_video_freq_ratio: Optional[int] = None,
    video_action_coupled: Optional[bool] = None,
) -> dict:
    (
        pred,
        gt,
        err,
        _,
        continuous_dims,
        discrete_dims,
        _,
    ) = _prepare_action_arrays(pred_action, gt_action, dim_is_continuous)
    num_steps, dim_count = pred.shape
    reference_scale = _validate_continuous_reference_scale(
        continuous_reference_scale, dim_count
    )
    groups = []
    display = {}

    if continuous_dims:
        continuous_values = np.concatenate(
            [pred[:, continuous_dims].ravel(), gt[:, continuous_dims].ravel()]
        )
        observed_abs_max = float(np.max(np.abs(continuous_values)))
        dim_reference = (
            None
            if reference_scale is None
            else reference_scale[np.asarray(continuous_dims, dtype=int)]
        )
        if dim_reference is not None:
            pred_values = pred[:, continuous_dims] / dim_reference[None, :]
            gt_values = gt[:, continuous_dims] / dim_reference[None, :]
            err_values = err[:, continuous_dims] / dim_reference[None, :]
            value_vmax = 1.0
            floor = None
            observed_fraction_max = float(
                max(np.abs(pred_values).max(), np.abs(gt_values).max())
            )
        else:
            robust_abs = float(np.percentile(np.abs(continuous_values), 98))
            floor = max(float(continuous_value_floor or 0.0), 1e-6)
            value_vmax = max(robust_abs, floor)
            pred_values = pred[:, continuous_dims]
            gt_values = gt[:, continuous_dims]
            err_values = err[:, continuous_dims]
            observed_fraction_max = observed_abs_max / value_vmax
        groups.append(
            {
                "name": "Continuous",
                "dims": continuous_dims,
                "pred_values": pred_values,
                "gt_values": gt_values,
                "err_values": err_values,
                "value_vmin": -value_vmax,
                "value_vmax": value_vmax,
                "error_vmax": value_vmax,
                "value_cmap": "coolwarm",
                "observed_abs_max": observed_abs_max,
                "observed_fraction_max": observed_fraction_max,
                "reference_scales": dim_reference,
            }
        )
        display["continuous"] = {
            "dims": [int(i) for i in continuous_dims],
            "value_vmax": value_vmax,
            "observed_abs_max": observed_abs_max,
            "scale_floor": floor,
            "reference_scale_by_dim": (
                None
                if dim_reference is None
                else {
                    str(dim): float(scale)
                    for dim, scale in zip(continuous_dims, dim_reference)
                }
            ),
            "observed_reference_fraction_max": observed_fraction_max,
        }

    if discrete_dims:
        observed_min = float(
            min(pred[:, discrete_dims].min(), gt[:, discrete_dims].min())
        )
        observed_max = float(
            max(pred[:, discrete_dims].max(), gt[:, discrete_dims].max())
        )
        groups.append(
            {
                "name": "Discrete",
                "dims": discrete_dims,
                "pred_values": pred[:, discrete_dims],
                "gt_values": gt[:, discrete_dims],
                "err_values": err[:, discrete_dims],
                "value_vmin": 0.0,
                "value_vmax": 1.0,
                "error_vmax": 1.0,
                "value_cmap": "Blues",
                "observed_abs_max": max(abs(observed_min), abs(observed_max)),
                "observed_fraction_max": max(abs(observed_min), abs(observed_max)),
                "reference_scales": None,
            }
        )
        display["discrete"] = {
            "dims": [int(i) for i in discrete_dims],
            "value_vmin": 0.0,
            "value_vmax": 1.0,
            "observed_min": observed_min,
            "observed_max": observed_max,
        }

    if not groups:
        raise ValueError("Action heatmap requires at least one action dimension")

    width = max(11.0, min(28.0, 6.0 + num_steps * 0.34))
    height = max(8.0, min(18.0, 4.0 + dim_count * 0.4))
    fig, axes = plt.subplots(
        3,
        len(groups),
        figsize=(width, height),
        constrained_layout=True,
        squeeze=False,
    )
    for col, group in enumerate(groups):
        dims = group["dims"]
        observed = group["observed_abs_max"]
        observed_fraction = group["observed_fraction_max"]
        scale = group["value_vmax"]
        panels = [
            ("Pred", group["pred_values"].T, group["value_cmap"], group["value_vmin"], scale),
            ("GT", group["gt_values"].T, group["value_cmap"], group["value_vmin"], scale),
            ("|Pred - GT|", group["err_values"].T, "magma", 0.0, group["error_vmax"]),
        ]
        for row, (label, values, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(
                values,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            suffix = ""
            if group["name"] == "Continuous":
                if group["reference_scales"] is not None:
                    suffix = (
                        "; / global normalizer scale"
                        f"; observed max={observed_fraction:.3g}x"
                    )
                else:
                    suffix = f"; scale={scale:.4g}, observed |max|={observed:.4g}"
            ax.set_title(f"{group['name']} {label}{suffix}", fontsize=10)
            ax.set_ylabel("dim")
            _set_dim_ticks(ax, dims, group["reference_scales"])
            _add_time_lines(
                ax,
                num_steps,
                block_size=block_size,
                action_video_freq_ratio=action_video_freq_ratio,
            )
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        axes[-1, col].set_xlabel("action step")

    ratio_text = (
        f", {int(action_video_freq_ratio)} action steps/video frame"
        if action_video_freq_ratio
        else ""
    )
    coupling_text = (
        ""
        if video_action_coupled is None
        else f", video/action coupled={'yes' if video_action_coupled else 'no'}"
    )
    fig.suptitle(
        f"Action heatmaps: T={num_steps}, D={dim_count}{ratio_text}{coupling_text}; "
        "thin lines=video frames, thick lines=AR blocks",
        fontsize=12,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return display


def save_action_trace_plot(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    path: str,
    block_size: Optional[int] = None,
    *,
    dim_is_continuous: Optional[Sequence[bool]] = None,
    continuous_reference_scale: Optional[Sequence[float]] = None,
    continuous_value_floor: Optional[float] = None,
    action_video_freq_ratio: Optional[int] = None,
) -> bool:
    pred, gt, _, dim_l1, _, _, top_cont_dims = _prepare_action_arrays(
        pred_action, gt_action, dim_is_continuous
    )
    if not top_cont_dims:
        return False
    num_steps = pred.shape[0]
    reference_scale = _validate_continuous_reference_scale(
        continuous_reference_scale, pred.shape[1]
    )
    width = max(11.0, min(26.0, 6.0 + num_steps * 0.32))
    height = max(4.0, 2.2 + len(top_cont_dims) * 0.55)
    fig, ax = plt.subplots(1, 1, figsize=(width, height), constrained_layout=True)
    x = np.arange(num_steps)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, dim in enumerate(top_cont_dims):
        color = colors[i % len(colors)]
        ax.plot(x, pred[:, dim], color=color, linewidth=1.9, label=f"pred d{dim} (L1={dim_l1[dim]:.3g})")
        ax.plot(x, gt[:, dim], color=color, linewidth=1.5, linestyle="--", label=f"gt d{dim}")
    _add_time_lines(
        ax,
        num_steps,
        block_size=block_size,
        action_video_freq_ratio=action_video_freq_ratio,
    )
    observed_abs_max = float(
        np.max(np.abs(np.concatenate([pred[:, top_cont_dims], gt[:, top_cont_dims]])))
    )
    reference_limit = (
        0.0
        if reference_scale is None
        else float(reference_scale[np.asarray(top_cont_dims, dtype=int)].max())
    )
    value_limit = max(
        observed_abs_max,
        reference_limit,
        float(continuous_value_floor or 0.0),
        1e-6,
    )
    ax.set_ylim(-value_limit * 1.05, value_limit * 1.05)
    ax.set_title("Top continuous-dim trajectories (solid=pred, dashed=GT)", fontsize=12)
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def save_action_binary_mask_plot(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    path: str,
    block_size: Optional[int] = None,
    *,
    dim_is_continuous: Optional[Sequence[bool]] = None,
    action_video_freq_ratio: Optional[int] = None,
) -> bool:
    pred, gt, _, _, _, discrete_dims, _ = _prepare_action_arrays(
        pred_action, gt_action, dim_is_continuous
    )
    if not discrete_dims:
        return False
    num_steps = pred.shape[0]
    width = max(11.0, min(26.0, 6.0 + num_steps * 0.32))
    height = max(4.5, 2.5 + len(discrete_dims) * 0.22)
    fig, axes = plt.subplots(2, 1, figsize=(width, height), constrained_layout=True)
    panels = [
        ("Pred discrete mask (>= 0.5)", (pred[:, discrete_dims] >= 0.5).T.astype(np.float32)),
        ("GT discrete mask (>= 0.5)", (gt[:, discrete_dims] >= 0.5).T.astype(np.float32)),
    ]
    for ax, (title, values) in zip(axes, panels):
        im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("dim")
        ax.set_yticks(np.arange(len(discrete_dims)))
        ax.set_yticklabels([f"d{d}" for d in discrete_dims], fontsize=8)
        _add_time_lines(
            ax,
            num_steps,
            block_size=block_size,
            action_video_freq_ratio=action_video_freq_ratio,
        )
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("time step")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def save_action_dim_l1_plot(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    path: str,
    *,
    dim_is_continuous: Optional[Sequence[bool]] = None,
) -> None:
    pred, _, _, dim_l1, _, discrete_dims, _ = _prepare_action_arrays(
        pred_action, gt_action, dim_is_continuous
    )
    dim_count = pred.shape[1]
    width = max(9.0, min(22.0, 4.0 + dim_count * 0.35))
    fig, ax = plt.subplots(1, 1, figsize=(width, 4.2), constrained_layout=True)
    colors = ["tab:orange" if i in discrete_dims else "tab:blue" for i in range(dim_count)]
    ax.bar(np.arange(dim_count), dim_l1, color=colors, width=0.8)
    ax.set_title("Per-dim L1 (blue=continuous, orange=discrete)", fontsize=12)
    ax.set_xlabel("dim")
    ax.set_ylabel("L1")
    ax.set_xticks(np.arange(dim_count))
    ax.set_xticklabels([f"d{i}" for i in range(dim_count)], rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_action_artifacts(
    prefix: str,
    metrics: dict,
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    block_size: Optional[int] = None,
    *,
    dim_is_continuous: Optional[Sequence[bool]] = None,
    continuous_reference_scale: Optional[Sequence[float]] = None,
    continuous_value_floor: Optional[float] = None,
    action_video_freq_ratio: Optional[int] = None,
    video_action_coupled: Optional[bool] = None,
) -> dict:
    json_path = f"{prefix}_action_metrics.json"
    heatmap_path = f"{prefix}_action_heatmap.png"
    trace_path = f"{prefix}_action_traces.png"
    binary_path = f"{prefix}_action_binary_masks.png"
    dim_l1_path = f"{prefix}_action_dim_l1.png"

    pred_arr = _as_action_td(pred_action)
    gt_arr = _as_action_td(gt_action)
    metrics["pred_action"] = pred_arr.tolist()
    metrics["gt_action"] = gt_arr.tolist()
    display_scales = save_action_heatmap(
        pred_action,
        gt_action,
        heatmap_path,
        block_size=block_size,
        dim_is_continuous=dim_is_continuous,
        continuous_reference_scale=continuous_reference_scale,
        continuous_value_floor=continuous_value_floor,
        action_video_freq_ratio=action_video_freq_ratio,
        video_action_coupled=video_action_coupled,
    )
    has_trace = save_action_trace_plot(
        pred_action,
        gt_action,
        trace_path,
        block_size=block_size,
        dim_is_continuous=dim_is_continuous,
        continuous_reference_scale=continuous_reference_scale,
        continuous_value_floor=continuous_value_floor,
        action_video_freq_ratio=action_video_freq_ratio,
    )
    has_binary = save_action_binary_mask_plot(
        pred_action,
        gt_action,
        binary_path,
        block_size=block_size,
        dim_is_continuous=dim_is_continuous,
        action_video_freq_ratio=action_video_freq_ratio,
    )
    save_action_dim_l1_plot(
        pred_action,
        gt_action,
        dim_l1_path,
        dim_is_continuous=dim_is_continuous,
    )
    metrics["artifact_display"] = {
        "action_video_freq_ratio": (
            None
            if action_video_freq_ratio is None
            else int(action_video_freq_ratio)
        ),
        "video_action_coupled": video_action_coupled,
        "heatmap_scales": display_scales,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    result = {
        "action_metrics_path": json_path,
        "action_heatmap_path": heatmap_path,
        "action_dim_l1_path": dim_l1_path,
    }
    if has_trace:
        result["action_trace_path"] = trace_path
    if has_binary:
        result["action_binary_mask_path"] = binary_path
    return result
