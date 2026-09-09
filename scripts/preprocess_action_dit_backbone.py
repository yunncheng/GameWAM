import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from gamewam.models.action.action_dit import ActionDiT
from gamewam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from gamewam.utils.cfg import register_default_resolvers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_ROOT = PROJECT_ROOT / "configs"


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        # Permute the target dimension to the end for interpolation
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        # Construct inverse permutation to restore original order
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        # Permute, interpolate, and restore original order
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _load_task_model_config(task_config: str) -> dict[str, Any]:
    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS_ROOT)):
        cfg = compose(config_name="train", overrides=[f"task={task_config}"])
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise TypeError("Composed `model` config must resolve to a mapping.")
    if "video_dit_config" not in model_cfg or "action_dit_config" not in model_cfg:
        raise ValueError(
            f"Task `{task_config}` must select a model with both "
            "`video_dit_config` and `action_dit_config`."
        )
    return model_cfg


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    return int(cfg[key])


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    return float(cfg[key])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ActionDiT backbone weights from WanVideoDiT and save as .pt payload."
    )
    parser.add_argument(
        "--task-config",
        required=True,
        help="Hydra task config, e.g. mc_uncond_ar_224_4e-5.",
    )
    parser.add_argument("--output", required=True, help="Output .pt path for preprocessed ActionDiT backbone.")
    parser.add_argument("--device", default="cpu", help="Device for loading model and preprocessing.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when the last dimension is resized (true/false). Default: true.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)

    model_cfg = _load_task_model_config(args.task_config)
    video_cfg = dict(model_cfg["video_dit_config"])
    action_cfg = dict(model_cfg["action_dit_config"])
    torch_dtype = _parse_dtype(args.dtype)

    int_fields = ["hidden_dim", "action_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim", "freq_dim"]
    for key in int_fields:
        action_cfg[key] = _require_int_config(action_cfg, key)
    action_cfg["eps"] = _require_float_config(action_cfg, "eps")

    print(f"[INFO] Composed model config from task={args.task_config}. "
          f"Preprocessing ActionDiT backbone with dtype={torch_dtype} on device={args.device}, "
          f"apply_alpha_scaling={apply_alpha_scaling}.")
    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=str(model_cfg["model_id"]),
        tokenizer_model_id=str(model_cfg["tokenizer_model_id"]),
        tokenizer_max_len=int(model_cfg["tokenizer_max_len"]),
        redirect_common_files=bool(model_cfg["redirect_common_files"]),
        dit_config=video_cfg,
        load_text_encoder=False,
    )
    video_expert = components.dit

    action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)
    if int(action_cfg["num_heads"]) != int(video_expert.num_heads):
        raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
    if int(action_cfg["attn_head_dim"]) != int(video_expert.attn_head_dim):
        raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
    if int(action_cfg["num_layers"]) != int(len(video_expert.blocks)):
        raise ValueError("ActionDiT `num_layers` must match video expert.")

    action_state = action_expert.state_dict()
    video_state = video_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in video expert state dict.")
        src = video_state[key]
        target = action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        },
    }
    torch.save(payload, str(output_path))

    skipped = len(action_state) - len(backbone_keys)
    print(
        "[INFO] Saved ActionDiT backbone payload to "
        f"{output_path} (copied={copied}, interpolated={interpolated}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()
