from __future__ import annotations

from typing import Literal, Dict, Annotated, Union, Any, List, Optional, Sequence
import torch
import json
from collections import defaultdict
import numpy as np
from omegaconf import OmegaConf
from gamewam.utils.logging_config import get_logger

from gamewam.utils.pytorch_utils import dict_apply

logger = get_logger(__name__)

ConstConstStr = Annotated[str, "format: 'const_min/const_max', where const_min and const_max give the constant range"]
NormMode = Union[Literal["min/max", "q01/q99", "z-score", "identity"], ConstConstStr]
PerDimModeSpec = List[Dict[str, Any]]
ModeSpec = Union[NormMode, PerDimModeSpec]

class LinearNormalizer:
    def __init__(
            self,
            shape_meta,
            use_stepwise_action_norm,
            default_mode: NormMode,
            exception_mode: Dict[str, Dict[str, ModeSpec]],
            stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]]
        ):
        super().__init__()
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats

        for meta in shape_meta["action"]:
            key = meta["key"]

            if use_stepwise_action_norm:
                cur_stats = {k.removeprefix("stepwise_"): v for k, v in stats["action"][key].items() if k.startswith("stepwise_")}
            else:
                cur_stats = {k.removeprefix("global_"): v for k, v in stats["action"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "action" in exception_mode and key in exception_mode["action"]:
                cur_mode = exception_mode["action"][key]
            else:
                cur_mode = default_mode
            cur_mode = _resolve_mode_spec(cur_mode, default_mode, meta)

            self.normalizers["action"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
            )

        for meta in shape_meta["state"]:
            key = meta["key"]
            cur_stats = {k.removeprefix("global_"): v for k, v in stats["state"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "state" in exception_mode and key in exception_mode["state"]:
                cur_mode = exception_mode["state"][key]
            else:
                cur_mode = default_mode
            cur_mode = _resolve_mode_spec(cur_mode, default_mode, meta)

            self.normalizers["state"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
            )

    def get_stats(self):
        stats = {
            "action": {key: norm.get_stats() for key, norm in self.normalizers["action"].items()},
            "state": {key: norm.get_stats() for key, norm in self.normalizers["state"].items()}
        }
        return stats
                
    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "action" in batch:
            for key, norm in self.normalizers["action"].items():
                batch["action"][key] = norm.forward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.forward(batch["state"][key])

        return batch

    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        for key, norm in self.normalizers["action"].items():
            batch["action"][key] = norm.backward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.backward(batch["state"][key])

        return batch

    def forward_state(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward-normalize only the ``state`` sub-dict. Used by route-aware
        processors that route-normalize ``action`` themselves but still need
        the standard state transform.
        """
        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.forward(batch["state"][key])
        return batch

    def backward_state(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Backward-denormalize only the ``state`` sub-dict. See
        ``forward_state`` for the route-aware use case.
        """
        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.backward(batch["state"][key])
        return batch

    def forward_action_by_route(
        self,
        action_dict: Dict[str, torch.Tensor],
        route_mask: torch.Tensor,
        gameplay_norm: Optional["RouteAwareActionNormalizer"] = None,
        gui_norm: Optional["RouteAwareActionNormalizer"] = None,
        route_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Route-aware forward normalize for the action sub-dict.

        When both ``gameplay_norm`` and ``gui_norm`` are provided, continuous
        dims are normalized with the route-specific normalizer (gameplay
        tokens → ``gameplay_norm``, GUI tokens → ``gui_norm``). Discrete
        dims are normalized with the global normalizer (i.e. ``self``).
        When only the global normalizer is provided, this falls back to
        the per-key ``SingleFieldLinearNormalizer.forward`` semantics
        (no state side-effect, since the route-aware processor splits
        state normalization through ``forward_state``).
        """
        if gameplay_norm is None and gui_norm is None:
            for key, norm in self.normalizers["action"].items():
                if key in action_dict:
                    action_dict[key] = norm.forward(action_dict[key])
            return action_dict
        for key, norm in self.normalizers["action"].items():
            x = action_dict[key]
            x_global = norm.forward(x)
            if isinstance(gameplay_norm, RouteAwareActionNormalizer) and isinstance(
                gui_norm, RouteAwareActionNormalizer
            ):
                # ``RouteAwareActionNormalizer`` works on a single action
                # tensor ``[T, D]`` / ``[B, T, D]`` with a route_mask
                # ``[T, 1]`` / ``[B, T, 1]``. The ``LinearNormalizer``
                # wrapper adds the action-key indirection.
                x_route = RouteAwareActionNormalizer.apply_to_tensor(
                    x, route_mask, gameplay_norm, gui_norm
                )
                if route_valid is not None:
                    route_valid_mask = route_valid.to(device=x.device, dtype=torch.bool)
                    if route_valid_mask.shape != route_mask.shape:
                        raise ValueError(
                            f"route_valid shape {tuple(route_valid_mask.shape)} must match "
                            f"route_mask shape {tuple(route_mask.shape)}"
                        )
                    x_route = torch.where(route_valid_mask.expand_as(x_route), x_route, x_global)
                # Splice: discrete dims from ``x_global``, continuous
                # dims from ``x_route``.
                cont_dims = gameplay_norm.continuous_dims
                disc_dims = gameplay_norm.discrete_dims
                merged = x_global.clone()
                if cont_dims:
                    cont_idx = gameplay_norm._cont_dim_index(x_global.device)
                    merged.index_copy_(-1, cont_idx, x_route.index_select(-1, cont_idx))
                action_dict[key] = merged
            else:
                action_dict[key] = x_global
        return action_dict

    def backward_action_by_route(
        self,
        action_dict: Dict[str, torch.Tensor],
        route_mask: torch.Tensor,
        gameplay_norm: Optional["RouteAwareActionNormalizer"] = None,
        gui_norm: Optional["RouteAwareActionNormalizer"] = None,
        route_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Route-aware backward denormalize for the action sub-dict. See
        ``forward_action_by_route`` for the semantics.
        """
        if gameplay_norm is None and gui_norm is None:
            for key, norm in self.normalizers["action"].items():
                if key in action_dict:
                    action_dict[key] = norm.backward(action_dict[key])
            return action_dict
        for key, norm in self.normalizers["action"].items():
            x = action_dict[key]
            x_global = norm.backward(x)
            if isinstance(gameplay_norm, RouteAwareActionNormalizer) and isinstance(
                gui_norm, RouteAwareActionNormalizer
            ):
                x_route = RouteAwareActionNormalizer.apply_to_tensor_backward(
                    x, route_mask, gameplay_norm, gui_norm
                )
                if route_valid is not None:
                    route_valid_mask = route_valid.to(device=x.device, dtype=torch.bool)
                    if route_valid_mask.shape != route_mask.shape:
                        raise ValueError(
                            f"route_valid shape {tuple(route_valid_mask.shape)} must match "
                            f"route_mask shape {tuple(route_mask.shape)}"
                        )
                    x_route = torch.where(route_valid_mask.expand_as(x_route), x_route, x_global)
                cont_dims = gameplay_norm.continuous_dims
                disc_dims = gameplay_norm.discrete_dims
                merged = x_global.clone()
                if cont_dims:
                    cont_idx = gameplay_norm._cont_dim_index(x_global.device)
                    merged.index_copy_(-1, cont_idx, x_route.index_select(-1, cont_idx))
                action_dict[key] = merged
            else:
                action_dict[key] = x_global
        return action_dict


class RouteAwareActionNormalizer:
    """Per-route action normalizer for the GUI route head.

    Wraps three ``SingleFieldLinearNormalizer`` instances
    (``global``, ``gameplay``, ``gui``) and dispatches per-token:

    - continuous dims: forward/backward through the route-specific
      normalizer (gameplay tokens → ``gameplay_norm``; GUI tokens →
      ``gui_norm``);
    - discrete dims: pass through the global normalizer.

    The wrapper never silently falls back to global on missing inputs;
    ``gui_label_dim_index`` / ``continuous_dims`` / ``discrete_dims`` are
    required up front, and missing or route-mask shape mismatches raise
    ``ValueError`` (fail-fast).
    """

    def __init__(
        self,
        global_norm: SingleFieldLinearNormalizer,
        gameplay_norm: SingleFieldLinearNormalizer,
        gui_norm: SingleFieldLinearNormalizer,
        continuous_dims: Sequence[int],
        discrete_dims: Sequence[int],
    ):
        if global_norm is None or gameplay_norm is None or gui_norm is None:
            raise ValueError(
                "RouteAwareActionNormalizer requires non-None global, "
                "gameplay, and gui normalizers."
            )
        if not continuous_dims:
            raise ValueError(
                "RouteAwareActionNormalizer requires at least one continuous dim."
            )
        n_global = int(global_norm.scale.shape[-1])
        for label, norm in (
            ("global", global_norm),
            ("gameplay", gameplay_norm),
            ("gui", gui_norm),
        ):
            if int(norm.scale.shape[-1]) != n_global:
                raise ValueError(
                    f"RouteAwareActionNormalizer: {label} norm scale last dim "
                    f"{int(norm.scale.shape[-1])} != global {n_global}."
                )
        self.global_norm = global_norm
        self.gameplay_norm = gameplay_norm
        self.gui_norm = gui_norm
        self.continuous_dims = list(int(d) for d in continuous_dims)
        self.discrete_dims = list(int(d) for d in discrete_dims)
        if any(d < 0 or d >= n_global for d in self.continuous_dims):
            raise ValueError(
                f"continuous_dims {self.continuous_dims} out of range for action dim {n_global}."
            )
        if any(d < 0 or d >= n_global for d in self.discrete_dims):
            raise ValueError(
                f"discrete_dims {self.discrete_dims} out of range for action dim {n_global}."
            )
        # Per-device cache for the continuous-dim index tensor used by the
        # per-sample route splice. Read-only downstream (index_select /
        # index_copy_ source).
        self._cont_dim_index_cache: dict = {}

    def _cont_dim_index(self, device: torch.device) -> torch.Tensor:
        idx = self._cont_dim_index_cache.get(device)
        if idx is None:
            idx = torch.as_tensor(self.continuous_dims, dtype=torch.long, device=device)
            self._cont_dim_index_cache[device] = idx
        return idx

    @staticmethod
    def _resolve_route_mask(
        x: torch.Tensor, route_mask: torch.Tensor
    ) -> torch.Tensor:
        """Validate route_mask shape and broadcast it to ``x.shape[:-1] + (1,)``.

        Accepts ``[T, 1]`` for ``x`` of shape ``[T, D]`` and ``[B, T, 1]``
        for ``x`` of shape ``[B, T, D]``. Anything else raises.
        """
        if not isinstance(route_mask, torch.Tensor):
            raise ValueError(
                f"route_mask must be a torch.Tensor, got {type(route_mask).__name__}."
            )
        route_mask = route_mask.to(device=x.device, dtype=torch.bool)
        if x.ndim == 2:
            if route_mask.ndim != 2 or route_mask.shape[-1] != 1:
                raise ValueError(
                    f"route_mask shape {tuple(route_mask.shape)} incompatible with "
                    f"x shape {tuple(x.shape)}; expected [T, 1]."
                )
            if route_mask.shape[0] != x.shape[0]:
                raise ValueError(
                    f"route_mask length {route_mask.shape[0]} != x length {x.shape[0]}."
                )
        elif x.ndim == 3:
            if route_mask.ndim != 3 or route_mask.shape[-1] != 1:
                raise ValueError(
                    f"route_mask shape {tuple(route_mask.shape)} incompatible with "
                    f"x shape {tuple(x.shape)}; expected [B, T, 1]."
                )
            if route_mask.shape[0] != x.shape[0] or route_mask.shape[1] != x.shape[1]:
                raise ValueError(
                    f"route_mask shape {tuple(route_mask.shape)} incompatible with "
                    f"x shape {tuple(x.shape)}."
                )
        else:
            raise ValueError(
                f"RouteAwareActionNormalizer expects [T, D] or [B, T, D] x, got {tuple(x.shape)}."
            )
        return route_mask

    def forward(
        self,
        x: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Route-aware forward normalize.

        Continuous dims use ``gameplay_norm`` / ``gui_norm`` per token;
        discrete dims use ``global_norm``. ``route_mask=True`` selects
        the GUI normalizer.
        """
        route_mask = self._resolve_route_mask(x, route_mask)
        out_global = self.global_norm.forward(x)
        out_gameplay = self.gameplay_norm.forward(x)
        out_gui = self.gui_norm.forward(x)
        # ``cont_idx`` must live on the same device as ``x`` (or any
        # of the ``out_*`` outputs) — otherwise ``index_select`` /
        # ``index_copy_`` raise a device mismatch when ``x`` is on CUDA.
        cont_idx = torch.as_tensor(
            self.continuous_dims, dtype=torch.long, device=x.device,
        )
        # Splice route-specific forward into continuous slots.
        merged = out_global.clone()
        # True = GUI → pick out_gui.
        gui_mask = route_mask.expand(*x.shape[:-1], len(self.continuous_dims))
        # Per-dim continuous slice via indexing for correctness under
        # arbitrary (non-contiguous) cont_idx.
        cont_slice_gp = out_gameplay.index_select(-1, cont_idx)
        cont_slice_gui = out_gui.index_select(-1, cont_idx)
        cont_merged = torch.where(gui_mask, cont_slice_gui, cont_slice_gp)
        merged.index_copy_(-1, cont_idx, cont_merged)
        return merged

    def backward(
        self,
        x: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Route-aware backward denormalize. Continuous dims are mapped
        back per route; discrete dims go through ``global_norm``.
        """
        route_mask = self._resolve_route_mask(x, route_mask)
        out_global = self.global_norm.backward(x)
        out_gameplay = self.gameplay_norm.backward(x)
        out_gui = self.gui_norm.backward(x)
        cont_idx = torch.as_tensor(
            self.continuous_dims, dtype=torch.long, device=x.device,
        )
        merged = out_global.clone()
        gui_mask = route_mask.expand(*x.shape[:-1], len(self.continuous_dims))
        cont_slice_gp = out_gameplay.index_select(-1, cont_idx)
        cont_slice_gui = out_gui.index_select(-1, cont_idx)
        cont_merged = torch.where(gui_mask, cont_slice_gui, cont_slice_gp)
        merged.index_copy_(-1, cont_idx, cont_merged)
        return merged

    @classmethod
    def apply_to_tensor(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        gameplay_norm: "RouteAwareActionNormalizer",
        gui_norm: "RouteAwareActionNormalizer",
    ) -> torch.Tensor:
        """Helper used by ``LinearNormalizer`` to splice a route-aware
        forward into the global ``forward`` pipeline. The two route
        normalizers must share the same continuous_dims / discrete_dims
        (the GUI ``RouteAwareActionNormalizer``).
        """
        return gui_norm.forward(x, route_mask)

    @classmethod
    def apply_to_tensor_backward(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        gameplay_norm: "RouteAwareActionNormalizer",
        gui_norm: "RouteAwareActionNormalizer",
    ) -> torch.Tensor:
        """Helper used by ``LinearNormalizer`` to splice a route-aware
        backward into the global ``backward`` pipeline.
        """
        return gui_norm.backward(x, route_mask)


class SingleFieldLinearNormalizer:
    std_reg = 1e-8
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0

    def __init__(self, stats, mode: ModeSpec = "min/max"):
        self.stats = stats
        self.mode = mode

        if _is_per_dim_spec(mode):
            n_dim = int(stats["min"].shape[-1])
            is_stepwise = stats["min"].ndim == 2
            if is_stepwise:
                action_size = int(stats["min"].shape[0])
                scale = torch.zeros(
                    action_size, n_dim,
                    dtype=stats["min"].dtype, device=stats["min"].device,
                )
                offset = torch.zeros(
                    action_size, n_dim,
                    dtype=stats["min"].dtype, device=stats["min"].device,
                )
            else:
                scale = torch.zeros(n_dim, dtype=stats["min"].dtype, device=stats["min"].device)
                offset = torch.zeros(n_dim, dtype=stats["min"].dtype, device=stats["min"].device)
            covered = torch.zeros(n_dim, dtype=torch.bool)
            for entry in mode:
                dims = list(entry["dims"])
                sub_mode = entry["mode"]
                sub_stats = {k: v[..., dims] for k, v in stats.items() if isinstance(v, torch.Tensor)}
                sub_norm = SingleFieldLinearNormalizer(sub_stats, mode=sub_mode)
                dims_t = torch.as_tensor(dims, dtype=torch.long)
                scale.index_copy_(-1, dims_t, sub_norm.scale)
                offset.index_copy_(-1, dims_t, sub_norm.offset)
                covered[dims_t] = True
            if not covered.all():
                missing = (~covered).nonzero().flatten().tolist()
                raise ValueError(
                    f"Per-dim mode spec does not cover dims {missing}; "
                    "LinearNormalizer should have appended a default-mode entry."
                )
            self.scale = scale
            self.offset = offset
            return

        if mode == "identity":
            scale = torch.ones_like(stats["min"])
            offset = torch.zeros_like(stats["min"])
        elif mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            scale = 1.0 / (input_std + self.std_reg)
            offset = - input_mean / (input_std + self.std_reg)
        else:
            if mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif mode == "q01/q99":
                input_min, input_max = stats["q01"], stats["q99"]
            else:
                # parse const_min/const_max
                input_min, input_max = map(float, mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]

        self.scale = scale
        self.offset = offset
    def get_stats(self):
        return self.stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.offset
        x = torch.clamp(x, -5.0, 5.0)
        return x
    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.offset) / self.scale
        return x


def _is_per_dim_spec(mode) -> bool:
    """A per-dim mode spec is a list/ListConfig of {dims, mode} dicts."""
    if OmegaConf.is_list(mode):
        try:
            mode = OmegaConf.to_container(mode, resolve=True)
        except Exception:
            return False
    if not isinstance(mode, list):
        return False
    if len(mode) == 0:
        return False
    return all(isinstance(e, dict) and "dims" in e and "mode" in e for e in mode)


def _resolve_mode_spec(cur_mode, default_mode, meta) -> ModeSpec:
    """Normalize user-provided mode specs and backfill default for uncovered dims.

    When ``cur_mode`` is a per-dim spec (list of {dims, mode}), any dims in
    ``meta["raw_shape"]`` not covered by an explicit entry are appended as a
    trailing {dims: remaining, mode: default_mode} entry. Scalar modes are
    returned unchanged.

    ``meta`` is expected to be a plain dict; BaseProcessor / BaseGameDataset
    normalize DictConfig → dict at their construction sites so no OmegaConf
    branch is needed here.
    """
    if OmegaConf.is_config(cur_mode):
        cur_mode = OmegaConf.to_container(cur_mode, resolve=True)
    if not _is_per_dim_spec(cur_mode):
        return cur_mode

    if not isinstance(meta, dict) or "raw_shape" not in meta:
        raise ValueError(
            f"Per-dim mode spec requires meta['raw_shape'] (int); "
            f"got meta={meta!r} (type={type(meta).__name__}). "
            "Upstream should pass shape_meta through BaseProcessor/BaseGameDataset, "
            "which normalize DictConfig → dict."
        )
    raw_shape = meta["raw_shape"]
    if isinstance(raw_shape, (list, tuple)):
        n_dim = int(raw_shape[-1])
    else:
        n_dim = int(raw_shape)

    entries = [dict(e) for e in cur_mode]
    covered = set()
    for entry in entries:
        dims = list(entry["dims"])
        overlap = covered.intersection(dims)
        if overlap:
            raise ValueError(
                f"Per-dim mode entries overlap on dims {sorted(overlap)} (key={meta.get('key')})."
            )
        covered.update(dims)

    remaining = [d for d in range(n_dim) if d not in covered]
    if remaining:
        entries.append({"dims": remaining, "mode": default_mode})
    return entries

def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):

    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_stats = convert_tensor(dataset_stats)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)

def load_dataset_stats_from_json(file_path: str, 
                                 try_convert_tensor: bool = True) -> Dict[str, Any]:

    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True  
            first = obj[0]
            if isinstance(first, (int, float)):
                return all(isinstance(x, (int, float)) for x in obj)
            elif isinstance(first, list):
                return all(is_numeric_list(item) for item in obj)
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    data = dict_apply(
        data,
        lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) else x,
    )

    return data


def load_normalizer_from_stats(stats_path: str, processor_cfg: Any) -> LinearNormalizer:
    """Load per-policy normalizer from stats file and processor config.

    Supports both regular and mixed training. Reads normalizer params from
    ``processor_cfg`` (a dict or DictConfig containing ``shape_meta``,
    ``use_stepwise_action_norm``, ``norm_default_mode``,
    ``norm_exception_mode``, ``action_output_dim``, ``proprio_output_dim``).

    Args:
        stats_path: Path to dataset_stats.json from training.
        processor_cfg: Processor config dict/DictConfig from ``data.common.processor``.

    Returns:
        Configured LinearNormalizer ready for forward/backward.
    """
    from omegaconf import OmegaConf

    if isinstance(processor_cfg, dict):
        proc = processor_cfg
    else:
        proc = OmegaConf.to_container(processor_cfg, resolve=True)

    shape_meta = proc.get("shape_meta", {})
    use_stepwise_action_norm = bool(proc.get("use_stepwise_action_norm", False))
    norm_default_mode = proc.get("norm_default_mode", "min/max")
    norm_exception_mode = proc.get("norm_exception_mode")

    stats = load_dataset_stats_from_json(stats_path)

    return LinearNormalizer(
        shape_meta=shape_meta,
        use_stepwise_action_norm=use_stepwise_action_norm,
        default_mode=norm_default_mode,
        exception_mode=norm_exception_mode,
        stats=stats,
    )
