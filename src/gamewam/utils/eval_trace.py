"""Lightweight, non-fatal rollout tracing for evaluation scripts.

The logger records model replans and environment steps without changing the
policy or environment execution path.  In particular, callers can preserve
both the raw model output and the action that the environment actually
executed when an adapter, threshold, or quantizer sits between them.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

logger = logging.getLogger(__name__)

TRACE_VERSION = 1


def to_jsonable(value: Any) -> Any:
    """Recursively convert common numerical objects to JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    # Avoid importing torch in evaluation helpers that do not otherwise need
    # it, while still supporting tensors present in inference debug payloads.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return to_jsonable(value.detach().cpu().tolist())
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def action_chunk_stats(
    chunk: Any,
    binary_action_indices: Iterable[int] = (),
) -> dict[str, Any]:
    """Return compact per-dimension statistics for an action chunk."""
    array = _as_numpy(chunk).astype(np.float32, copy=False)
    if array.ndim != 2:
        return {"error": f"unexpected shape {tuple(array.shape)}"}
    stats: dict[str, Any] = {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
    }
    indices = tuple(int(index) for index in binary_action_indices)
    if indices:
        stats["binary_indices"] = list(indices)
        stats["binary_active_ratio"] = (
            array[:, indices] >= 0.5
        ).mean(axis=0).tolist()
    return stats


def _atomic_json(path: Path, value: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(value),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    os.replace(tmp_path, path)


class RolloutTraceLogger:
    """Persist one rollout's config, replans, executed steps, and summary.

    Logging is deliberately best-effort: an artifact I/O failure is reported
    but never changes policy outputs, environment actions, or episode outcome.
    JSONL files are flushed per record so a failed worker still leaves a useful
    partial trace.
    """

    def __init__(
        self,
        episode_dir: str | Path,
        *,
        action_dim: int,
        action_names: Iterable[str] | None = None,
        binary_action_indices: Iterable[int] = (),
    ) -> None:
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.action_dim = int(action_dim)
        self.action_names = (
            tuple(str(name) for name in action_names)
            if action_names is not None
            else None
        )
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.action_names is not None and len(self.action_names) != self.action_dim:
            raise ValueError(
                "action_names length must match action_dim: "
                f"{len(self.action_names)} != {self.action_dim}"
            )
        self.binary_action_indices = tuple(
            int(index) for index in binary_action_indices
        )
        if any(index < 0 or index >= self.action_dim for index in self.binary_action_indices):
            raise ValueError(
                f"binary_action_indices must be in [0, {self.action_dim}), "
                f"got {self.binary_action_indices}"
            )

        self._trace_path = self.episode_dir / "inference_trace.jsonl"
        self._step_path = self.episode_dir / "step_log.jsonl"
        self._config_path = self.episode_dir / "config.json"
        self._summary_path = self.episode_dir / "episode_summary.json"
        self._trace_file = self._open_jsonl(self._trace_path)
        self._step_file = self._open_jsonl(self._step_path)
        self.replan_count = 0
        self.current_replan_idx = -1
        self.step_count = 0
        self.total_infer_latency = 0.0
        self.total_predicted_steps = 0
        self.total_executed_steps = 0
        self.finalized = False

    @staticmethod
    def _open_jsonl(path: Path):
        try:
            return path.open("w", encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not open rollout trace %s: %s", path, exc)
            return None

    def write_config(self, config: Mapping[str, Any]) -> None:
        try:
            payload = dict(config)
            payload.setdefault("rollout_trace_version", TRACE_VERSION)
            payload.setdefault(
                "action_trace",
                {
                    "action_dim": self.action_dim,
                    "action_names": list(self.action_names) if self.action_names else None,
                    "binary_action_indices": list(self.binary_action_indices),
                    "step_action_vec": "raw_model_output",
                    "step_executed_action_vec": "environment_executed_action",
                },
            )
            _atomic_json(self._config_path, payload)
        except Exception as exc:
            logger.warning("RolloutTraceLogger.write_config failed: %s", exc)

    def on_inference(self, payload: Mapping[str, Any]) -> None:
        if self._trace_file is None or self.finalized:
            return
        try:
            chunk = payload.get("action_chunk")
            record = {
                "replan_idx": self.replan_count,
                "env_step": int(payload.get("step_count", self.step_count)),
                "timestamp": time.time(),
                "prompt": payload.get("prompt"),
                "task_str": payload.get("task_str"),
                "fine_task": payload.get("fine_task"),
                "coarse_task": payload.get("coarse_task"),
                "num_inference_steps": payload.get("num_inference_steps"),
                "text_cfg_scale": payload.get("text_cfg_scale"),
                "sampling_seed": payload.get("sampling_seed"),
                "base_seed": payload.get("base_seed"),
                "global_replan_index": payload.get("global_replan_index"),
                "prediction_horizon": payload.get("prediction_horizon"),
                "execution_horizon": payload.get("execution_horizon"),
                "infer_latency_sec": payload.get("infer_latency_sec"),
                "action_chunk_shape": (
                    list(_as_numpy(chunk).shape) if chunk is not None else None
                ),
                "action_chunk_stats": (
                    action_chunk_stats(chunk, self.binary_action_indices)
                    if chunk is not None
                    else None
                ),
                "action_debug": payload.get("action_debug"),
                "action_route_mask": payload.get("action_route_mask"),
                "action_route_logit": payload.get("action_route_logit"),
                "ar_cycle_step": payload.get("ar_cycle_step"),
                "ar_block_index": payload.get("ar_block_index"),
                "proprio_expected": payload.get("proprio_expected"),
                "proprio_actual": payload.get("proprio_actual"),
                "proprio_raw_indices": payload.get("proprio_raw_indices"),
                "runtime_state": payload.get("runtime_state"),
                "clean_latents_in_cache": payload.get("clean_latents_in_cache"),
                "clean_kv_tokens": payload.get("clean_kv_tokens"),
                "ar_history_mode": payload.get("ar_history_mode"),
            }
            self._trace_file.write(
                json.dumps(to_jsonable(record), ensure_ascii=False, allow_nan=False)
                + "\n"
            )
            self._trace_file.flush()
            self.current_replan_idx = self.replan_count
            self.replan_count += 1
            self.total_infer_latency += float(
                payload.get("infer_latency_sec", 0.0) or 0.0
            )
            self.total_predicted_steps += int(
                payload.get("prediction_horizon", 0) or 0
            )
            self.total_executed_steps += int(
                payload.get("execution_horizon", 0) or 0
            )
        except Exception as exc:
            logger.warning("RolloutTraceLogger.on_inference failed: %s", exc)

    def on_step(
        self,
        *,
        step_idx: int,
        action_vec: Any,
        env_action: Mapping[str, Any],
        executed_action_vec: Any,
        executed_env_action: Mapping[str, Any],
        reward: float,
        done: bool,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        if self._step_file is None or self.finalized:
            return
        try:
            raw = _as_numpy(action_vec).astype(np.float32, copy=False).reshape(-1)
            executed = (
                _as_numpy(executed_action_vec)
                .astype(np.float32, copy=False)
                .reshape(-1)
            )
            if raw.shape != (self.action_dim,) or executed.shape != (self.action_dim,):
                raise ValueError(
                    "step action shape mismatch: "
                    f"raw={raw.shape}, executed={executed.shape}, "
                    f"expected={(self.action_dim,)}"
                )
            record = {
                "step": int(step_idx),
                "from_replan_idx": self.current_replan_idx,
                "action_vec": raw.tolist(),
                "env_action": to_jsonable(env_action),
                "executed_action_vec": executed.tolist(),
                "executed_env_action": to_jsonable(executed_env_action),
                "reward": float(reward),
                "done": bool(done),
                "info": to_jsonable(info) if info is not None else None,
            }
            self._step_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
            self._step_file.flush()
            self.step_count = int(step_idx) + 1
        except Exception as exc:
            logger.warning(
                "RolloutTraceLogger.on_step failed at step %s: %s",
                step_idx,
                exc,
            )

    def finalize(self, summary: Mapping[str, Any]) -> None:
        if self.finalized:
            return
        try:
            payload = dict(summary)
            payload.setdefault("rollout_trace_version", TRACE_VERSION)
            payload.setdefault("num_replans", self.replan_count)
            payload.setdefault("logged_steps", self.step_count)
            payload.setdefault("infer_time_sec", round(self.total_infer_latency, 4))
            if self.total_infer_latency > 0:
                payload.setdefault(
                    "plan_hz",
                    round(
                        self.total_predicted_steps / self.total_infer_latency,
                        4,
                    ),
                )
                payload.setdefault(
                    "exec_hz",
                    round(
                        self.total_executed_steps / self.total_infer_latency,
                        4,
                    ),
                )
            else:
                payload.setdefault("plan_hz", 0.0)
                payload.setdefault("exec_hz", 0.0)
            _atomic_json(self._summary_path, payload)
        except Exception as exc:
            logger.warning("RolloutTraceLogger.finalize failed: %s", exc)
        finally:
            self.finalized = True
            self.close()

    def close(self) -> None:
        for handle_name in ("_trace_file", "_step_file"):
            handle = getattr(self, handle_name, None)
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
            setattr(self, handle_name, None)

