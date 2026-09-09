"""Success rate visualization and result aggregation utilities.

Ported from JarvisVLA draw_utils.py — produces the same output format
so that results from both projects can be compared directly.

Result data format (``end.json``):
    A list of tuples: ``[(success: bool, steps: int, episode_id: str), ...]``
"""

import json
import os
from typing import List, Tuple

from experiments.minecraft.utils.task_sets import (
    MINI_TASKS_SET,
    task_category,
)


def load_json(path: str, default=None):
    """Load a JSON file; return *default* if it does not exist."""
    if not os.path.exists(path):
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path: str):
    """Atomically write *data* to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def show_success_rate(data: List[Tuple], file_path: str):
    """Plot the cumulative success rate over episode steps.

    Args:
        data: [(success, steps, episode_id), ...]
        file_path: Output PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filtered_data = sorted([step for success, step, _ in data if success])

    last_data = 0
    cumulative_percent = []
    new_filtered_data = []
    for i in range(len(filtered_data) - 1, -1, -1):
        if last_data == filtered_data[i]:
            continue
        last_data = filtered_data[i]
        new_filtered_data.append(filtered_data[i])
        cumulative_percent.append((1 + i) / len(data) * 100)
    cumulative_percent.append(0)
    new_filtered_data.append(0)
    new_filtered_data.reverse()
    cumulative_percent.reverse()

    plt.figure(figsize=(10, 5))
    plt.plot(new_filtered_data, cumulative_percent, marker="o", linestyle="-", color="b")
    plt.xlabel("Game Playing Steps")
    plt.ylabel("% of Successful Episodes")
    plt.title("Cumulative Success over Game Playing Steps")
    plt.grid(True)
    plt.ylim(0, 100)
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    plt.savefig(file_path)
    plt.close()


def compute_success_rate(data: List[Tuple]) -> float:
    """Return success rate as a float in [0, 1]."""
    if not data:
        return 0.0
    successes = sum(1 for success, _, _ in data if success)
    return successes / len(data)


def print_summary_table(output_dir: str, tasks: List[str]) -> dict:
    """Print & persist a formatted summary table for all tasks."""
    lines: List[str] = []

    def _emit(line: str = ""):
        print(line)
        lines.append(line)

    header = f"{'Task':<40} {'Episodes':>8} {'Success':>8} {'Rate':>8} {'Avg Steps':>10}"
    _emit("=" * len(header))
    _emit(header)
    _emit("-" * len(header))

    # Per-task rows + accumulate (success-only) step totals for category stats.
    # cat_data values: [total_eps, total_success, success_step_sum]
    cat_data = {}
    cat_mini = {}
    # Per-task success rates for std computation
    cat_task_rates = {}
    cat_mini_task_rates = {}
    per_task: List[dict] = []
    for task_name in tasks:
        task_dir = os.path.join(output_dir, task_name.replace(":", "_"))
        log_path = os.path.join(task_dir, "end.json")
        data = load_json(log_path, default=[])
        cat = task_category(task_name)
        is_mini = task_name in MINI_TASKS_SET

        if cat not in cat_data:
            cat_data[cat] = [0, 0, 0]
            cat_mini[cat] = [0, 0, 0]
            cat_task_rates[cat] = []
            cat_mini_task_rates[cat] = []

        if not data:
            _emit(f"{task_name:<40} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            per_task.append({
                "task": task_name,
                "category": cat,
                "in_mini": is_mini,
                "episodes": 0,
                "success": 0,
                "success_rate": None,
                "avg_steps_on_success": None,
            })
            continue
        n = len(data)
        n_success = sum(1 for s, _, _ in data if s)
        rate = compute_success_rate(data)
        success_step_sum = sum(step for s, step, _ in data if s)
        # Success-only average. `None` when no episode succeeded (avoid
        # printing a misleading 0.0).
        avg_steps = (success_step_sum / n_success) if n_success else None
        avg_steps_str = f"{avg_steps:>10.1f}" if avg_steps is not None else f"{'N/A':>10}"
        _emit(f"{task_name:<40} {n:>8} {n_success:>8} {rate:>8.1%} {avg_steps_str}")

        cat_data[cat][0] += n
        cat_data[cat][1] += n_success
        cat_data[cat][2] += success_step_sum
        cat_task_rates[cat].append(rate)
        if is_mini:
            cat_mini[cat][0] += n
            cat_mini[cat][1] += n_success
            cat_mini[cat][2] += success_step_sum
            cat_mini_task_rates[cat].append(rate)

        per_task.append({
            "task": task_name,
            "category": cat,
            "in_mini": is_mini,
            "episodes": n,
            "success": n_success,
            "success_rate": rate,
            "avg_steps_on_success": avg_steps,
        })

    _emit("=" * len(header))

    # Category-level summary.
    _emit()
    w_cat = 12
    w_num = 14
    cat_header = (
        f"{'Category':<{w_cat}} "
        f"{'ASR(Mini)':>{w_num}} {'Steps':>9} "
        f"{'ASR(All)':>{w_num}} {'Steps':>9}"
    )
    sep = "-" * len(cat_header)
    _emit(sep)
    _emit(cat_header)
    _emit(sep)

    import numpy as _np

    def _asr_with_std(rates):
        """Format ASR ± std from per-task success rates."""
        if not rates:
            return None, None, None
        mean = _np.mean(rates)
        std = _np.std(rates)
        return mean, std, f"{mean:.1%}±{std:.1%}"

    def _fmt_asr_cell(asr_str, width):
        return f"{asr_str if asr_str is not None else 'N/A':>{width}}"

    def _fmt_steps_cell(value):
        return f"{value:>9.0f}" if value is not None else f"{'N/A':>9}"

    total_all_eps, total_all_succ, total_all_succ_steps = 0, 0, 0
    total_mini_eps, total_mini_succ, total_mini_succ_steps = 0, 0, 0
    all_task_rates = []
    all_mini_task_rates = []
    # Category-level values, averaged once more for the "Overall(Cat)" row so
    # that every category weighs equally regardless of its task count.
    cat_mini_means, cat_all_means = [], []
    cat_mini_step_means, cat_all_step_means = [], []
    categories_out: dict = {}
    for cat in ["Embodied", "GUI", "Combat", "Other"]:
        if cat not in cat_data:
            continue
        ae, as_, ast_ = cat_data[cat]
        me, ms, mt = cat_mini[cat]
        # Success-only Steps. N/A if 0 successful eps in this category.
        all_avg = (ast_ / as_) if as_ else None

        # Per-task std
        mini_mean, mini_std, mini_asr_str = _asr_with_std(cat_mini_task_rates.get(cat, []))
        all_mean, all_std, all_asr_str_fmt = _asr_with_std(cat_task_rates.get(cat, []))

        mini_asr_str = _fmt_asr_cell(mini_asr_str, w_num)
        all_asr_str_fmt = _fmt_asr_cell(all_asr_str_fmt, w_num)

        mini_avg_val = (mt / ms) if (me and ms) else None
        mini_avg_str = _fmt_steps_cell(mini_avg_val)
        all_avg_str = _fmt_steps_cell(all_avg)
        _emit(
            f"{cat:<{w_cat}} "
            f"{mini_asr_str} {mini_avg_str} "
            f"{all_asr_str_fmt} {all_avg_str}"
        )
        categories_out[cat] = {
            "mini": {"episodes": me, "success": ms, "asr": mini_mean,
                     "asr_std": mini_std,
                     "avg_steps_on_success": mini_avg_val},
            "all": {"episodes": ae, "success": as_, "asr": all_mean,
                    "asr_std": all_std,
                    "avg_steps_on_success": all_avg},
        }
        total_all_eps += ae; total_all_succ += as_; total_all_succ_steps += ast_
        total_mini_eps += me; total_mini_succ += ms; total_mini_succ_steps += mt
        all_task_rates.extend(cat_task_rates.get(cat, []))
        all_mini_task_rates.extend(cat_mini_task_rates.get(cat, []))
        if mini_mean is not None:
            cat_mini_means.append(mini_mean)
        if all_mean is not None:
            cat_all_means.append(all_mean)
        if mini_avg_val is not None:
            cat_mini_step_means.append(mini_avg_val)
        if all_avg is not None:
            cat_all_step_means.append(all_avg)

    _emit(sep)
    overall_mini_mean, overall_mini_std, overall_mini_str = _asr_with_std(all_mini_task_rates)
    overall_all_mean, overall_all_std, overall_all_str = _asr_with_std(all_task_rates)
    overall_mini_str = _fmt_asr_cell(overall_mini_str, w_num)
    overall_all_str = _fmt_asr_cell(overall_all_str, w_num)
    overall_mini_avg = (
        (total_mini_succ_steps / total_mini_succ) if total_mini_succ else None
    )
    overall_mini_avg_str = _fmt_steps_cell(overall_mini_avg)
    overall_all_avg = (total_all_succ_steps / total_all_succ) if total_all_succ else None
    overall_all_avg_str = _fmt_steps_cell(overall_all_avg)
    _emit(
        f"{'Overall':<{w_cat}} "
        f"{overall_mini_str} {overall_mini_avg_str} "
        f"{overall_all_str} {overall_all_avg_str}"
    )

    # Macro row: average the per-category means so Embodied / GUI / Combat
    # contribute equally instead of being dominated by task counts.
    cat_mini_mean, cat_mini_std, cat_mini_asr_str = _asr_with_std(cat_mini_means)
    cat_all_mean, cat_all_std, cat_all_asr_str = _asr_with_std(cat_all_means)
    cat_mini_steps = _np.mean(cat_mini_step_means) if cat_mini_step_means else None
    cat_all_steps = _np.mean(cat_all_step_means) if cat_all_step_means else None
    _emit(
        f"{'Overall(Cat)':<{w_cat}} "
        f"{_fmt_asr_cell(cat_mini_asr_str, w_num)} {_fmt_steps_cell(cat_mini_steps)} "
        f"{_fmt_asr_cell(cat_all_asr_str, w_num)} {_fmt_steps_cell(cat_all_steps)}"
    )
    _emit(sep)

    # Persist
    summary = {
        "note": (
            "avg_steps_on_success is averaged over SUCCESSFUL episodes only; "
            "null = no successful episodes. "
            "asr_std is the standard deviation of per-task success rates within each category. "
            "overall_by_category averages the per-category values so every category "
            "weighs equally; its asr_std is the spread across category means."
        ),
        "per_task": per_task,
        "categories": categories_out,
        "overall": {
            "mini": {
                "episodes": total_mini_eps,
                "success": total_mini_succ,
                "asr": overall_mini_mean,
                "asr_std": overall_mini_std,
                "avg_steps_on_success": overall_mini_avg,
            },
            "all": {
                "episodes": total_all_eps,
                "success": total_all_succ,
                "asr": overall_all_mean,
                "asr_std": overall_all_std,
                "avg_steps_on_success": overall_all_avg,
            },
        },
        "overall_by_category": {
            "mini": {
                "categories": len(cat_mini_means),
                "asr": cat_mini_mean,
                "asr_std": cat_mini_std,
                "avg_steps_on_success": (
                    float(cat_mini_steps) if cat_mini_steps is not None else None
                ),
            },
            "all": {
                "categories": len(cat_all_means),
                "asr": cat_all_mean,
                "asr_std": cat_all_std,
                "avg_steps_on_success": (
                    float(cat_all_steps) if cat_all_steps is not None else None
                ),
            },
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "summary.txt")
    json_path = os.path.join(output_dir, "summary.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    dump_json(summary, json_path)

    return summary
