"""Convert game data (VPT contractor, VPT event sub-clips) to LeRobot format."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_batch_ratios(spec: str):
    """Parse a 'name=ratio,name=ratio' string into a list of (name, ratio)."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"Bad ratios token: {tok!r} (expected name=ratio)")
        name, r = tok.split("=", 1)
        name = name.strip()
        r = float(r)
        if not (0.0 < r <= 1.0):
            raise ValueError(f"Ratio for {name} must be in (0, 1], got {r}")
        out.append((name, r))
    if not out:
        raise ValueError("ratios spec is empty")
    return out


def _parse_batch_caps(spec: str):
    """Parse a 'name=int,name=int' string into a list of (name, cap)."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"Bad caps token: {tok!r} (expected name=int)")
        name, n = tok.split("=", 1)
        name = name.strip()
        n = int(n)
        if n <= 0:
            raise ValueError(f"Cap for {name} must be positive, got {n}")
        out.append((name, n))
    if not out:
        raise ValueError("caps spec is empty")
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert game data to LeRobot format")
    parser.add_argument("--source_type", type=str, required=True,
                        choices=["vpt_contractor", "vpt_contractor_multi",
                                 "vpt_event", "vpt_event_multi"],
                        help="Source data format type")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Path to source data (single-batch modes)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--task", type=str, default=None,
                        help="Task description. Ignored in "
                             "vpt_contractor_multi / vpt_event_multi modes "
                             "(use per-batch defaults). Default None → auto-infer.")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second (VPT is 20)")
    parser.add_argument(
        "--cursor_reference_size",
        type=float,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "Minecraft cursor coordinate reference size. If omitted, each "
            "VPT source video's native width/height is used; all selected "
            "videos must resolve to one common size."
        ),
    )
    parser.add_argument(
        "--vpt_mouse_reference_size",
        type=float,
        nargs=2,
        default=(1280.0, 720.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Native VPT JSONL mouse coordinate size (default: 1280 720).",
    )

    # ─── Single-source subsampling ───────────────────────────────────
    parser.add_argument("--ratio", type=float, default=None,
                        help="Fraction of episodes/pairs to convert (0, 1]. "
                             "Single-source modes (vpt_contractor / "
                             "vpt_event).")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Hard cap on number of episodes / sub-clips. "
                             "Single-source modes only (vpt_contractor / "
                             "vpt_event). For multi-batch modes, "
                             "use --batch_max_episodes.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the candidate list (pairs / episodes / "
                             "events) before applying --ratio / --max_episodes. "
                             "Works for ALL modes. Recommended whenever "
                             "--ratio<1 or any quota is set, otherwise the "
                             "subset is biased toward the alphabetically-"
                             "earliest files.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--skip_null_actions", action="store_true",
                        help="Skip null-action frames during conversion. "
                             "This changes the effective clip timeline and may "
                             "increase event-window collapse.")

    # ─── Multi-batch mode (VPT contractor / VPT event) ───────────────
    parser.add_argument("--vpt_root", type=str, default=None,
                        help="Root directory containing 6xx/7xx/8xx/9xx/10xx "
                             "subdirs (vpt_contractor_multi / vpt_event_multi "
                             "/ vpt_event modes). Auto-discovers available "
                             "batch directories.")
    parser.add_argument("--batch_ratios", type=str, default=None,
                        help="Per-batch fractional quotas. "
                             "Format: 6xx=0.3,7xx=0.3,8xx=1.0,9xx=0.5,10xx=0.7. "
                             "Unlisted batches default to 1.0. "
                             "Supported by vpt_contractor_multi and vpt_event_multi.")
    parser.add_argument("--batch_max_episodes", type=str, default=None,
                        help="Per-batch absolute caps. Format: "
                             "6xx=2000,7xx=2000,8xx=2500,9xx=2500,10xx=6625. "
                             "Applied after --batch_ratios. Supported by "
                             "both vpt_contractor_multi and vpt_event_multi.")
    parser.add_argument("--batch_filter", type=str, default=None,
                        help="Comma-separated batch names to process "
                             "(vpt_event_multi / vpt_contractor_multi). "
                             "Default: all available. "
                             "E.g. 6xx,10xx.")

    # ─── vpt_event-specific ──────────────────────────────────────────
    parser.add_argument("--win_len", type=int, default=256,
                        help="Sub-clip window length in frames (vpt_event "
                             "modes, default 256)")
    parser.add_argument("--bias", type=int, default=0,
                        help="Window-end offset for vpt_event modes. "
                             "Uses end = event_time + 1 + bias. "
                             "bias=0 includes the event frame as the last frame; "
                             "positive values include more post-event frames.")
    parser.add_argument("--min_event_stride", type=int, default=64,
                        help="Extra spacing between kept same-type events during "
                             "joint filtering (vpt_event modes, default 64). "
                             "Usually set it larger than --min_clip_stride so "
                             "same-type events are sparser than general clips.")
    parser.add_argument("--min_clip_stride", type=int, default=1,
                        help="Minimum spacing between kept clip starts within "
                             "the same episode during joint filtering (vpt_event "
                             "modes, default 1). Default 1 collapses byte-identical "
                             "windows (same-frame multi-stat events). If "
                             "--min_event_stride is not larger, clip spacing "
                             "already dominates.")

    args = parser.parse_args()

    # Ensure the parent dir exists (e.g. /path/to/datasets/), but do NOT create
    # output_dir itself — LeRobotDataset.create() requires it to be absent,
    # and append-mode opens the existing one.
    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)

    if args.source_type == "vpt_contractor":
        if not args.source_dir:
            parser.error("--source_dir is required for vpt_contractor")
        from scripts.datasets.minecraft.converters.vpt import (
            convert_vpt_contractor_dataset,
        )
        created = convert_vpt_contractor_dataset(
            data_dir=args.source_dir,
            output_dir=args.output_dir,
            task=args.task,
            fps=args.fps,
            max_episodes=args.max_episodes,
            ratio=args.ratio,
            skip_null_actions=args.skip_null_actions,
            cursor_reference_size=args.cursor_reference_size,
            vpt_mouse_reference_size=args.vpt_mouse_reference_size,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        print(f"Converted {len(created)} VPT episodes to {args.output_dir}")

    elif args.source_type == "vpt_contractor_multi":
        if not args.vpt_root:
            parser.error("--vpt_root is required for vpt_contractor_multi")
        from scripts.datasets.minecraft.converters.vpt import (
            convert_vpt_contractor_multi_batch,
        )

        batch_ratios = None
        if args.batch_ratios:
            batch_ratios = dict(_parse_batch_ratios(args.batch_ratios))

        batch_max_episodes = None
        if args.batch_max_episodes:
            batch_max_episodes = dict(_parse_batch_caps(args.batch_max_episodes))

        batch_filter = None
        if args.batch_filter:
            batch_filter = [b.strip() for b in args.batch_filter.split(",")]

        created = convert_vpt_contractor_multi_batch(
            vpt_root=args.vpt_root,
            output_dir=args.output_dir,
            fps=args.fps,
            skip_null_actions=args.skip_null_actions,
            cursor_reference_size=args.cursor_reference_size,
            vpt_mouse_reference_size=args.vpt_mouse_reference_size,
            batch_filter=batch_filter,
            batch_ratios=batch_ratios,
            batch_max_episodes=batch_max_episodes,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        print(f"Converted {len(created)} VPT episodes (multi-batch) "
              f"to {args.output_dir}")

    elif args.source_type == "vpt_event":
        if not args.vpt_root:
            parser.error("--vpt_root is required for vpt_event")
        from scripts.datasets.minecraft.converters.vpt_event import (
            convert_vpt_event_dataset,
            convert_vpt_event_pooled,
        )
        from scripts.datasets.minecraft.converters.vpt import (
            discover_vpt_batches,
        )
        # Route multiple batch subdirectories through one max-min cap over
        # their union; process a single batch directory directly.
        available_batches = discover_vpt_batches(args.vpt_root)
        if len(available_batches) > 1:
            if args.max_episodes is None:
                parser.error(
                    "--max_episodes is required when --vpt_root contains "
                    "multiple batch subdirs (the pooled driver allocates one "
                    "global budget across all batches)."
                )
            result = convert_vpt_event_pooled(
                vpt_root=args.vpt_root,
                output_dir=args.output_dir,
                max_episodes=args.max_episodes,
                fps=args.fps,
                win_len=args.win_len,
                bias=args.bias,
                min_event_stride=args.min_event_stride,
                min_clip_stride=args.min_clip_stride,
                skip_null_actions=args.skip_null_actions,
                cursor_reference_size=args.cursor_reference_size,
                vpt_mouse_reference_size=args.vpt_mouse_reference_size,
                shuffle=args.shuffle,
                seed=args.seed,
            )
            total = sum(len(v) for v in result.values())
            print(
                f"Converted {total} pooled sub-clips across "
                f"{len(result)} batch(es) to {args.output_dir}"
            )
        else:
            created = convert_vpt_event_dataset(
                vpt_root=args.vpt_root,
                output_dir=args.output_dir,
                fps=args.fps,
                win_len=args.win_len,
                bias=args.bias,
                min_event_stride=args.min_event_stride,
                cursor_reference_size=args.cursor_reference_size,
                vpt_mouse_reference_size=args.vpt_mouse_reference_size,
                skip_null_actions=args.skip_null_actions,
                min_clip_stride=args.min_clip_stride,
                max_episodes=args.max_episodes,
                ratio=args.ratio,
                shuffle=args.shuffle,
                seed=args.seed,
            )
            print(f"Converted {len(created)} VPT event sub-clips to {args.output_dir}")

    elif args.source_type == "vpt_event_multi":
        if not args.vpt_root:
            parser.error("--vpt_root is required for vpt_event_multi")
        from scripts.datasets.minecraft.converters.vpt_event import (
            convert_vpt_event_multi_batch,
        )
        batch_filter = None
        if args.batch_filter:
            batch_filter = [b.strip() for b in args.batch_filter.split(",")]

        batch_ratios = None
        if args.batch_ratios:
            batch_ratios = dict(_parse_batch_ratios(args.batch_ratios))

        batch_max_episodes = None
        if args.batch_max_episodes:
            batch_max_episodes = dict(_parse_batch_caps(args.batch_max_episodes))

        created = convert_vpt_event_multi_batch(
            vpt_root=args.vpt_root,
            output_dir=args.output_dir,
            fps=args.fps,
            win_len=args.win_len,
            bias=args.bias,
            min_event_stride=args.min_event_stride,
            cursor_reference_size=args.cursor_reference_size,
            vpt_mouse_reference_size=args.vpt_mouse_reference_size,
            skip_null_actions=args.skip_null_actions,
            min_clip_stride=args.min_clip_stride,
            batch_filter=batch_filter,
            batch_ratios=batch_ratios,
            batch_max_episodes=batch_max_episodes,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        print(f"Converted {len(created)} VPT event sub-clips (multi-batch) "
              f"to {args.output_dir}")

    else:
        print(f"Unknown source type: {args.source_type}")
        sys.exit(1)
if __name__ == "__main__":
    main()
