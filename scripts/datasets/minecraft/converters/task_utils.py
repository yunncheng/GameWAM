"""Shared output-directory validation for Minecraft data converters."""
from pathlib import Path
from typing import Union


def assert_output_dir_empty(output_dir: Union[str, Path]) -> None:
    """Raise if ``output_dir`` already holds converted episodes.

    Multi-batch / multi-env converters share one LeRobot root across
    batches. If that root still contains files from a previous (possibly
    aborted) run, ``LeRobotDataset.create()`` will build fresh meta with
    ``num_episodes=0`` while ``videos/`` still holds stale mp4s; the final
    ``rglob("*.mp4") == num_episodes`` assert inside ``save_episode`` then
    fires with a cryptic AssertionError.

    Fail fast here with a clear message so the user knows what to do.
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        return
    stale_mp4 = (
        list((out_path / "videos").rglob("*.mp4"))
        if (out_path / "videos").exists()
        else []
    )
    stale_parquet = (
        list((out_path / "data").rglob("*.parquet"))
        if (out_path / "data").exists()
        else []
    )
    total = len(stale_mp4) + len(stale_parquet)
    if total:
        raise FileExistsError(
            f"output_dir {output_dir} already contains {total} stale file(s) "
            f"from a previous run ({len(stale_mp4)} mp4 + "
            f"{len(stale_parquet)} parquet). Remove it first:\n"
            f"    rm -rf {output_dir}"
        )
