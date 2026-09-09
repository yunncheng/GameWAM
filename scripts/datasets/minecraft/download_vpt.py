"""Download OpenAI VPT Contractor Data."""

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# Batch definitions
# ─────────────────────────────────────────────────────────────────────
VPT_BATCHES = {
    "6xx": {
        "openai_index": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json",
        "subdir": "all_6xx_Jun_29",
        "desc": "6.x — core gameplay recordings",
    },
    "7xx": {
        "openai_index": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json",
        "subdir": "all_7xx_Apr_6",
        "desc": "7.x — gameplay with changed prompts",
    },
    "8xx": {
        "openai_index": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_8xx_Jun_29.json",
        "subdir": "all_8xx_Jun_29",
        "desc": "8.x — house building from scratch",
    },
    "9xx": {
        "openai_index": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_9xx_Jun_29.json",
        "subdir": "all_9xx_Jun_29",
        "desc": "9.x — house building from random starting materials",
    },
    "10xx": {
        "openai_index": "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json",
        "subdir": "all_10xx_Jun_29",
        "desc": "10.x — obtain diamond pickaxe",
    },
}


# ─────────────────────────────────────────────────────────────────────
# Download: OpenAI source
# ─────────────────────────────────────────────────────────────────────

def download_batch_openai(batch_key: str, raw_dir: str, workers: int = 4):
    """Download a VPT batch from the OpenAI Azure blob storage.

    Files are organized into <batch_dir>/<subdir>/videos/ and
    <batch_dir>/<subdir>/jsonl/.
    """
    info = VPT_BATCHES[batch_key]
    url = info["openai_index"]
    subdir = info["subdir"]

    print(f"\n{'='*60}")
    print(f"Batch: {batch_key} — {info['desc']}")
    print(f"Source: OpenAI ({url})")
    print(f"{'='*60}")

    batch_dir = os.path.join(raw_dir, batch_key)
    target_dir = os.path.join(batch_dir, subdir)
    videos_dir = os.path.join(target_dir, "videos")
    jsonl_dir = os.path.join(target_dir, "jsonl")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(jsonl_dir, exist_ok=True)

    print("Fetching index ...")
    with urllib.request.urlopen(url) as resp:
        index = json.loads(resp.read().decode())

    basedir = index["basedir"]
    relpaths = index["relpaths"]
    print(f"  {len(relpaths)} recordings in this batch")

    def _download_pair(relpath):
        stem = relpath[:-4] if relpath.endswith(".mp4") else relpath
        local_name = stem.replace("/", "_")
        local_mp4 = os.path.join(videos_dir, local_name + ".mp4")
        local_jsonl = os.path.join(jsonl_dir, local_name + ".jsonl")

        if os.path.exists(local_mp4) and os.path.exists(local_jsonl):
            return (local_mp4, local_jsonl, True)

        try:
            urllib.request.urlretrieve(f"{basedir}{stem}.mp4", local_mp4)
            urllib.request.urlretrieve(f"{basedir}{stem}.jsonl", local_jsonl)
            return (local_mp4, local_jsonl, False)
        except Exception as e:
            for p in [local_mp4, local_jsonl]:
                if os.path.exists(p):
                    os.remove(p)
            return (None, None, False, str(e))

    new = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_pair, rp): rp for rp in relpaths}
        with tqdm(total=len(relpaths), desc=f"  Downloading {batch_key}", unit="pair") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if len(result) == 4:
                    failed += 1
                    if failed <= 5:
                        tqdm.write(f"    [FAIL] {result[3]}")
                    elif failed == 6:
                        tqdm.write(f"    [FAIL] ... suppressing further error messages")
                elif result[2]:
                    skipped += 1
                else:
                    new += 1
                pbar.set_postfix(new=new, skip=skipped, fail=failed)
                pbar.update(1)

    print(f"  Done: {new} new, {skipped} skipped, {failed} failed")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download VPT Contractor Data."
    )
    parser.add_argument(
        "--batches", nargs="+", default=None,
        help=f"Batch keys to process. Available: {list(VPT_BATCHES.keys())}. Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save downloaded VPT data (required unless --list is used).",
    )
    parser.add_argument("--workers", type=int, default=32, help="Download threads (default: 32).")
    parser.add_argument("--list", action="store_true", help="List batches and exit.")

    args = parser.parse_args()

    if args.list:
        print("Available VPT Contractor Data batches:\n")
        for key, info in VPT_BATCHES.items():
            print(f"  {key:15s}  {info['desc']}")
        return
    if not args.output_dir:
        parser.error("--output-dir is required unless --list is used")

    batches = args.batches or list(VPT_BATCHES.keys())
    for b in batches:
        if b not in VPT_BATCHES:
            print(f"Error: unknown batch '{b}'. Use --list to see available.")
            sys.exit(1)

    print(f"Batches:    {batches}")
    print(f"Source:     OpenAI")
    print(f"Output dir: {args.output_dir}")
    print(f"Workers:    {args.workers}")

    for batch_key in batches:
        download_batch_openai(batch_key, args.output_dir, workers=args.workers)

if __name__ == "__main__":
    main()
