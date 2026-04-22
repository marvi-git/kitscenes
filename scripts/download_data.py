#!/usr/bin/env python3
"""Download KITScenes-LongTail splits from HuggingFace.

Requires prior login:
    huggingface-cli login

Usage:
    python scripts/download_data.py --splits train   # ~245 MB
    python scripts/download_data.py --splits test    # ~34 GB
"""

import argparse
import sys
from pathlib import Path

try:
    import huggingface_hub
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")

REPO_ID = "KIT-MRT/KITScenes-LongTail"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def check_login() -> str:
    try:
        info = huggingface_hub.whoami()
        return info["name"]
    except Exception:
        sys.exit(
            "Not logged in to HuggingFace.\n"
            "Run: huggingface-cli login\n"
            "Then retry."
        )


def download_split(split: str) -> None:
    username = check_login()
    print(f"Authenticated as: {username}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{split}' split from {REPO_ID} → {DATA_DIR}")
    print("(This may take a while for large splits.)\n")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
            allow_patterns=[f"*{split}*"],
        )
    except HfHubHTTPError as exc:
        if "403" in str(exc):
            sys.exit(
                f"Access denied to {REPO_ID}.\n"
                "Request access at: https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail"
            )
        sys.exit(f"Download failed: {exc}")

    print(f"\nDone. Files saved to {DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download KITScenes-LongTail dataset splits."
    )
    parser.add_argument(
        "--splits",
        choices=["train", "test"],
        required=True,
        help="Split to download: train (~245 MB) or test (~34 GB).",
    )
    args = parser.parse_args()
    download_split(args.splits)


if __name__ == "__main__":
    main()
