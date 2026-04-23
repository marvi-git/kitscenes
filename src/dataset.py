"""Parquet dataset loader for KITScenes LongTail.

Returns plain Python dicts so callers don't need to import pyarrow.
Requires pyarrow (listed in requirements.txt).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

Variant = Literal["processed", "raw"]


def _shard_paths(split: str, variant: Variant) -> list[Path]:
    prefix = f"{split}_raw" if variant == "raw" else split
    return sorted(DATA_DIR.joinpath(split).glob(f"{prefix}-*.parquet"))


def load_instance(split: str, index: int, *, variant: Variant = "processed") -> dict:
    """Return the row at *index* as a plain Python dict.

    Reads only the shard that contains the requested row.
    """
    shards = _shard_paths(split, variant)
    if not shards:
        raise FileNotFoundError(
            f"No parquet files found for split='{split}' variant='{variant}' "
            f"under {DATA_DIR / split}. Run scripts/download_data.py first."
        )

    offset = index
    for path in shards:
        n = pq.read_metadata(str(path)).num_rows
        if offset < n:
            table = pq.read_table(str(path))
            return table.slice(offset, 1).to_pylist()[0]
        offset -= n

    total = sum(pq.read_metadata(str(p)).num_rows for p in shards)
    raise IndexError(f"index {index} out of range for split '{split}' ({total} rows)")


def iter_instances(
    split: str,
    *,
    variant: Variant = "processed",
    limit: int | None = None,
    batch_size: int = 32,
) -> Iterator[dict]:
    """Yield rows as plain Python dicts without loading the whole split in memory."""
    shards = _shard_paths(split, variant)
    if not shards:
        raise FileNotFoundError(
            f"No parquet files found for split='{split}' variant='{variant}' "
            f"under {DATA_DIR / split}. Run scripts/download_data.py first."
        )

    count = 0
    for path in shards:
        for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                if limit is not None and count >= limit:
                    return
                yield row
                count += 1
