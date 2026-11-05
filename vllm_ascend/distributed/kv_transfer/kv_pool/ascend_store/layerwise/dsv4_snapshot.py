# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Logical snapshot identity shared by the producer and resident decoder."""

from dataclasses import dataclass
from uuid import UUID

DSV4_HANDOFF_KEY = "ascend_dsv4_snapshot"
_SNAPSHOT_VERSION = 1


def public_history_suffix(block_hash: str, producer_replica: int | None) -> str:
    """Prevent concurrent DP writers from opening the same public object."""
    return block_hash if producer_replica is None else f"producer-dp:{producer_replica}:{block_hash}"


@dataclass(frozen=True)
class DSV4Snapshot:
    generation: str
    end_token: int
    # Each group names the logical page indices stored privately. Destination
    # block IDs come from the reader's live table and never enter the key.
    private_pages: tuple[tuple[int, ...], ...]
    producer_replica: int | None = None

    def __post_init__(self) -> None:
        if UUID(hex=self.generation).hex != self.generation or self.end_token <= 0:
            raise ValueError("Invalid DSV4 snapshot identity")
        for pages in self.private_pages:
            if tuple(sorted(set(pages))) != pages or any(page < 0 for page in pages):
                raise ValueError("Invalid DSV4 snapshot page indices")

    def suffix(self, group: int, page: int) -> str:
        return f"snapshot-v{_SNAPSHOT_VERSION}:{self.generation}:{self.end_token}:{group}:{page}"

    def to_wire(self) -> dict:
        return {
            "version": _SNAPSHOT_VERSION,
            "generation": self.generation,
            "end_token": self.end_token,
            "producer_replica": self.producer_replica,
            "private_pages": [list(pages) for pages in self.private_pages],
        }

    @classmethod
    def from_wire(cls, metadata: dict, group_count: int) -> "DSV4Snapshot":
        if type(metadata.get("version")) is not int or metadata["version"] != _SNAPSHOT_VERSION:
            raise ValueError("Unsupported DSV4 snapshot version")
        pages = metadata.get("private_pages")
        end = metadata.get("end_token")
        generation = metadata.get("generation")
        replica = metadata.get("producer_replica")
        if (
            not isinstance(pages, list)
            or len(pages) != group_count
            or any(not isinstance(row, list) or any(type(page) is not int for page in row) for row in pages)
            or type(end) is not int
            or not isinstance(generation, str)
            or (replica is not None and (type(replica) is not int or replica < 0))
        ):
            raise ValueError("Malformed DSV4 snapshot metadata")
        return cls(generation, end, tuple(tuple(row) for row in pages), replica)


def mutable_snapshot_start(end_token: int, block_size: int, retained_tokens: int | None) -> int:
    """Keep the full trailing window and any block overlapping its start."""
    return 0 if retained_tokens is None else max(0, end_token - retained_tokens) // block_size


def select_private_snapshot_pages(
    block_ids: tuple[list[int], ...],
    block_sizes: list[int],
    end_token: int,
    mutable_groups: set[int],
    history_ratios: dict[int, int] | None = None,
    mutable_windows: dict[int, int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Select the active private pages from the scheduler's live block table.

    Mutable groups need every resident page at the exact endpoint. Immutable
    history uses public keys for complete pages and a private key for the tail.
    Null block IDs identify logical positions that the scheduler has discarded.
    """
    if len(block_ids) != len(block_sizes) or end_token <= 0 or any(size <= 0 for size in block_sizes):
        raise ValueError("Invalid DSV4 snapshot block geometry")
    selections = []
    for group, (ids, span) in enumerate(zip(block_ids, block_sizes, strict=True)):
        complete, tail = divmod(end_token, span)
        start = (
            mutable_snapshot_start(end_token, span, (mutable_windows or {}).get(group))
            if group in mutable_groups
            else complete
        )
        ratio = (history_ratios or {}).get(group, 1)
        stop = complete + (tail >= ratio)
        selections.append(tuple(index for index in range(start, min(stop, len(ids))) if ids[index] != 0))
    return tuple(selections)
