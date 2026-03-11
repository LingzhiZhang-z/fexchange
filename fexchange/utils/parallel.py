"""
MPI wrapper with serial fallback.

Spec reference: 00-06-MPI_PARALLEL_RUNTIME.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("fexchange")


class ParallelContext:
    """Runtime parallel context with MPI or serial fallback (00-06 §2-3)."""

    def __init__(self) -> None:
        self.parallel_enabled = False
        self.parallel_backend = "serial"
        self.world_size = 1
        self.rank = 0
        self.root_rank = 0
        self.local_rank = 0
        self.gather_policy = "gather_to_root"
        self.shard_axis = "root_serial"
        self.shard_ranges: list[dict[str, Any]] = [{"rank": 0, "scope": "full"}]

        # Only attempt MPI initialisation if runtime env indicates an MPI launch.
        # This avoids side effects/noise in normal serial execution.
        mpi_world_size = _detect_mpi_world_size()
        if mpi_world_size > 1:
            try:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                self.world_size = comm.Get_size()
                self.rank = comm.Get_rank()
                self.local_rank = int(
                    os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
                    or os.environ.get("MPI_LOCALRANKID")
                    or "0"
                )
                if self.world_size > 1:
                    self.parallel_enabled = True
                    self.parallel_backend = "mpi4py"
                    self._comm = comm
            except ImportError:
                pass

        logger.info(
            "Parallel context: backend=%s, world_size=%d, rank=%d",
            self.parallel_backend, self.world_size, self.rank,
        )

    @property
    def is_root(self) -> bool:
        return self.rank == self.root_rank

    @property
    def can_persist(self) -> bool:
        """Only root rank may write final artifacts (00-06 §4)."""
        return self.is_root

    def barrier(self) -> None:
        if self.parallel_enabled:
            self._comm.Barrier()

    def bcast(self, data: Any, root: int = 0) -> Any:
        if self.parallel_enabled:
            return self._comm.bcast(data, root=root)
        return data

    def gather(self, data: Any, root: int = 0) -> list[Any] | None:
        if self.parallel_enabled:
            return self._comm.gather(data, root=root)
        return [data]

    def set_sharding(self, shard_axis: str, shard_ranges: list[dict[str, Any]]) -> None:
        """Set effective sharding metadata for current stage."""
        self.shard_axis = shard_axis
        self.shard_ranges = shard_ranges

    def meta(self) -> dict[str, Any]:
        """Return parallel metadata (00-06 §3)."""
        return {
            "parallel_enabled": self.parallel_enabled,
            "parallel_backend": self.parallel_backend,
            "world_size": self.world_size,
            "rank": self.rank,
            "root_rank": self.root_rank,
            "local_rank": self.local_rank,
            "gather_policy": self.gather_policy,
            "shard_axis": self.shard_axis,
            "shard_ranges": self.shard_ranges,
        }


def _detect_mpi_world_size() -> int:
    for key in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MPI_WORLD_SIZE",
        "WORLD_SIZE",
    ):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return 1
