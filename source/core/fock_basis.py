from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List

import numpy as np

from .orbitals import ORBITAL_ORDER_ID


@dataclass(frozen=True)
class FockBasis:
    n: int
    dets: np.ndarray
    index: Dict[int, int]
    basis_id: str


def popcount(x: int) -> int:
    return int(x.bit_count())


def iter_dets(n: int, n_orb: int = 14) -> Iterable[int]:
    for occ in combinations(range(n_orb), n):
        det = 0
        for p in occ:
            det |= 1 << p
        yield det


def build_fock_basis(n: int, n_orb: int = 14) -> FockBasis:
    dets_list: List[int] = list(iter_dets(n, n_orb))
    dets = np.array(sorted(dets_list), dtype=np.int64)
    index = {int(det): i for i, det in enumerate(dets)}
    basis_id = f"fock{n_orb}_n{n}_lex_v1"
    return FockBasis(n=n, dets=dets, index=index, basis_id=basis_id)
