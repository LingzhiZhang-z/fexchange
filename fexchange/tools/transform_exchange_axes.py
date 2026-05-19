"""Rotate stage-local exchange.txt matrices between spin axes."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fexchange.io.disk import write_exchange_matrix_txt


_FLOAT_SPLIT_RE = re.compile(r"[\s,;]+")


@dataclass(frozen=True)
class ExchangeRows:
    labels: list[str]
    residuals: NDArray[np.float64]
    matrices: NDArray[np.float64]


def parse_axes_matrix(value: str) -> NDArray[np.float64]:
    """Parse a 3x3 axes matrix.

    The matrix is read in row-major order. Its columns are the target axes
    expressed in the input exchange frame.
    """
    tokens = [token for token in _FLOAT_SPLIT_RE.split(value.strip()) if token]
    if len(tokens) != 9:
        raise ValueError("axes matrix must contain exactly 9 numbers")
    matrix = np.asarray([float(token) for token in tokens], dtype=float).reshape(3, 3)
    gram = matrix.T @ matrix
    if not np.allclose(gram, np.eye(3), atol=1.0e-10, rtol=0.0):
        raise ValueError("axes matrix columns must be orthonormal")
    if not np.isclose(abs(float(np.linalg.det(matrix))), 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("axes matrix must be nonsingular with determinant magnitude 1")
    return matrix


def read_exchange_matrix_txt(path: Path) -> ExchangeRows:
    """Read FOPT/SOPT stage-local exchange.txt rows."""
    labels: list[str] = []
    residuals: list[float] = []
    matrices: list[NDArray[np.float64]] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 11:
            raise ValueError(f"{path}:{lineno}: expected 11 columns, got {len(parts)}")
        labels.append(parts[0])
        residuals.append(float(parts[1]))
        matrices.append(np.asarray([float(token) for token in parts[2:]], dtype=float).reshape(3, 3))
    if not labels:
        raise ValueError(f"{path}: no exchange rows found")
    return ExchangeRows(labels, np.asarray(residuals, dtype=float), np.asarray(matrices, dtype=float))


def transform_exchange_rows(
    rows: ExchangeRows,
    axes_i: NDArray[np.float64],
    axes_j: NDArray[np.float64],
) -> ExchangeRows:
    """Apply J_out = A_i.T @ J_in @ A_j to every exchange row."""
    matrices = np.asarray([axes_i.T @ matrix @ axes_j for matrix in rows.matrices], dtype=float)
    return ExchangeRows(list(rows.labels), np.asarray(rows.residuals, dtype=float), matrices)


def write_exchange_rows(path: Path, rows: ExchangeRows) -> Path:
    """Write transformed rows with the repository exchange.txt format."""
    process_labels = rows.labels[1:] if len(rows.labels) > 1 else None
    process_matrices = rows.matrices[1:] if len(rows.labels) > 1 else None
    process_residuals = rows.residuals[1:] if len(rows.labels) > 1 else None
    return write_exchange_matrix_txt(
        path,
        J_mu=rows.matrices[0],
        mapping_residual=float(rows.residuals[0]),
        process_labels=process_labels,
        J_mu_processes=process_matrices,
        mapping_residual_processes=process_residuals,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transform exchange.txt matrices with J_out = A_i.T @ J_in @ A_j. "
            "Each axes matrix is 3x3 row-major; columns are output axes in input coordinates."
        )
    )
    parser.add_argument("input", type=Path, help="Input exchange.txt path")
    parser.add_argument("output", type=Path, help="Output transformed exchange.txt path")
    parser.add_argument(
        "--axes-i",
        required=True,
        help='Site-i axes matrix, e.g. "1,0,0;0,1,0;0,0,1"',
    )
    parser.add_argument(
        "--axes-j",
        help="Site-j axes matrix. Defaults to --axes-i.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    axes_i = parse_axes_matrix(args.axes_i)
    axes_j = parse_axes_matrix(args.axes_j) if args.axes_j is not None else axes_i
    rows = read_exchange_matrix_txt(args.input)
    transformed = transform_exchange_rows(rows, axes_i, axes_j)
    write_exchange_rows(args.output, transformed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
