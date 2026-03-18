from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump L4 NPZ tensors to plain text")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default="Heff_mu_abcd")
    args = parser.parse_args()

    data = np.load(Path(args.input))
    if args.key not in data:
        raise KeyError(f"Missing key {args.key!r} in {args.input}")
    array = np.asarray(data[args.key])
    if array.shape == (2, 2, 2, 2):
        array = array.reshape(4, 4)
    elif array.shape != (4, 4):
        raise ValueError(f"Unsupported shape for {args.key}: {array.shape}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write("# row col real imag\n")
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                value = complex(array[row, col])
                handle.write(f"{row} {col} {value.real:.12f} {value.imag:.12f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
