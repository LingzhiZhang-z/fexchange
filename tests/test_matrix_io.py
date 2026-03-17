from pathlib import Path

import numpy as np
import pytest

from fexchange.io.matrix import load_matrix_file
from fexchange.utils.errors import SchemaError


def test_load_matrix_file_accepts_txt_complex_matrix(tmp_path: Path) -> None:
    path = tmp_path / "t_mu.txt"
    matrix = np.array([[1 + 2j, 3 - 4j], [5 + 0j, -6j]], dtype=complex)
    np.savetxt(path, matrix, fmt="%.12f")

    loaded = load_matrix_file(path, preferred_key="t_mu")

    assert loaded.dtype == np.complex128
    assert loaded.shape == (2, 2)
    assert np.allclose(loaded, matrix)


def test_load_matrix_file_rejects_invalid_txt(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("not a matrix\n")

    with pytest.raises(SchemaError, match="Invalid text matrix file"):
        load_matrix_file(path, preferred_key="W")
