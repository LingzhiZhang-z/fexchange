import numpy as np

from fexchange.spectrum.ground import build_W


def test_cef_mode_returns_symmetry_meta(monkeypatch):
    cfg = {
        "paths": {"kramer_file": "unused"},
        "sources": {"kramer_source": "cef", "kramer_name": "demo"},
        "model": {"symmetry": "Oh", "B4": 0.0, "B6": 0.0},
    }
    state = {
        "soc0": {"J0": 3.5, "n_j": 8},
        "lsjm_1": {"j_order_id": "jm_asc_v1"},
    }

    monkeypatch.setattr(
        "fexchange.spectrum.ground.build_hcef_matrix_J",
        lambda J, B, symmetry, mode_q3: np.zeros((8, 8), dtype=complex),
    )
    monkeypatch.setattr(
        "fexchange.spectrum.ground.select_kramers_doublet",
        lambda J, evals, evecs, c_g=1.0, point_group="Oh", symmetry_meta=None: {
            "kramer_vectors": np.eye(8, 2, dtype=complex),
            "gauge_meta": {"ok": True},
            "symmetry_meta": symmetry_meta,
        },
    )
    monkeypatch.setattr(
        "fexchange.spectrum.ground.analyze_cef_symmetry",
        lambda J, point_group, **kwargs: {
            "irrep_primary": "Gamma6",
            "irrep_display": "Γ6",
            "irrep_aliases": [],
            "mapping_unverified": True,
            "allowed_multipoles": ["magnetic_dipole"],
            "excited_irreps": [],
        },
    )
    out = build_W(cfg, state, n_ele=1)
    assert "symmetry_meta" in out
    assert out["symmetry_meta"]["irrep_primary"] == "Gamma6"
