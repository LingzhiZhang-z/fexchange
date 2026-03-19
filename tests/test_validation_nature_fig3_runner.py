from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from fexchange.io.run_input import load_run_input
from tests.validation import run_nature_fig3_particle as runner


def test_fig3_cases_cover_requested_panels() -> None:
    active = {case.key for case in runner.active_cases()}
    assert active == {"f1_ce_g7", "f3_nd_g6", "f11_er_g7", "f13_yb_g6"}

    asset_only = {case.key for case in runner.asset_only_cases()}
    assert asset_only == {"f5_sm_g7", "f9_dy_g6", "f9_dy_g7"}


def test_build_projector_matches_expected_shapes_and_is_orthonormal() -> None:
    expected_rows = {
        "f1_ce_g7": 6,
        "f3_nd_g6": 10,
        "f11_er_g7": 16,
        "f13_yb_g6": 8,
        "f5_sm_g7": 6,
        "f9_dy_g6": 16,
        "f9_dy_g7": 16,
    }
    for case in runner.all_cases():
        W = runner.build_projector(case)
        assert W.shape == (expected_rows[case.key], 2)
        np.testing.assert_allclose(W.conj().T @ W, np.eye(2), atol=1.0e-12)


def test_extract_channels_canonicalizes_exchange() -> None:
    J_mu = np.array(
        [
            [1.131696, -0.00186, 0.361401],
            [-0.00186, 1.151126, 0.007834],
            [0.361401, 0.007834, 2.295936],
        ],
        dtype=float,
    )
    raw = runner.extract_channels(J_mu)
    assert raw["symmetry_leakage"] > 0.3

    J_can, meta = runner.canonicalize_exchange(J_mu)
    fixed = runner.extract_channels(J_can)
    assert meta["axis_rule"] == "min_abs_z_eigenvector"
    assert fixed["symmetry_leakage"] < 1.0e-10


def test_write_run_input_includes_nm1_np1_branch_overrides(tmp_path: Path) -> None:
    case = runner.case_by_key("f11_er_g7")
    hopping = tmp_path / "t.dat"
    projector = tmp_path / "w.dat"
    np.savetxt(hopping, np.eye(14, dtype=np.complex128), fmt="%.12f")
    np.savetxt(projector, np.eye(16, 2, dtype=np.complex128), fmt="%.12f")

    run_input = tmp_path / "run_input.toml"
    runner.write_run_input(
        case,
        ratio=0.7,
        hopping_file=hopping,
        projector_file=projector,
        output_path=run_input,
    )
    cfg = load_run_input(run_input)

    assert cfg["_branches"]["n"]["physics"]["n_ele"] == 11
    assert cfg["_branches"]["nm1"]["physics"]["n_ele"] == 10
    assert cfg["_branches"]["np1"]["physics"]["n_ele"] == 12
    assert cfg["sopt"]["energy_reference"] == "zero"
    assert cfg["_branches"]["nm1"]["sopt"]["offset"] == pytest.approx(27410.81246392870)
    assert cfg["_branches"]["np1"]["sopt"]["offset"] == pytest.approx(18603.29259035310)
    assert cfg["_branches"]["nm1"]["sopt"]["Jh"] == pytest.approx(800.0)
    assert cfg["_branches"]["np1"]["sopt"]["Jh"] == pytest.approx(500.0)


def test_reference_curves_exist_and_are_sorted() -> None:
    for case in runner.active_cases():
        ref = runner.load_reference_curve(case)
        assert ref.shape == (10, 5)
        np.testing.assert_allclose(ref[:, 0], np.linspace(0.1, 1.0, 10), atol=1.0e-12)


def test_static_assets_exist_and_are_not_executed_by_default() -> None:
    for case in runner.asset_only_cases():
        paths = runner.static_asset_paths(case)
        assert paths["projector"].exists()
        assert paths["run_input"].exists()

    default_keys = {case.key for case in runner.default_run_cases()}
    assert "f5_sm_g7" not in default_keys
    assert "f9_dy_g6" not in default_keys
    assert "f9_dy_g7" not in default_keys


def test_fig3_assets_include_ratio_sweep_inputs_for_active_cases() -> None:
    for case in runner.active_cases():
        paths = runner.fig3_asset_paths(case)
        assert paths["projector"].exists()
        assert paths["root"].name == case.key
        assert len(paths["run_inputs"]) == len(runner.RATIO_VALUES)
        for ratio in runner.RATIO_VALUES:
            assert paths["run_inputs"][ratio].exists()


def test_table_assets_cover_active_and_single_point_cases() -> None:
    expected = {
        "f1_ce_g7",
        "f3_nd_g6",
        "f5_sm_g7",
        "f9_dy_g6",
        "f9_dy_g7",
        "f11_er_g7",
        "f13_yb_g6",
    }
    actual = {case.key for case in runner.table_estimate_cases()}
    assert actual == expected

    for case in runner.table_estimate_cases():
        paths = runner.table_asset_paths(case)
        assert paths["projector"].exists()
        assert paths["run_input"].exists()


def test_table_estimate_scale_matches_article_defaults() -> None:
    assert runner.table_estimate_scale_mev() == pytest.approx(15.00625)
    assert runner.table_estimate_scale_mev(tpfsigma_eV=0.35, delta_pf_eV=2.0) == pytest.approx(3.7515625)


def test_shell_and_gnuplot_entrypoints_exist_and_are_valid() -> None:
    assert runner.NATURE_SHELL_RUNNER.exists()
    assert runner.NATURE_GNUPLOT_SCRIPT.exists()

    proc = subprocess.run(
        ["bash", "-n", str(runner.NATURE_SHELL_RUNNER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    shell_text = runner.NATURE_SHELL_RUNNER.read_text(encoding="utf-8")
    assert "TABLE2_OUTPUT_ROOT" in shell_text
    assert "TABLE2_WORK_ROOT" in shell_text
    assert "--skip-table2" in shell_text
    assert "--table2-case" in shell_text
    assert "--prepare-only" in shell_text
    assert "--strict-table" not in shell_text
    assert 'rm -rf "$OUTPUT_ROOT"' not in shell_text


def test_intermediate_work_roots_are_separate_from_curated_outputs() -> None:
    assert runner.FIG3_WORK_ROOT == runner.OUTPUT_ROOT / "workdirs" / "fig3"
    assert runner.TABLE2_WORK_ROOT == runner.OUTPUT_ROOT / "workdirs" / "table2"
    assert runner.TABLE_WORK_ROOT == runner.OUTPUT_ROOT / "workdirs" / "table_estimates"


def test_build_complex_hopping_supports_absolute_pf_sigma_and_delta_scaling(tmp_path: Path) -> None:
    unit = runner.build_complex_hopping(
        0.7,
        tmp_path / "unit.dat",
        pf_sigma_eV=1.0,
        delta_pf_eV=1.0,
    )
    strict = runner.build_complex_hopping(
        0.7,
        tmp_path / "strict.dat",
        pf_sigma_eV=0.35,
        delta_pf_eV=2.0,
    )

    unit_data = np.loadtxt(unit, dtype=complex)
    strict_data = np.loadtxt(strict, dtype=complex)

    np.testing.assert_allclose(strict_data, (0.35 ** 2 / 2.0) * unit_data, atol=1.0e-12)


def test_table_strict_defaults_to_active_cases_and_separate_outputs() -> None:
    assert {case.key for case in runner.table_strict_cases()} == {
        "f1_ce_g7",
        "f3_nd_g6",
        "f11_er_g7",
        "f13_yb_g6",
    }
    assert runner.TABLE_STRICT_OUTPUT_ROOT == runner.OUTPUT_ROOT / "table_strict"
    assert runner.TABLE_STRICT_WORK_ROOT == runner.OUTPUT_ROOT / "workdirs" / "table_strict"


def test_table_strict_assets_exist_and_embed_selected_hopping() -> None:
    expected_hopping = runner.build_complex_hopping(
        runner.TABLE_RATIO,
        pf_sigma_eV=runner.TABLE_TPFSIGMA_EV,
        delta_pf_eV=runner.TABLE_DELTA_PF_EV,
    )
    for case in runner.table_strict_cases():
        paths = runner.table_strict_asset_paths(case)
        assert paths["projector"].exists()
        assert paths["run_input"].exists()
        assert paths["hopping"].exists()
        assert paths["root"] == runner.TABLE_STRICT_ASSET_ROOT / case.key
        assert paths["hopping"] == expected_hopping
        run_input_text = paths["run_input"].read_text(encoding="utf-8")
        assert str(expected_hopping).replace("\\", "/") in run_input_text


def test_table_strict_assets_write_runtime_scaled_physical_hopping(tmp_path: Path) -> None:
    unit = runner.build_complex_hopping(runner.TABLE_RATIO, tmp_path / "unit.dat")
    strict = runner.table_strict_asset_paths(runner.case_by_key("f1_ce_g7"))["hopping"]

    unit_data = np.loadtxt(unit, dtype=complex)
    strict_data = np.loadtxt(strict, dtype=complex)
    expected_scale = np.sqrt(1000.0) * runner.TABLE_TPFSIGMA_EV ** 2 / runner.TABLE_DELTA_PF_EV

    np.testing.assert_allclose(strict_data, expected_scale * unit_data, atol=1.0e-12)


def test_table2_defaults_to_all_cases_and_uses_dedicated_roots() -> None:
    assert {case.key for case in runner.table2_cases()} == {
        "f1_ce_g7",
        "f3_nd_g6",
        "f5_sm_g7",
        "f9_dy_g6",
        "f9_dy_g7",
        "f11_er_g7",
        "f13_yb_g6",
    }
    assert {case.key for case in runner.table2_default_run_cases()} == {
        "f1_ce_g7",
        "f3_nd_g6",
        "f11_er_g7",
        "f13_yb_g6",
    }
    assert runner.TABLE2_ASSET_ROOT == runner.DATA_ROOT / "table2"
    assert runner.TABLE2_OUTPUT_ROOT == runner.OUTPUT_ROOT / "table2"
    assert runner.TABLE2_WORK_ROOT == runner.OUTPUT_ROOT / "workdirs" / "table2"


def test_table2_literature_reference_exists_and_matches_article_rows() -> None:
    table = runner.load_table2_literature()

    assert set(table) == {
        "f1_ce_g7",
        "f3_nd_g6",
        "f5_sm_g7",
        "f9_dy_g6",
        "f9_dy_g7",
        "f11_er_g7",
        "f13_yb_g6",
    }
    assert table["f1_ce_g7"]["J_meV"] == pytest.approx(0.121)
    assert table["f1_ce_g7"]["K_meV"] == pytest.approx(0.172)
    assert table["f13_yb_g6"]["abs_K_over_J"] == pytest.approx(0.00303)


def test_table2_assets_exist_and_embed_selected_hopping() -> None:
    expected_hopping = runner.build_complex_hopping(
        runner.TABLE_RATIO,
        pf_sigma_eV=runner.TABLE_TPFSIGMA_EV,
        delta_pf_eV=runner.TABLE_DELTA_PF_EV,
        output_scale=runner.STRICT_TABLE_RUNTIME_HOPPING_SCALE,
    )
    for case in runner.table2_cases():
        paths = runner.table2_asset_paths(case)
        assert paths["projector"].exists()
        assert paths["run_input"].exists()
        assert paths["hopping"].exists()
        assert paths["root"] == runner.TABLE2_ASSET_ROOT / case.key
        assert paths["hopping"] == expected_hopping
        run_input_text = paths["run_input"].read_text(encoding="utf-8")
        assert str(expected_hopping).replace("\\", "/") in run_input_text


def test_table_estimate_from_point_result_uses_fig3_unit_scale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case = runner.case_by_key("f3_nd_g6")

    def fake_summary(_path: Path) -> dict[str, object]:
        return {
            "channels_raw": {
                "J": 1.0e-5,
                "K": 2.0e-5,
                "Gamma": 3.0e-5,
                "GammaPrime": 4.0e-5,
                "symmetry_leakage": 5.0e-5,
            },
            "channels_fig3": {
                "J": 1.0e-2,
                "K": 2.0e-2,
                "Gamma": 3.0e-2,
                "GammaPrime": 4.0e-2,
                "symmetry_leakage": 5.0e-2,
            },
            "mapping_residual": 0.0,
            "canonical_meta": {},
            "raw_J_mu": np.zeros((3, 3), dtype=float),
            "canonical_J_mu": np.zeros((3, 3), dtype=float),
        }

    monkeypatch.setattr(runner, "summarize_point_result", fake_summary)
    entry = runner.table_estimate_from_point_result(case, path=tmp_path / "point.txt")
    scale = runner.table_estimate_scale_mev()

    assert entry["J_meV"] == pytest.approx(1.0e-2 * scale)
    assert entry["K_meV"] == pytest.approx(2.0e-2 * scale)
    assert entry["Gamma_meV"] == pytest.approx(3.0e-2 * scale)
    assert entry["GammaPrime_meV"] == pytest.approx(4.0e-2 * scale)


def test_table_strict_from_point_result_uses_direct_runtime_scale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = runner.case_by_key("f3_nd_g6")

    def fake_summary(_path: Path) -> dict[str, object]:
        return {
            "channels_raw": {
                "J": 1.0e-5,
                "K": 2.0e-5,
                "Gamma": 3.0e-5,
                "GammaPrime": 4.0e-5,
                "symmetry_leakage": 5.0e-5,
            },
            "channels_fig3": {
                "J": 1.0e-2,
                "K": 2.0e-2,
                "Gamma": 3.0e-2,
                "GammaPrime": 4.0e-2,
                "symmetry_leakage": 5.0e-2,
            },
            "mapping_residual": 0.0,
            "canonical_meta": {},
            "raw_J_mu": np.zeros((3, 3), dtype=float),
            "canonical_J_mu": np.zeros((3, 3), dtype=float),
        }

    monkeypatch.setattr(runner, "summarize_point_result", fake_summary)
    entry = runner.table_strict_from_point_result(case, path=tmp_path / "point.txt")

    assert entry["scale_meV"] == pytest.approx(1.0)
    assert entry["J_meV"] == pytest.approx(1.0e-2)
    assert entry["K_meV"] == pytest.approx(2.0e-2)
    assert entry["Gamma_meV"] == pytest.approx(3.0e-2)
    assert entry["GammaPrime_meV"] == pytest.approx(4.0e-2)


def test_table2_from_point_result_includes_literature_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = runner.case_by_key("f1_ce_g7")

    def fake_summary(_path: Path) -> dict[str, object]:
        return {
            "channels_raw": {
                "J": 1.0e-5,
                "K": 2.0e-5,
                "Gamma": 3.0e-5,
                "GammaPrime": 4.0e-5,
                "symmetry_leakage": 5.0e-5,
            },
            "channels_fig3": {
                "J": 0.125,
                "K": 0.154,
                "Gamma": 0.0,
                "GammaPrime": 0.0,
                "symmetry_leakage": 0.0,
            },
            "mapping_residual": 0.0,
            "canonical_meta": {},
            "raw_J_mu": np.zeros((3, 3), dtype=float),
            "canonical_J_mu": np.zeros((3, 3), dtype=float),
        }

    monkeypatch.setattr(runner, "summarize_point_result", fake_summary)
    entry = runner.table2_from_point_result(case, path=tmp_path / "point.txt")

    assert entry["J_meV"] == pytest.approx(0.125)
    assert entry["literature_J_meV"] == pytest.approx(0.121)
    assert entry["delta_J_meV"] == pytest.approx(0.004)
    assert entry["literature_K_meV"] == pytest.approx(0.172)
    assert entry["delta_K_meV"] == pytest.approx(-0.018)


def test_build_complex_hopping_matches_expected_ratio_0p1_coefficients(tmp_path: Path) -> None:
    hopping = runner.build_complex_hopping(0.1, tmp_path / "t_mu.dat")
    data = np.loadtxt(hopping, dtype=complex)
    assert data.shape == (14, 14)

    values = sorted({
        float(np.real_if_close(x))
        for x in data.ravel()
        if abs(x) > 1.0e-12
    })
    assert values == pytest.approx(
        sorted([
            0.15309310892394865,
            0.039528470752104715,
            -0.00625,
            -0.030618621784789746,
            0.0075,
        ]),
        abs=1.0e-12,
    )


def test_compare_curves_accepts_ratio_subset() -> None:
    case = runner.case_by_key("f1_ce_g7")
    reference = runner.load_reference_curve(case)
    smoke = reference[:1].copy()

    compare = runner.compare_curves(smoke, reference)

    assert compare["max_abs"]["J"] == pytest.approx(0.0)
    assert compare["max_abs"]["K"] == pytest.approx(0.0)


def test_scale_channels_to_mev_scales_exchange_outputs_only() -> None:
    scaled = runner.scale_channels_to_mev(
        {
            "J": 0.1,
            "K": 0.2,
            "Gamma": 0.3,
            "GammaPrime": 0.4,
            "symmetry_leakage": 0.5,
        }
    )

    assert scaled["J"] == pytest.approx(100.0)
    assert scaled["K"] == pytest.approx(200.0)
    assert scaled["Gamma"] == pytest.approx(300.0)
    assert scaled["GammaPrime"] == pytest.approx(400.0)
    assert scaled["symmetry_leakage"] == pytest.approx(500.0)


def test_vacuum_and_full_shell_branches_zero_out_hund_and_soc(tmp_path: Path) -> None:
    for case_key, side in (("f1_ce_g7", "nm1"), ("f13_yb_g6", "np1")):
        case = runner.case_by_key(case_key)
        hopping = tmp_path / f"{case_key}_{side}_t.dat"
        projector = tmp_path / f"{case_key}_{side}_w.npz"
        np.savetxt(hopping, np.eye(14, dtype=np.complex128), fmt="%.12f")
        np.savez(projector, W=np.eye(case.j2 + 1, 2, dtype=np.complex128))

        run_input = tmp_path / "run_input.toml"
        runner.write_run_input(
            case,
            ratio=0.7,
            hopping_file=hopping,
            projector_file=projector,
            output_path=run_input,
        )
        cfg = load_run_input(run_input)
        branch = cfg["_branches"][side]

        assert branch["sopt"]["Jh"] == pytest.approx(0.0)
        assert branch["sopt"]["zeta"] == pytest.approx(0.0)
        assert branch["physics"]["F2"] == pytest.approx(0.0)
        assert branch["physics"]["F4"] == pytest.approx(0.0)
        assert branch["physics"]["F6"] == pytest.approx(0.0)
