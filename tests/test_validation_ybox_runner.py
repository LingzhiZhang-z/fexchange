from pathlib import Path
import importlib.util

import numpy as np

from tests.validation import run_ybox_prb_particle as runner


def _case(prefix: str) -> runner.LegacyCase:
    return runner.LegacyCase(
        material="YbOBr",
        prefix=prefix,
        hr_relpath="dummy_hr.dat",
        slater_ratios=(12.327, 7.757, 5.587),
        U_list_eV=(4.0, 6.0, 8.0),
        ratio_list=(0.1,),
        spinor=True,
        ndims=(14, 14, 1),
        atoms0=(0, 1, 0, 1),
        cells=((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
        input_path=Path(prefix + ".dat"),
    )


def test_case_family_detects_first_and_second() -> None:
    assert runner.case_family("YbOBr_soc_1st_bond1") == "1st"
    assert runner.case_family("YbOBr_soc_2nd_bond1") == "2nd"


def test_material_family_projectors_split_first_and_second(tmp_path, monkeypatch) -> None:
    cases = [
        _case("YbOBr_soc_1st_bond1"),
        _case("YbOBr_soc_1st_bond2"),
        _case("YbOBr_soc_2nd_bond1"),
        _case("YbOBr_soc_2nd_bond2"),
    ]

    def fake_extract(case: runner.LegacyCase, case_dir: Path) -> tuple[Path, Path]:
        case_dir.mkdir(parents=True, exist_ok=True)
        family = runner.case_family(case.prefix)
        value = 1.0 if family == "1st" else 2.0
        h_local_path = case_dir / "h_local.txt"
        np.savetxt(h_local_path, np.eye(2) * value)
        hopping_path = case_dir / "hopping.txt"
        np.savetxt(hopping_path, np.eye(2))
        return h_local_path, hopping_path

    def fake_build(h_local_path: Path, *, out_dir: Path, kramer_name: str, **_kwargs) -> tuple[Path, float]:
        out_dir.mkdir(parents=True, exist_ok=True)
        projector = out_dir / f"{kramer_name}.npz"
        np.savez(projector, projector=np.eye(2))
        zeta = 391.0 if "1st" in kramer_name else 392.0
        return projector, zeta

    monkeypatch.setattr(runner, "extract_h_local_and_hopping", fake_extract)
    monkeypatch.setattr(runner, "build_projector_from_h_local", fake_build)

    info = runner.material_family_projectors(cases, tmp_path)

    assert set(info) == {"1st", "2nd"}
    assert info["1st"]["zeta_meV"] == 391.0
    assert info["2nd"]["zeta_meV"] == 392.0
    assert info["1st"]["consistency"]["h_local_max_diff"] == 0.0
    assert info["2nd"]["consistency"]["h_local_max_diff"] == 0.0
    assert Path(info["1st"]["projector_file"]).name == "YbOBr_1st_particle.npz"
    assert Path(info["2nd"]["projector_file"]).name == "YbOBr_2nd_particle.npz"


def test_prb_canonical_gauge_removes_symmetry_leakage() -> None:
    J_mu = np.array(
        [
            [1.131696, -0.00186, 0.361401],
            [-0.00186, 1.151126, 0.007834],
            [0.361401, 0.007834, 2.295936],
        ],
        dtype=float,
    )
    raw = runner.extract_prb_channels(J_mu)
    assert raw["symmetry_leakage"] > 0.3

    J_can, meta = runner.canonicalize_prb_exchange(J_mu)
    fixed = runner.extract_prb_channels(J_can)

    assert meta["axis_rule"] == "min_abs_z_eigenvector"
    assert fixed["symmetry_leakage"] < 1.0e-10


def test_prb_plot_styles_match_reference_channel_labels() -> None:
    styles = runner.prb_plot_styles()
    assert styles["colors"]["Gamma"] == "#f08a24"
    assert styles["colors"]["Gamma_p"] == "#5566cc"
    assert styles["labels"]["Gamma"] == r"$\Gamma$"
    assert styles["labels"]["Gamma_p"] == r"$\Gamma'$"


def test_build_prb_projector_literature_matches_expected_gamma7(tmp_path) -> None:
    projector = runner.build_prb_projector_literature(tmp_path)
    data = np.load(projector)
    expected = np.array(
        [
            [0.0, -1.0 / np.sqrt(6.0)],
            [np.sqrt(5.0 / 6.0), 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, np.sqrt(5.0 / 6.0)],
            [-1.0 / np.sqrt(6.0), 0.0],
        ],
        dtype=complex,
    )
    assert data["ground_irrep"].item() == "Gamma7"
    assert data["kramer_name"].item() == "prb_gamma7_literature"
    assert np.allclose(data["W"], expected)


def test_build_prb_hopping_uses_identical_spin_blocks(tmp_path) -> None:
    hopping = runner.build_prb_hopping(tmp_path, "K")
    data = np.loadtxt(hopping, dtype=complex)
    assert data.shape == (14, 14)
    up = data[1::2, 1::2]
    dn = data[0::2, 0::2]
    assert np.allclose(dn, up)

    spec = importlib.util.spec_from_file_location(
        "sk_pf",
        runner.TOOLS_DIR / "slater_koster_pf.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cubic_7x7 = module.literature_a2pro3_transfer_cubic_7x7("K")
    cubic = module.literature_a2pro3_transfer_cubic("K")
    assert cubic_7x7.shape == (7, 7)
    assert cubic.shape == (14, 14)
    spinor_cubic = np.zeros((14, 14), dtype=complex)
    for i in range(7):
        for j in range(7):
            spinor_cubic[2 * i, 2 * j] = cubic_7x7[i, j]
            spinor_cubic[2 * i + 1, 2 * j + 1] = cubic_7x7[i, j]
    assert np.allclose(cubic, spinor_cubic)
    u = module._build_U_c2y(spinor=True)
    expected = u @ spinor_cubic @ u.conj().T
    assert np.allclose(data, expected)


def test_build_prb_hopping_calls_slater_koster_tool(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run_cmd(cmd: list[str], *, cwd: Path):
        calls.append(cmd)
        np.savetxt(tmp_path / "prb_t_mu.dat", np.eye(14, dtype=complex), fmt="%.12f")
        class Dummy:
            stdout = ""
        return Dummy()

    monkeypatch.setattr(runner, "run_cmd", fake_run_cmd)
    runner.build_prb_hopping(tmp_path, "K")

    assert calls
    assert calls[0][1].endswith("slater_koster_pf.py")
    assert calls[0][2] == "export"
