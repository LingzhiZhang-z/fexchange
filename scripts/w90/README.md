# W90 bond workflow

These scripts are thin workflow helpers. They do not modify or wrap the
numerical implementations in `fexchange.tools`.

Per bond:

```bash
python scripts/w90/prepare_extract.py path/to/bond_info.txt
python -m fexchange.tools.w90_extract path/to/bond/w90/extract.toml

python scripts/w90/prepare_downfold.py path/to/bond/w90
python -m fexchange.tools.w90_downfold path/to/bond/w90/downfold.toml
```

Per material, run CEF once from a chosen representative bond:

```bash
scripts/w90/run_cef_once.sh path/to/material J1-z
```

This keeps:

```text
material/cef/onsite_params.txt
material/cef/cef_REChX_C3v_sin.toml
material/cef/source.txt
```

Run CEF once for every material:

```bash
scripts/w90/run_cef_all.sh
```

The CEF-all script chooses `x` when present, otherwise `J1-z`, and stops if
neither representative bond exists. It also converts each generated CEF TOML
into:

```text
material/cef/kramer_projector.txt
```

`prepare_downfold.py` computes bond-local `Delta` and ligand `lambda_p` from
that bond's `onsite.txt`. The material-level CEF fit is intentionally separate.

Run the serial extract/downfold workflow for all materials:

```bash
scripts/w90/run_extract_downfold_all.sh
```

The all-material script stops on the first failed material or bond. CEF is not
run by this script; run `run_cef_once.sh` separately for the representative bond.
