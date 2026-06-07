"""Parameter sweep: one resolved base input + a ``[sweep]`` table -> N cases.

The sweep acts on a complete parsed input. Bare physical columns such as ``U``,
``Jh``, ``offset``, and ``zeta`` are shorthand for setting the same leaf on all
three f sectors (``fsite``, ``fsite_nm1``, ``fsite_np1``). Qualified columns set
only the named path.

Parallelism: under a launcher (mpi4py, ``COMM_WORLD`` size > 1 as reported by MPI
itself) every rank runs an even static slice ``cases[rank::size]``; otherwise
everything runs in one process. Rank 0 warms the shared content-addressed core
once before the fan-out.

Progress: a plain-text file ``<output_root>/sweep_<stem>.txt`` -- a header, one
``ok``/``FAIL`` line per finished case, and a final ``# done: k/N`` line. Tail it
to watch a run live.

Spec: ``05-io/05-05-SWEEP_INPUT.md``.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any

from fexchange.io.run_input import load_run_input_from_dict, load_toml
from fexchange.utils.errors import InputError, RuntimeError_

logger = logging.getLogger("fexchange")

_BARE_TARGETS: dict[str, tuple[tuple[str, ...], ...]] = {
    "U": (("fsite",), ("fsite_nm1",), ("fsite_np1",)),
    "Jh": (("fsite",), ("fsite_nm1",), ("fsite_np1",)),
    "offset": (("fsite",), ("fsite_nm1",), ("fsite_np1",)),
    "zeta": (("fsite",), ("fsite_nm1",), ("fsite_np1",)),
    "Uplus": (("fsite_np1",),),
    "Uminus": (("fsite_nm1",),),
    "Delta": (("ligand", "1"), ("ligand", "2")),
    "U_p": (("ligand", "1"), ("ligand", "2")),
    "lambda_p": (("ligand", "1"), ("ligand", "2")),
}
_QUALIFIED_PATHS: frozenset[str] = frozenset(
    {
        "fsite.U",
        "fsite.Jh",
        "fsite.offset",
        "fsite.zeta",
        "fsite_nm1.U",
        "fsite_nm1.Jh",
        "fsite_nm1.offset",
        "fsite_nm1.zeta",
        "fsite_nm1.Uminus",
        "fsite_np1.U",
        "fsite_np1.Jh",
        "fsite_np1.offset",
        "fsite_np1.zeta",
        "fsite_np1.Uplus",
        "ligand.1.Delta",
        "ligand.1.U_p",
        "ligand.1.lambda_p",
        "ligand.2.Delta",
        "ligand.2.U_p",
        "ligand.2.lambda_p",
        "inputs.hopping_file",
        "inputs.kramer_file",
        "inputs.hcef_file",
        "runtime.run_name",
    }
)


# ---------------------------------------------------------------------------
# Parse the [sweep] table  (pure, no I/O)
# ---------------------------------------------------------------------------


def parse_sweep_table(base: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Parse ``[sweep].table`` into ``(paths, rows)``.

    Grammar: a multi-line string whose first non-blank line is an integer case
    count, the second is whitespace-separated EXACT dotted paths (the columns),
    and every remaining line is a whitespace-separated data row. A numeric cell
    becomes a ``float`` unless its column's leaf is ``run_name`` or ends in
    ``_file`` -- those stay raw strings so ``5`` is not mangled into ``5.0``.
    """
    sweep = base.get("sweep")
    if not isinstance(sweep, dict) or not isinstance(sweep.get("table"), str):
        raise InputError("FXE-INPUT-002", "sweep base requires a [sweep].table string")

    lines = [ln.strip() for ln in sweep["table"].splitlines() if ln.strip()]
    if len(lines) < 3:
        raise InputError(
            "FXE-INPUT-003",
            "[sweep].table needs a count line, a header line, and >=1 data row",
        )

    try:
        count = int(lines[0])
    except ValueError as exc:
        raise InputError(
            "FXE-INPUT-003", f"[sweep].table line 1 must be an integer count, got {lines[0]!r}"
        ) from exc
    if count < 1:
        raise InputError("FXE-INPUT-003", f"[sweep].table count must be >= 1, got {count}")

    paths = lines[1].split()
    if len(set(paths)) != len(paths):
        dupes = sorted({c for c in paths if paths.count(c) > 1})
        raise InputError("FXE-INPUT-003", f"[sweep].table has duplicate columns: {dupes}")
    for path in paths:
        _validate_sweep_path(path)
    _validate_no_target_conflicts(paths)
    if "runtime.run_name" not in paths:
        raise InputError("FXE-INPUT-003", "[sweep].table must include a runtime.run_name column")

    data = lines[2:]
    if count != len(data):
        raise InputError(
            "FXE-INPUT-003", f"[sweep].table count {count} != {len(data)} data rows"
        )
    rows: list[list[Any]] = []
    for i, line in enumerate(data):
        cells = line.split()
        if len(cells) != len(paths):
            raise InputError(
                "FXE-INPUT-003",
                f"[sweep].table row {i} has {len(cells)} values, expected {len(paths)}",
            )
        rows.append([_typed(p, c) for p, c in zip(paths, cells)])
    return paths, rows


def _typed(path: str, cell: str) -> Any:
    """A cell stays a raw string for run_name / ``*_file`` columns, else float-if-numeric."""
    leaf = path.split(".")[-1]
    if leaf == "run_name" or leaf.endswith("_file"):
        return cell
    try:
        return float(cell)
    except ValueError:
        return cell


# ---------------------------------------------------------------------------
# Build one case  (pure)
# ---------------------------------------------------------------------------


def build_case(base_cfg: dict[str, Any], paths: list[str], row: list[Any]) -> dict[str, Any]:
    """Deep-copy a complete base cfg and assign one sweep row."""
    cfg = copy.deepcopy(base_cfg)
    for path in paths:
        _validate_sweep_path(path)
    _validate_no_target_conflicts(paths)
    for path, value in zip(paths, row):
        if "." in path:
            _set_path(cfg, path.split("."), value)
            _apply_exclusion(cfg, tuple(path.split(".")))
        else:
            _set_bare_path(cfg, path, value)
    return cfg


def _set_bare_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    targets = _BARE_TARGETS.get(path)
    if targets is None:
        raise InputError("FXE-INPUT-003", f"unknown bare sweep field: {path}")
    for target in targets:
        _set_path(cfg, [*target, path], value)
        _apply_exclusion(cfg, (*target, path))


def _validate_sweep_path(path: str) -> None:
    if "." not in path:
        if path not in _BARE_TARGETS:
            raise InputError("FXE-INPUT-003", f"unknown bare sweep field: {path}")
        return
    if path not in _QUALIFIED_PATHS:
        raise InputError("FXE-INPUT-003", f"unsupported sweep path: {path}")


def _validate_no_target_conflicts(paths: list[str]) -> None:
    owners: dict[tuple[str, ...], str] = {}
    targets_by_path: dict[str, tuple[tuple[str, ...], ...]] = {}
    for path in paths:
        targets = _expanded_targets(path)
        targets_by_path[path] = targets
        for target in targets:
            previous = owners.get(target)
            if previous is not None:
                leaf = ".".join(target)
                raise InputError(
                    "FXE-INPUT-003",
                    f"sweep columns {previous!r} and {path!r} both set {leaf}; use exactly one",
                )
            owners[target] = path
    _validate_no_exclusion_conflicts(targets_by_path)


def _validate_no_exclusion_conflicts(targets_by_path: dict[str, tuple[tuple[str, ...], ...]]) -> None:
    owners = {
        target: path
        for path, targets in targets_by_path.items()
        for target in targets
    }
    for a, b in (
        (("fsite_np1", "Uplus"), ("fsite_np1", "offset")),
        (("fsite_nm1", "Uminus"), ("fsite_nm1", "offset")),
    ):
        if a in owners and b in owners:
            raise InputError(
                "FXE-INPUT-003",
                f"sweep columns {owners[a]!r} and {owners[b]!r} set mutually exclusive fields",
            )


def _expanded_targets(path: str) -> tuple[tuple[str, ...], ...]:
    if "." in path:
        return (tuple(path.split(".")),)
    targets = _BARE_TARGETS.get(path)
    if targets is None:
        raise InputError("FXE-INPUT-003", f"unknown bare sweep field: {path}")
    return tuple((*target, path) for target in targets)


def _set_path(node: dict[str, Any], segments: list[str], value: Any) -> None:
    """Assign ``node[seg0][seg1]...[segN] = value``, creating intermediate tables."""
    for key in segments[:-1]:
        child = node.get(key)
        if child is None:
            child = node[key] = {}
        elif not isinstance(child, dict):
            raise InputError(
                "FXE-INPUT-003", f"sweep path cannot descend into non-table at {key!r}"
            )
        node = child
    node[segments[-1]] = value


def _apply_exclusion(cfg: dict[str, Any], segments: tuple[str, ...]) -> None:
    if segments == ("fsite_np1", "Uplus"):
        cfg.get("fsite_np1", {}).pop("offset", None)
    elif segments == ("fsite_nm1", "Uminus"):
        cfg.get("fsite_nm1", {}).pop("offset", None)
    elif segments == ("fsite_np1", "offset"):
        cfg.get("fsite_np1", {}).pop("Uplus", None)
    elif segments == ("fsite_nm1", "offset"):
        cfg.get("fsite_nm1", {}).pop("Uminus", None)


def _run_names(paths: list[str], rows: list[list[Any]]) -> list[str]:
    """The per-row run_name values; reject empties and duplicates up front."""
    col = paths.index("runtime.run_name")
    names = [str(row[col]) for row in rows]
    positions: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        if not name.strip():
            raise InputError("FXE-INPUT-003", f"empty runtime.run_name at sweep row {i}")
        positions.setdefault(name, []).append(i)
    dups = {n: idx for n, idx in positions.items() if len(idx) > 1}
    if dups:
        detail = "; ".join(f"{n!r} at rows {idx}" for n, idx in sorted(dups.items()))
        raise InputError("FXE-INPUT-003", f"duplicate runtime.run_name in sweep: {detail}")
    return names


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_sweep(base_path: str, *, log_level: str) -> int:
    """Run a parameter sweep from *base_path* (a base run-input with ``[sweep]``)."""
    comm = _detect_comm()
    if comm is not None:
        _warn_oversubscription()

    raw_base = load_toml(base_path)
    paths, rows = parse_sweep_table(raw_base)
    names = _run_names(paths, rows)
    base_no_sweep = {k: v for k, v in raw_base.items() if k != "sweep"}
    output_base = _output_run_base(raw_base)
    base_cfg = load_run_input_from_dict(copy.deepcopy(base_no_sweep), output_run_base=output_base)
    output_root = _output_root(raw_base)
    progress = Path(output_root) / f"sweep_{Path(base_path).stem}.txt"
    ctx = (base_cfg, output_base, paths, rows, names, progress)

    if comm is None:
        return _run_serial(*ctx, log_level=log_level)
    return _run_mpi(comm, *ctx, log_level=log_level)


def _run_serial(
    base_cfg: dict[str, Any],
    output_base: str,
    paths: list[str],
    rows: list[list[Any]],
    names: list[str],
    progress: Path,
    *,
    log_level: str,
) -> int:
    _start_progress(progress, len(rows))
    results = [
        _run_case(i, names[i], base_cfg, output_base, paths, rows[i], progress, rank=None, log_level=log_level)
        for i in range(len(rows))
    ]
    _finish_progress(progress, results, len(rows))
    return 0 if all(r["ok"] for r in results) else 1


def _run_mpi(
    comm: Any,
    base_cfg: dict[str, Any],
    output_base: str,
    paths: list[str],
    rows: list[list[Any]],
    names: list[str],
    progress: Path,
    *,
    log_level: str,
) -> int:
    rank, size = comm.Get_rank(), comm.Get_size()
    try:
        # Rank 0 writes the header and warms the shared content-addressed core
        # once (the cases share it); the others wait so the core is read-only
        # before any case runs, avoiding a multi-rank build race.
        if rank == 0:
            _start_progress(progress, len(rows))
            _prewarm(load_run_input_from_dict(build_case(base_cfg, paths, rows[0]), output_run_base=output_base))
        comm.Barrier()

        results = [
            _run_case(i, names[i], base_cfg, output_base, paths, rows[i], progress, rank=rank, log_level=log_level)
            for i in range(rank, len(rows), size)
        ]
        gathered = comm.gather(results, root=0)
        if rank == 0:
            flat = [r for sub in gathered for r in sub]
            _finish_progress(progress, flat, len(rows))
            return 0 if all(r["ok"] for r in flat) else 1
        return 0
    except BaseException:  # free peers parked in Barrier/gather before teardown
        logger.exception("sweep: rank %d failed; aborting MPI job to release peers", comm.Get_rank())
        comm.Abort(1)
        raise


def _run_case(
    idx: int,
    name: str,
    base_cfg: dict[str, Any],
    output_base: str,
    paths: list[str],
    row: list[Any],
    progress: Path,
    *,
    rank: int | None,
    log_level: str,
) -> dict[str, Any]:
    """Build, load, and run one case in-process; never raise (failure -> record).

    Building + loading the case is inside this guard too, so a bad case is a
    FAIL record, not a crash that aborts the whole job.
    """
    from fexchange.pipeline.runner import run_pipeline_from_cfg

    t0 = time.time()
    try:
        cfg = load_run_input_from_dict(build_case(base_cfg, paths, row), output_run_base=output_base)
        rc = run_pipeline_from_cfg(cfg, log_level=log_level)
        rec: dict[str, Any] = {"name": name, "ok": rc == 0, "rank": rank, "secs": time.time() - t0}
        if rc != 0:
            rec["error"] = f"pipeline exit code {rc}"
    except Exception as exc:  # one bad case must not crash the rank
        logger.exception("sweep: case %d (run_name=%s) failed", idx, name)
        rec = {"name": name, "ok": False, "rank": rank, "secs": time.time() - t0, "error": str(exc)}
    _append_progress(progress, rec)
    return rec


# ---------------------------------------------------------------------------
# Warm the shared core once (rank 0)
# ---------------------------------------------------------------------------


def _prewarm(cfg: dict[str, Any]) -> None:
    """Build the shared content-addressed core {LMSM, LSJM, L0} for *cfg*.

    Assumes every case shares this core -- true when the swept columns do not
    change n_ele / Slater ratios / scheme (the usual case: sweeping U, Jh,
    hopping_file, ...). If a sweep changes those, drop the rank-0 prewarm and
    let each case build its own core.
    """
    from fexchange.pipeline.resolve import resolve_core_params
    from fexchange.pipeline.stages import (
        ensure_l0_fopt,
        ensure_l0_sopt,
        ensure_lsjm_all_three,
        ensure_lsms_all_three,
    )
    from fexchange.utils.constants import N_ORB

    n_ele, r42, r62, scheme = resolve_core_params(cfg)
    state: dict[str, Any] = {}
    ensure_lsms_all_three(cfg, state, n_ele=n_ele, n_orb=N_ORB, r42=r42, r62=r62, scheme=scheme)
    ensure_lsjm_all_three(cfg, state, n_ele=n_ele, n_orb=N_ORB, r42=r42, r62=r62, scheme=scheme)
    if str(cfg["runtime"]["branch"]) == "fopt":
        ensure_l0_fopt(cfg, state, n_ele=n_ele, n_orb=N_ORB)
    else:
        ensure_l0_sopt(cfg, state, n_ele=n_ele, n_orb=N_ORB)


# ---------------------------------------------------------------------------
# Live plain-text progress
# ---------------------------------------------------------------------------


def _start_progress(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# sweep: {n} cases (running)\n", encoding="utf-8")


def _append_progress(path: Path, rec: dict[str, Any]) -> None:
    """Append one short, single-line record per finished case.

    POSIX O_APPEND makes a short write atomic, so ranks can append concurrently
    on a shared filesystem (single node). Fields are flattened to one line and
    capped so that assumption holds.
    """
    rank = rec.get("rank")
    rtag = f"r{rank}" if rank is not None else "  -"
    line = f"{'ok  ' if rec['ok'] else 'FAIL'}  {rtag:>3}  {rec.get('secs', 0.0):7.2f}s  {_one_line(rec['name'], 120)}"
    if rec.get("error"):
        line += f"   -- {_one_line(rec['error'], 200)}"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _finish_progress(path: Path, results: list[dict[str, Any]], n: int) -> None:
    failed = sum(1 for r in results if not r["ok"])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"# done: {len(results)}/{n}, {failed} failed\n")
    logger.info("sweep: %d/%d done, %d failed -> %s", len(results), n, failed, path)


def _one_line(text: Any, cap: int) -> str:
    """Flatten whitespace (incl. newlines) to single spaces and cap length."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= cap else flat[: cap - 3] + "..."


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _output_root(base: dict[str, Any]) -> str:
    paths = base.get("paths")
    if not isinstance(paths, dict) or not isinstance(paths.get("output_root"), str):
        raise InputError("FXE-INPUT-003", "sweep base requires a string paths.output_root")
    return str(paths["output_root"]).strip()


def _output_run_base(base: dict[str, Any]) -> str:
    paths = base.get("paths")
    if not isinstance(paths, dict):
        raise InputError("FXE-INPUT-003", "sweep base requires a [paths] table")
    output_run = paths.get("output_run")
    if isinstance(output_run, str) and output_run.strip():
        return output_run.strip()
    output_root = paths.get("output_root")
    if isinstance(output_root, str) and output_root.strip():
        return output_root.strip()
    raise InputError("FXE-INPUT-003", "sweep base requires paths.output_root or paths.output_run")


# Env vars a process inherits ONLY when launched as a distinct MPI/scheduler rank.
# Presence (not value) signals "possibly parallel"; the authoritative world size
# then comes from MPI.COMM_WORLD.Get_size(), so this list never has to encode size
# semantics -- adding a launcher only widens the fail-fast net, it can never change
# how cases are sliced. Allocation-wide vars (e.g. SLURM_NTASKS, which sbatch sets
# for the whole job) are deliberately excluded so a plain serial process inside an
# allocation is not mistaken for a rank.
_LAUNCH_ENV = (
    "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "PMI_RANK", "PMI_SIZE",                          # MPICH / Intel MPI (PMI-1/2)
    "PMIX_RANK",                                     # PMIx (Open MPI 5 / PRRTE)
    "MV2_COMM_WORLD_RANK",                           # MVAPICH2
    "MPI_LOCALRANKID", "MPI_LOCALNRANKS",            # Intel MPI
    "SLURM_PROCID",                                  # Slurm srun (set per task)
    "FLUX_TASK_RANK",                                # Flux
    "PALS_RANKID",                                   # HPE PALS / Cray EX
    "ALPS_APP_PE",                                   # Cray ALPS aprun
)


def _under_launcher() -> bool:
    """True if this process looks launched as one rank of a parallel job."""
    return any(os.environ.get(var) for var in _LAUNCH_ENV)


def _detect_comm() -> Any:
    """Return ``MPI.COMM_WORLD`` iff this is one of >1 ranks, else ``None`` (serial).

    The world size is read from MPI itself -- authoritative for whatever launcher
    is in use -- rather than guessed from an env allowlist. The environment is
    consulted only to decide whether mpi4py is *required*: under a launcher but
    with mpi4py missing we fail fast, because otherwise every rank would silently
    take the serial path and race on the shared output / progress file /
    content-addressed core. A serial run (no launcher) never imports mpi4py, so it
    needs no MPI install and triggers no MPI_Init on a login node.
    """
    if not _under_launcher():
        return None
    try:
        from mpi4py import MPI  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError_(
            "FXE-RUNTIME-001",
            "launched under an MPI launcher but mpi4py is not installed; "
            "install fexchange[mpi], or launch without srun/mpirun for the serial path",
            module="sweep",
            stage="run_sweep",
            op="mpi_detect",
        )
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() > 1 else None


def _warn_oversubscription() -> None:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        logger.warning(
            "OMP_NUM_THREADS=%s (expected '1'): ranks x BLAS-threads may oversubscribe cores. "
            "Export OMP_NUM_THREADS=1 (and OPENBLAS/MKL equivalents) before mpirun.",
            os.environ.get("OMP_NUM_THREADS"),
        )
