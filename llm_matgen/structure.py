"""MatGen 本地结构检查与单原子修订；迁自已测试的旧 builder 机制。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
    from ase.io import read, write
    from ase.neighborlist import neighbor_list

    ASE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # Keep --help usable when ASE is unavailable.
    np = None  # type: ignore[assignment]
    read = None  # type: ignore[assignment]
    write = None  # type: ignore[assignment]
    neighbor_list = None  # type: ignore[assignment]
    ASE_IMPORT_ERROR = exc


def require_ase() -> None:
    if ASE_IMPORT_ERROR is not None:
        raise RuntimeError(f"需要 ASE 才能处理结构；原始导入错误：{ASE_IMPORT_ERROR}")


def file_sha256(path: Path) -> str:
    return bytes_sha256(path.read_bytes())


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_structure(path_value: str) -> Any:
    require_ase()
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"输入结构不存在：{path}")
    atoms = read(str(path), index=-1)
    if atoms is None or len(atoms) == 0:
        raise ValueError(f"没有从文件中读到原子：{path}")
    return atoms


def is_vasp_path(path: Path) -> bool:
    name = path.name.upper()
    return (
        name in {"POSCAR", "CONTCAR"}
        or name.startswith("POSCAR.")
        or name.startswith("CONTCAR.")
        or path.suffix.lower() in {".vasp", ".poscar"}
    )


def output_format(path: Path) -> str:
    if is_vasp_path(path):
        return "vasp"
    suffix = path.suffix.lower().lstrip(".")
    if not suffix:
        raise ValueError(f"无法从输出文件名推断 ASE 格式：{path}")
    return suffix


def load_structure_snapshot(path: Path, snapshot: bytes, temp_root: Path) -> Any:
    """Parse exactly the bytes whose hash is recorded in the sidecar."""
    require_ase()
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".matgen-structure-input-",
        dir=temp_root,
    ) as directory:
        snapshot_path = Path(directory) / path.name
        snapshot_path.write_bytes(snapshot)
        atoms = read(str(snapshot_path), index=-1)
    if atoms is None or len(atoms) == 0:
        raise ValueError(f"没有从输入快照读到原子：{path}")
    return atoms


def write_structure(path: Path, atoms: Any, format_name: str) -> None:
    options: dict[str, Any] = {"format": format_name}
    if format_name == "vasp":
        options.update(direct=True, vasp5=True, sort=False)
    write(str(path), atoms, **options)


def geometry_analysis(atoms: Any) -> tuple[float | None, tuple[int, int] | None]:
    """Return the shortest MIC pair through ASE neighbor lists, never a distance matrix."""
    if len(atoms) < 2:
        return None, None
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        return None, None
    pbc = np.asarray(atoms.pbc, dtype=bool)
    periodic_upper_bound = 0.5 * sum(
        float(np.linalg.norm(atoms.cell.array[axis])) for axis in range(3) if pbc[axis]
    )
    nonperiodic_extent = float(np.linalg.norm(np.ptp(atoms.positions, axis=0)))
    upper_bound = float(np.nextafter(max(1.0, periodic_upper_bound + nonperiodic_extent), np.inf))
    cutoff = min(1.0, upper_bound)
    while True:
        indices_i, indices_j, distances = neighbor_list("ijd", atoms, cutoff)
        distinct = indices_i != indices_j
        if np.any(distinct):
            valid_i = indices_i[distinct]
            valid_j = indices_j[distinct]
            valid_distances = distances[distinct]
            selected = int(np.argmin(valid_distances))
            return (
                float(valid_distances[selected]),
                (int(valid_i[selected]) + 1, int(valid_j[selected]) + 1),
            )
        if cutoff >= upper_bound:
            return None, None
        cutoff = min(upper_bound, cutoff * 2.0)


def structure_summary(
    atoms: Any,
    geometry: tuple[float | None, tuple[int, int] | None] | None = None,
) -> dict[str, Any]:
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    minimum, pair = geometry if geometry is not None else geometry_analysis(atoms)
    return {
        "formula": atoms.get_chemical_formula(mode="hill"),
        "atom_count": int(len(atoms)),
        "cell_lengths_angstrom": [float(x) for x in lengths],
        "cell_angles_degree": [float(x) for x in angles],
        "cell_volume_angstrom3": float(atoms.cell.volume),
        "pbc": [bool(x) for x in atoms.pbc],
        "minimum_pair_distance_angstrom": minimum,
        "minimum_pair_indices_1based": list(pair) if pair else None,
    }


def validation_issues(
    atoms: Any,
    min_distance: float,
    geometry: tuple[float | None, tuple[int, int] | None] | None = None,
) -> list[str]:
    issues: list[str] = []
    if len(atoms) == 0:
        issues.append("结构中没有原子。")
    if not np.isfinite(atoms.get_positions()).all():
        issues.append("原子坐标含 NaN 或无穷值。")
    lengths = atoms.cell.lengths()
    if any(float(value) <= 1.0e-8 for value in lengths):
        issues.append("至少一个晶格矢量长度接近零，不适合作为 VASP 三维周期结构。")
    if not math.isfinite(float(atoms.cell.volume)) or float(atoms.cell.volume) <= 1.0e-8:
        issues.append("晶胞体积无效或接近零。")
    minimum, pair = geometry if geometry is not None else geometry_analysis(atoms)
    if minimum is not None and minimum < min_distance:
        issues.append(
            f"原子 {pair[0]} 与 {pair[1]} 的最短距离为 {minimum:.4f} Å，"
            f"低于报警阈值 {min_distance:.4f} Å。"
        )
    return issues


def validate_finite_structure(atoms: Any) -> None:
    if not np.isfinite(atoms.positions).all():
        raise ValueError("输入结构的原子坐标含 NaN 或无穷值。")
    if not np.isfinite(atoms.cell.array).all():
        raise ValueError("输入结构的晶胞含 NaN 或无穷值。")


def validate_finite_values(values: Any, option_name: str) -> Any:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{option_name} 不能包含 NaN 或无穷值。")
    return array


def write_report(args: argparse.Namespace, atoms: Any) -> list[str]:
    if not math.isfinite(args.min_distance) or args.min_distance < 0:
        raise ValueError("--min-distance 必须是有限的非负数。")
    geometry = geometry_analysis(atoms)
    issues = validation_issues(atoms, args.min_distance, geometry)
    payload = {
        "command": args.structure_command,
        "structure": structure_summary(atoms, geometry),
        "validation_issues": issues,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return issues


def validate_1based_index(index: int, atom_count: int) -> int:
    if index < 1 or index > atom_count:
        raise IndexError(f"原子索引超出 1..{atom_count} 范围：{index}")
    return index


def temporary_path(directory: Path, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, value = tempfile.mkstemp(prefix=".matgen-structure-", suffix=suffix, dir=directory)
    os.close(descriptor)
    return Path(value)


def remove_if_present(path: Path | None) -> OSError | None:
    if path is not None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            return exc
    return None


def publish_new_file(
    temporary: Path, target: Path, on_linked: Callable[[], None]
) -> None:
    """Publish a checked temporary file atomically without overwriting a target."""
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise FileExistsError(f"目标已存在，拒绝覆盖：{target}") from None
    on_linked()
    os.unlink(temporary)


def make_sidecar(
    input_path: Path,
    output_path: Path,
    parent_sha256: str,
    index: int,
    element: str,
    old_cartesian: Any,
    new_cartesian: Any,
    old_fractional: Any,
    new_fractional: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": "vasp-structure-revision",
        "version": 1,
        "parent_sha256": parent_sha256,
        "output_sha256": file_sha256(output_path),
        "index_1based": index,
        "element": element,
        "old_cartesian_angstrom": [float(value) for value in old_cartesian],
        "new_cartesian_angstrom": [float(value) for value in new_cartesian],
        "old_fractional": [float(value) for value in old_fractional],
        "new_fractional": [float(value) for value in new_fractional],
        "cartesian_delta_angstrom": [float(value) for value in new_cartesian - old_cartesian],
        "reason": reason,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path_absolute": str(input_path.resolve()),
        "input_filename": input_path.name,
        "output_filename": output_path.name,
    }


def validate_temporary_revision(
    before: Any, expected: Any, after: Any, changed_index: int
) -> None:
    if len(after) != len(before):
        raise ValueError("临时输出校验失败：原子数发生变化。")
    if after.get_chemical_symbols() != before.get_chemical_symbols():
        raise ValueError("临时输出校验失败：元素顺序发生变化。")
    if not np.allclose(after.cell.array, before.cell.array) or tuple(after.pbc) != tuple(before.pbc):
        raise ValueError("临时输出校验失败：晶胞或周期边界发生变化。")
    if repr(after.constraints) != repr(before.constraints):
        raise ValueError("临时输出校验失败：原有约束发生变化。")
    unchanged = [index for index in range(len(before)) if index != changed_index]
    if unchanged and not np.allclose(after.positions[unchanged], before.positions[unchanged]):
        raise ValueError("临时输出校验失败：未声明原子的坐标发生变化。")
    if not np.allclose(after.positions[changed_index], expected.positions[changed_index]):
        raise ValueError("临时输出校验失败：声明原子的坐标未按要求序列化。")


def handle_adjust_coordinate(args: argparse.Namespace) -> int:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    sidecar_path = Path(args.sidecar).resolve() if args.sidecar else Path(f"{output_path}.revision.json")
    if len({input_path, output_path, sidecar_path}) != 3:
        raise ValueError("输入、输出与 sidecar 必须是三个不同路径。")
    if input_path == output_path:
        raise ValueError("输入与输出解析后的路径不得相同。")
    if output_path.exists() or sidecar_path.exists():
        existing = output_path if output_path.exists() else sidecar_path
        raise FileExistsError(f"目标已存在，拒绝覆盖：{existing}")
    reason = args.reason.strip()
    if not reason:
        raise ValueError("--reason 去除空白后不得为空。")

    coordinate_option, coordinate_values = next(
        (name, value)
        for name, value in (
            ("--set-cartesian", args.set_cartesian),
            ("--set-fractional", args.set_fractional),
            ("--delta-cartesian", args.delta_cartesian),
        )
        if value is not None
    )
    coordinate_values = validate_finite_values(coordinate_values, coordinate_option)
    input_snapshot = input_path.read_bytes()
    parent_sha256 = bytes_sha256(input_snapshot)
    atoms = load_structure_snapshot(input_path, input_snapshot, output_path.parent)
    validate_finite_structure(atoms)
    index = validate_1based_index(args.index, len(atoms))
    result = atoms.copy()
    atom_index = index - 1
    old_cartesian = result.positions[atom_index].copy()
    old_fractional = result.get_scaled_positions(wrap=False)[atom_index].copy()
    if args.set_cartesian is not None:
        new_cartesian = coordinate_values
        result.positions[atom_index] = new_cartesian
    elif args.set_fractional is not None:
        new_cartesian = coordinate_values @ result.cell.array
        result.positions[atom_index] = new_cartesian
    else:
        new_cartesian = old_cartesian + coordinate_values
        result.positions[atom_index] = new_cartesian
    new_fractional = result.get_scaled_positions(wrap=False)[atom_index].copy()
    validate_finite_structure(result)
    output_format_name = output_format(output_path)

    structure_temp: Path | None = None
    sidecar_temp: Path | None = None
    output_published = False
    sidecar_published = False
    try:
        structure_temp = temporary_path(output_path.parent, output_path.suffix)
        write_structure(structure_temp, result, output_format_name)
        verified = read(str(structure_temp), index=-1, format=output_format_name)
        validate_temporary_revision(atoms, result, verified, atom_index)
        sidecar = make_sidecar(
            input_path, structure_temp, parent_sha256, index, result[atom_index].symbol, old_cartesian,
            new_cartesian, old_fractional, new_fractional, reason,
        )
        sidecar["output_sha256"] = file_sha256(structure_temp)
        sidecar["output_filename"] = output_path.name
        sidecar_temp = temporary_path(sidecar_path.parent, ".json")
        sidecar_temp.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        json.loads(sidecar_temp.read_text(encoding="utf-8"))
        if file_sha256(input_path) != parent_sha256:
            raise RuntimeError("输入结构在修订期间发生变化，拒绝发布。")
        if output_path.exists() or sidecar_path.exists():
            raise FileExistsError("目标在发布前出现，拒绝覆盖。")

        def mark_output_published() -> None:
            nonlocal output_published
            output_published = True

        def mark_sidecar_published() -> None:
            nonlocal sidecar_published
            sidecar_published = True

        publish_new_file(structure_temp, output_path, mark_output_published)
        structure_temp = None
        publish_new_file(sidecar_temp, sidecar_path, mark_sidecar_published)
        sidecar_temp = None
    except Exception as exc:
        cleanup_errors = [
            error for error in (
                remove_if_present(structure_temp),
                remove_if_present(sidecar_temp),
                remove_if_present(sidecar_path) if sidecar_published else None,
                remove_if_present(output_path) if output_published else None,
            ) if error is not None
        ]
        if cleanup_errors:
            raise RuntimeError(
                "发布失败且无法完整清理临时结果：" + "; ".join(str(error) for error in cleanup_errors)
            ) from exc
        raise
    return 0


def _pymatgen_pair_minima(structure: Any) -> dict[str, float]:
    symbols = np.asarray([site.specie.symbol for site in structure])
    distances = structure.distance_matrix.copy()
    translations = np.asarray([
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    ], dtype=float)
    shortest_periodic = float(
        np.min(np.linalg.norm(translations @ structure.lattice.matrix, axis=1))
    )
    np.fill_diagonal(distances, shortest_periodic)
    result: dict[str, float] = {}
    elements = sorted(set(symbols))
    for index, left in enumerate(elements):
        for right in elements[index:]:
            result[f"{left}-{right}"] = float(
                distances[np.ix_(symbols == left, symbols == right)].min()
            )
    return result


def _nonaffine_assignment(initial: Any, final: Any) -> tuple[float, float]:
    from scipy.optimize import linear_sum_assignment

    distances: list[float] = []
    elements = sorted({site.specie.symbol for site in initial})
    for element in elements:
        initial_fractional = np.asarray([
            site.frac_coords for site in initial if site.specie.symbol == element
        ])
        final_fractional = np.asarray([
            site.frac_coords for site in final if site.specie.symbol == element
        ])
        matrix = initial.lattice.get_all_distances(initial_fractional, final_fractional)
        rows, columns = linear_sum_assignment(matrix)
        distances.extend(float(value) for value in matrix[rows, columns])
    values = np.asarray(distances, dtype=float)
    return float(np.sqrt(np.mean(values ** 2))), float(values.max(initial=0.0))


def _space_group_summary(structure: Any, symprec: float) -> dict[str, Any]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    return {
        "symbol": analyzer.get_space_group_symbol(),
        "number": int(analyzer.get_space_group_number()),
        "crystal_system": analyzer.get_crystal_system(),
    }


def audit_relaxation(
    initial_path: str | Path,
    final_path: str | Path,
    *,
    label: str,
    reason: str,
    symprec: float = 0.001,
    large_cell_percent: float = 10.0,
    runaway_cell_percent: float = 25.0,
    runaway_volume_percent: float = 100.0,
    nonaffine_displacement_A: float = 0.5,
    network_change_fraction: float = 0.25,
) -> dict[str, Any]:
    """Compare an initial/final structure pair without inferring convergence."""
    from pymatgen.core import Structure

    if label not in {"passed", "failed"}:
        raise ValueError("label must be passed or failed")
    reason = reason.strip()
    if not reason:
        raise ValueError("reason must not be blank")
    initial_path = Path(initial_path).resolve()
    final_path = Path(final_path).resolve()
    initial = Structure.from_file(initial_path)
    final = Structure.from_file(final_path)
    if len(initial) != len(final):
        raise ValueError("initial/final atom counts differ")
    if initial.composition != final.composition:
        raise ValueError("initial/final composition differs")
    if not math.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be positive and finite")
    initial_lengths = np.asarray(initial.lattice.abc, dtype=float)
    final_lengths = np.asarray(final.lattice.abc, dtype=float)
    lattice_change = (final_lengths / initial_lengths - 1.0) * 100.0
    angle_change = np.asarray(final.lattice.angles) - np.asarray(initial.lattice.angles)
    volume_change = (float(final.volume) / float(initial.volume) - 1.0) * 100.0
    rms_displacement, max_displacement = _nonaffine_assignment(initial, final)
    initial_group = _space_group_summary(initial, symprec)
    final_group = _space_group_summary(final, symprec)
    initial_minima = _pymatgen_pair_minima(initial)
    final_minima = _pymatgen_pair_minima(final)
    ratios = {
        key: final_minima[key] / value
        for key, value in initial_minima.items()
        if value > 0 and key in final_minima
    }
    anomalies: list[str] = []
    if float(np.max(np.abs(lattice_change))) > large_cell_percent:
        anomalies.append("large_cell_change")
    if (
        float(np.max(np.abs(lattice_change))) > runaway_cell_percent
        or abs(volume_change) > runaway_volume_percent
    ):
        anomalies.append("runaway_cell")
    if initial_group["number"] != final_group["number"]:
        anomalies.append("symmetry_changed")
    if max_displacement > nonaffine_displacement_A:
        anomalies.append("large_nonaffine_displacement")
    if any(value < 1.0 - network_change_fraction for value in ratios.values()):
        anomalies.append("contact_collapse")
    if ratios and float(np.median(list(ratios.values()))) > 1.0 + network_change_fraction:
        anomalies.append("network_dilution")
    return {
        "schema": "llm-matgen-relaxation-audit",
        "version": 1,
        "label": label,
        "reason": reason,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "initial": {
            "path": str(initial_path),
            "sha256": file_sha256(initial_path),
            "formula": initial.composition.reduced_formula,
            "atom_count": len(initial),
            "cell_lengths_A": [float(value) for value in initial_lengths],
            "cell_angles_degree": [float(value) for value in initial.lattice.angles],
            "volume_A3": float(initial.volume),
            "density_g_cm3": float(initial.density),
            "space_group": initial_group,
            "pair_min_A": initial_minima,
        },
        "final": {
            "path": str(final_path),
            "sha256": file_sha256(final_path),
            "formula": final.composition.reduced_formula,
            "atom_count": len(final),
            "cell_lengths_A": [float(value) for value in final_lengths],
            "cell_angles_degree": [float(value) for value in final.lattice.angles],
            "volume_A3": float(final.volume),
            "density_g_cm3": float(final.density),
            "space_group": final_group,
            "pair_min_A": final_minima,
        },
        "metrics": {
            "lattice_change_percent": [float(value) for value in lattice_change],
            "angle_change_degree": [float(value) for value in angle_change],
            "volume_change_percent": float(volume_change),
            "density_change_percent": float((float(final.density) / float(initial.density) - 1.0) * 100.0),
            "nonaffine_rms_displacement_A": rms_displacement,
            "nonaffine_max_displacement_A": max_displacement,
            "pair_distance_ratios": ratios,
        },
        "thresholds": {
            "symprec": symprec,
            "large_cell_percent": large_cell_percent,
            "runaway_cell_percent": runaway_cell_percent,
            "runaway_volume_percent": runaway_volume_percent,
            "nonaffine_displacement_A": nonaffine_displacement_A,
            "network_change_fraction": network_change_fraction,
        },
        "anomalies": anomalies,
        "interpretation": "The user supplied label is preserved; anomaly codes do not establish DFT convergence or stability.",
    }


def handle_relaxation_audit(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"target already exists; refusing to overwrite: {output}")
    report = audit_relaxation(
        args.initial,
        args.final,
        label=args.label,
        reason=args.reason,
        symprec=args.symprec,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"ok": True, "output": str(output), "anomalies": report["anomalies"]}, ensure_ascii=False))
    return 0


def add_read_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="输入结构文件")
    parser.add_argument("--min-distance", type=float, default=0.7, help="重叠报警阈值，单位 Å")


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    subparsers = parser.add_subparsers(dest="structure_command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="读取并汇总结构")
    add_read_args(inspect_parser)
    inspect_parser.set_defaults(handler=lambda args: (write_report(args, load_structure(args.input)), 0)[1])
    validate_parser = subparsers.add_parser("validate", help="检查结构和原子重叠")
    add_read_args(validate_parser)
    validate_parser.set_defaults(handler=lambda args: 2 if write_report(args, load_structure(args.input)) else 0)
    adjust_parser = subparsers.add_parser("adjust-coordinate", help="修订一个原子的坐标并写入审计 sidecar")
    adjust_parser.add_argument("--input", required=True, help="输入结构文件")
    adjust_parser.add_argument("--index", required=True, type=int, help="要修订的 1-based 原子索引")
    adjust_parser.add_argument("--reason", required=True, help="本次人工修订的原因")
    adjust_parser.add_argument("--output", required=True, help="新结构输出文件")
    adjust_parser.add_argument("--sidecar", help="审计 sidecar；默认是 <output>.revision.json")
    modes = adjust_parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--set-cartesian", nargs=3, type=float, metavar=("X", "Y", "Z"))
    modes.add_argument("--set-fractional", nargs=3, type=float, metavar=("U", "V", "W"))
    modes.add_argument("--delta-cartesian", nargs=3, type=float, metavar=("DX", "DY", "DZ"))
    adjust_parser.set_defaults(handler=handle_adjust_coordinate)
    audit_parser = subparsers.add_parser("relaxation-audit", help="审计优化前后结构变化，不推断DFT收敛")
    audit_parser.add_argument("--initial", required=True, help="优化前POSCAR")
    audit_parser.add_argument("--final", required=True, help="优化后CONTCAR")
    audit_parser.add_argument("--label", required=True, choices=("passed", "failed"), help="用户提供的结果标签")
    audit_parser.add_argument("--reason", required=True, help="人工判断原因")
    audit_parser.add_argument("--output", required=True, help="新的JSON审计文件")
    audit_parser.add_argument("--symprec", type=float, default=0.001)
    audit_parser.set_defaults(handler=handle_relaxation_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = configure_parser(argparse.ArgumentParser(description="检查或人工修订单个原子坐标。"))
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
