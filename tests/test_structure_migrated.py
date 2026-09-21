from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ase import Atoms
from ase.constraints import FixAtoms


import argparse
from llm_matgen import structure as MODULE

SCRIPT = Path(MODULE.__file__).resolve()


class ParserTests(unittest.TestCase):
    def test_four_public_commands_are_registered(self) -> None:
        parser = MODULE.configure_parser(argparse.ArgumentParser())
        subparsers = next(
            action for action in parser._actions
            if action.__class__.__name__ == "_SubParsersAction"
        )
        self.assertEqual(
            set(subparsers.choices),
            {"inspect", "validate", "adjust-coordinate", "relaxation-audit"},
        )
        for legacy in (
            "convert", "supercell", "slab", "vacuum", "vacancy", "substitute",
            "interstitial", "adsorbate", "stack",
        ):
            with self.assertRaises(SystemExit):
                parser.parse_args([legacy])

    def test_adjust_coordinate_requires_exactly_one_mode_and_has_no_force(self) -> None:
        parser = MODULE.configure_parser(argparse.ArgumentParser())
        common = [
            "adjust-coordinate", "--input", "in.traj", "--index", "1",
            "--reason", "manual correction", "--output", "out.traj",
        ]
        with self.assertRaises(SystemExit):
            parser.parse_args(common)
        with self.assertRaises(SystemExit):
            parser.parse_args(common + ["--set-cartesian", "1", "2", "3", "--set-fractional", "0", "0", "0"])
        with self.assertRaises(SystemExit):
            parser.parse_args(common + ["--set-cartesian", "1", "2", "3", "--force"])


@unittest.skipIf(MODULE.ASE_IMPORT_ERROR is not None, "ASE is not installed")
class CoordinateRevisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.input_path = self.directory / "input.traj"
        self.output_path = self.directory / "revised.traj"
        atoms = Atoms(
            "Si2",
            scaled_positions=[(0.10, 0.20, 0.30), (0.40, 0.50, 0.60)],
            cell=[5.43, 5.43, 5.43],
            pbc=(True, True, False),
        )
        atoms.set_constraint(FixAtoms(indices=[0]))
        MODULE.write(str(self.input_path), atoms)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def revise(self, *mode: str, sidecar: Path | None = None, index: int = 2, reason: str = "correct source coordinate") -> int:
        arguments = [
            "adjust-coordinate", "--input", str(self.input_path), "--index", str(index),
            "--reason", reason, "--output", str(self.output_path), *mode,
        ]
        if sidecar is not None:
            arguments.extend(["--sidecar", str(sidecar)])
        return MODULE.main(arguments)

    def test_set_cartesian_preserves_structure_and_writes_complete_sidecar(self) -> None:
        before = MODULE.load_structure(str(self.input_path))
        self.assertEqual(self.revise("--set-cartesian", "2.0", "2.1", "2.2"), 0)
        after = MODULE.load_structure(str(self.output_path))
        self.assertEqual(len(after), len(before))
        self.assertEqual(after.get_chemical_symbols(), before.get_chemical_symbols())
        self.assertTrue(MODULE.np.allclose(after.cell.array, before.cell.array))
        self.assertEqual(tuple(after.pbc), tuple(before.pbc))
        self.assertEqual(after.constraints[0].get_indices().tolist(), [0])
        self.assertTrue(MODULE.np.allclose(after.positions[0], before.positions[0]))
        self.assertTrue(MODULE.np.allclose(after.positions[1], [2.0, 2.1, 2.2]))

        sidecar = Path(f"{self.output_path}.revision.json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "vasp-structure-revision")
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["parent_sha256"], self.sha256(self.input_path))
        self.assertEqual(payload["output_sha256"], self.sha256(self.output_path))
        self.assertEqual(payload["index_1based"], 2)
        self.assertEqual(payload["element"], "Si")
        self.assertEqual(payload["reason"], "correct source coordinate")
        self.assertEqual(payload["input_path_absolute"], str(self.input_path.resolve()))
        self.assertEqual(payload["input_filename"], self.input_path.name)
        self.assertEqual(payload["output_filename"], self.output_path.name)
        self.assertIn("created_utc", payload)
        self.assertTrue(MODULE.np.allclose(payload["old_cartesian_angstrom"], before.positions[1]))
        self.assertTrue(MODULE.np.allclose(payload["new_cartesian_angstrom"], after.positions[1]))
        self.assertTrue(MODULE.np.allclose(payload["old_fractional"], before.get_scaled_positions()[1]))
        self.assertTrue(MODULE.np.allclose(payload["new_fractional"], after.get_scaled_positions()[1]))
        self.assertTrue(MODULE.np.allclose(payload["cartesian_delta_angstrom"], after.positions[1] - before.positions[1]))

    def test_set_fractional_changes_only_selected_atom(self) -> None:
        before = MODULE.load_structure(str(self.input_path))
        self.assertEqual(self.revise("--set-fractional", "1.25", "-0.35", "0.45"), 0)
        after = MODULE.load_structure(str(self.output_path))
        self.assertTrue(MODULE.np.allclose(after.get_scaled_positions(wrap=False)[1], [1.25, -0.35, 0.45]))
        self.assertTrue(MODULE.np.allclose(after.positions[0], before.positions[0]))
        payload = json.loads(Path(f"{self.output_path}.revision.json").read_text(encoding="utf-8"))
        self.assertTrue(MODULE.np.allclose(payload["new_fractional"], [1.25, -0.35, 0.45]))

    def test_poscar_revised_output_uses_target_format_for_temporary_io(self) -> None:
        input_path = self.directory / "POSCAR"
        output_path = self.directory / "POSCAR.revised"
        atoms = Atoms("Si2", positions=[(0, 0, 0), (1, 1, 1)], cell=[5, 5, 5], pbc=True)
        MODULE.write(str(input_path), atoms, format="vasp", direct=True, vasp5=True, sort=False)
        self.assertEqual(MODULE.main([
            "adjust-coordinate", "--input", str(input_path), "--index", "2",
            "--reason", "verify POSCAR derived output", "--output", str(output_path),
            "--delta-cartesian", "0", "0", "0.2",
        ]), 0)
        revised = MODULE.read(str(output_path), format="vasp")
        self.assertTrue(MODULE.np.allclose(revised.positions[1], [1, 1, 1.2]))

    def test_rejects_nonfinite_coordinate_values_in_all_modes(self) -> None:
        for mode in (
            ("--set-cartesian", "nan", "0", "0"),
            ("--set-fractional", "inf", "0", "0"),
            ("--delta-cartesian", "0", "inf", "0"),
        ):
            self.assertEqual(self.revise(*mode), 1)
            self.assertFalse(self.output_path.exists())

    def test_neighbor_search_grows_from_a_bounded_cutoff_for_sparse_large_cell(self) -> None:
        atoms = Atoms("He2", positions=[(0, 0, 0), (2.5, 0, 0)], cell=[10000, 10000, 10000], pbc=False)
        original_neighbor_list = MODULE.neighbor_list
        cutoffs: list[float] = []

        def record_neighbor_list(*arguments: object) -> object:
            cutoffs.append(float(arguments[2]))
            return original_neighbor_list(*arguments)

        with mock.patch.object(MODULE, "neighbor_list", side_effect=record_neighbor_list):
            minimum, pair = MODULE.geometry_analysis(atoms)
        self.assertEqual(pair, (1, 2))
        self.assertAlmostEqual(minimum, 2.5)
        self.assertLessEqual(max(cutoffs), 8.0)

    def test_link_success_then_temporary_unlink_failure_leaves_no_result(self) -> None:
        original_unlink = MODULE.os.unlink
        calls = 0

        def fail_only_first_unlink(path: str) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("simulated temporary unlink failure")
            original_unlink(path)

        with mock.patch.object(MODULE.os, "unlink", side_effect=fail_only_first_unlink):
            self.assertEqual(self.revise("--delta-cartesian", "0", "0", "1"), 1)
        self.assertFalse(self.output_path.exists())
        self.assertFalse(Path(f"{self.output_path}.revision.json").exists())

    def test_rejects_input_changed_after_initial_hash_before_publish(self) -> None:
        original_write_structure = MODULE.write_structure

        def write_then_mutate_input(path: Path, atoms: object, *format_args: str) -> None:
            original_write_structure(path, atoms, *format_args)
            self.input_path.write_bytes(b"changed after initial hash")

        with mock.patch.object(MODULE, "write_structure", side_effect=write_then_mutate_input):
            self.assertEqual(self.revise("--delta-cartesian", "0", "0", "1"), 1)
        self.assertFalse(self.output_path.exists())
        self.assertFalse(Path(f"{self.output_path}.revision.json").exists())

    def test_snapshot_binds_parent_hash_and_atoms_during_a_to_b_to_a_swap(self) -> None:
        original_atoms = MODULE.load_structure(str(self.input_path))
        original_bytes = self.input_path.read_bytes()
        replacement = original_atoms.copy()
        replacement.positions[1] = [3.0, 3.0, 3.0]
        original_load_structure_snapshot = MODULE.load_structure_snapshot

        def parse_snapshot_while_live_path_is_replacement(
            path: Path,
            snapshot: bytes,
            temp_root: Path,
        ) -> object:
            self.assertEqual(path, self.input_path.resolve())
            self.assertEqual(snapshot, original_bytes)
            self.assertEqual(temp_root, self.output_path.resolve().parent)
            MODULE.write(str(self.input_path), replacement)
            try:
                return original_load_structure_snapshot(path, snapshot, temp_root)
            finally:
                self.input_path.write_bytes(original_bytes)

        with mock.patch.object(
            MODULE,
            "load_structure_snapshot",
            side_effect=parse_snapshot_while_live_path_is_replacement,
        ):
            self.assertEqual(self.revise("--delta-cartesian", "0", "0", "0.2"), 0)
        revised = MODULE.load_structure(str(self.output_path))
        sidecar = json.loads(Path(f"{self.output_path}.revision.json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["parent_sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertTrue(MODULE.np.allclose(sidecar["old_cartesian_angstrom"], original_atoms.positions[1]))
        self.assertTrue(MODULE.np.allclose(revised.positions[1], original_atoms.positions[1] + [0, 0, 0.2]))

    def test_snapshot_temporary_copy_uses_resolved_output_parent(self) -> None:
        original_temporary_directory = tempfile.TemporaryDirectory
        observed_directories: list[object] = []

        def record_temporary_directory(*args: object, **kwargs: object) -> object:
            observed_directories.append(kwargs.get("dir"))
            return original_temporary_directory(*args, **kwargs)

        with tempfile.TemporaryDirectory(dir=SCRIPT.parents[1]) as output_directory:
            self.output_path = Path(output_directory) / "revised.traj"
            with mock.patch.object(
                MODULE.tempfile,
                "TemporaryDirectory",
                side_effect=record_temporary_directory,
            ):
                self.assertEqual(self.revise("--delta-cartesian", "0", "0", "0.2"), 0)

            self.assertEqual(len(observed_directories), 1)
            self.assertIsNotNone(observed_directories[0])
            observed_root = Path(observed_directories[0]).resolve()
            self.assertEqual(observed_root, self.output_path.resolve().parent)
            self.assertFalse(observed_root.is_relative_to(Path(tempfile.gettempdir()).resolve()))

    def test_snapshot_loader_preserves_xdatcar_input_support(self) -> None:
        self.input_path = self.directory / "XDATCAR"
        atoms = Atoms(
            "Si2",
            positions=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        )
        MODULE.write(str(self.input_path), atoms, format="vasp-xdatcar")
        parent_hash = self.sha256(self.input_path)

        self.assertEqual(self.revise("--delta-cartesian", "0", "0", "0.2"), 0)

        revised = MODULE.load_structure(str(self.output_path))
        sidecar = json.loads(Path(f"{self.output_path}.revision.json").read_text(encoding="utf-8"))
        self.assertTrue(MODULE.np.allclose(revised.positions[1], [1.0, 1.0, 1.2]))
        self.assertEqual(sidecar["parent_sha256"], parent_hash)
        self.assertTrue(MODULE.np.allclose(sidecar["old_cartesian_angstrom"], [1.0, 1.0, 1.0]))

    def test_snapshot_loader_preserves_compressed_xyz_input_support(self) -> None:
        self.input_path = self.directory / "input.xyz.gz"
        atoms = Atoms("Si2", positions=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
        MODULE.write(str(self.input_path), atoms, format="xyz")
        parent_hash = self.sha256(self.input_path)

        self.assertEqual(self.revise("--delta-cartesian", "0", "0", "0.2"), 0)

        revised = MODULE.load_structure(str(self.output_path))
        sidecar = json.loads(Path(f"{self.output_path}.revision.json").read_text(encoding="utf-8"))
        self.assertTrue(MODULE.np.allclose(revised.positions[1], [1.0, 1.0, 1.2]))
        self.assertEqual(sidecar["parent_sha256"], parent_hash)
        self.assertTrue(MODULE.np.allclose(sidecar["old_cartesian_angstrom"], [1.0, 1.0, 1.0]))

    def test_explicit_sidecar_in_other_directory_is_published_from_its_own_directory(self) -> None:
        sidecar = self.directory / "audit" / "revision.json"
        original_link = MODULE.os.link
        parents: list[tuple[Path, Path]] = []

        def record_link(source: str, target: str) -> None:
            parents.append((Path(source).parent, Path(target).parent))
            original_link(source, target)

        with mock.patch.object(MODULE.os, "link", side_effect=record_link):
            self.assertEqual(self.revise("--delta-cartesian", "0", "0", "1", sidecar=sidecar), 0)
        self.assertTrue(self.output_path.exists())
        self.assertTrue(sidecar.exists())
        self.assertEqual(len(parents), 2)
        for source_parent, target_parent in parents:
            self.assertTrue(source_parent.samefile(target_parent))

    def test_delta_cartesian_changes_only_selected_atom(self) -> None:
        before = MODULE.load_structure(str(self.input_path))
        self.assertEqual(self.revise("--delta-cartesian", "0.1", "-0.2", "0.3"), 0)
        after = MODULE.load_structure(str(self.output_path))
        self.assertTrue(MODULE.np.allclose(after.positions[1], before.positions[1] + [0.1, -0.2, 0.3]))
        self.assertTrue(MODULE.np.allclose(after.positions[0], before.positions[0]))

    def test_rejects_invalid_index_blank_reason_and_same_input_output(self) -> None:
        self.assertEqual(self.revise("--delta-cartesian", "0", "0", "1", index=3), 1)
        self.assertEqual(self.revise("--delta-cartesian", "0", "0", "1", reason="   "), 1)
        arguments = [
            "adjust-coordinate", "--input", str(self.input_path), "--index", "2",
            "--reason", "valid", "--output", str(self.input_path),
            "--delta-cartesian", "0", "0", "1",
        ]
        self.assertEqual(MODULE.main(arguments), 1)
        self.assertFalse(self.output_path.exists())

    def test_refuses_existing_output_or_sidecar_without_overwrite(self) -> None:
        self.output_path.write_text("existing output", encoding="utf-8")
        self.assertEqual(self.revise("--delta-cartesian", "0", "0", "1"), 1)
        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "existing output")
        self.output_path.unlink()
        sidecar = Path(f"{self.output_path}.revision.json")
        sidecar.write_text("existing sidecar", encoding="utf-8")
        self.assertEqual(self.revise("--delta-cartesian", "0", "0", "1"), 1)
        self.assertFalse(self.output_path.exists())
        self.assertEqual(sidecar.read_text(encoding="utf-8"), "existing sidecar")

    def test_second_publish_failure_removes_first_new_result(self) -> None:
        sidecar = self.directory / "audit.json"
        original_link = MODULE.os.link
        calls = 0

        def fail_second_publish(source: str, target: str) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated sidecar publish failure")
            original_link(source, target)

        with mock.patch.object(MODULE.os, "link", side_effect=fail_second_publish):
            self.assertEqual(self.revise("--delta-cartesian", "0", "0", "1", sidecar=sidecar), 1)
        self.assertFalse(self.output_path.exists())
        self.assertFalse(sidecar.exists())

    def test_geometry_analysis_avoids_quadratic_distances_and_is_shared_per_report(self) -> None:
        atoms = MODULE.load_structure(str(self.input_path))
        with mock.patch.object(Atoms, "get_all_distances", side_effect=AssertionError):
            summary = MODULE.structure_summary(atoms)
            issues = MODULE.validation_issues(atoms, 0.7)
        self.assertGreater(summary["minimum_pair_distance_angstrom"], 0.7)
        self.assertEqual(issues, [])
        with mock.patch.object(MODULE, "geometry_analysis", wraps=MODULE.geometry_analysis) as analyse:
            self.assertEqual(MODULE.main(["inspect", "--input", str(self.input_path)]), 0)
        self.assertEqual(analyse.call_count, 1)

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
