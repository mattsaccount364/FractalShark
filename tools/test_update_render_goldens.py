"""Focused tests for selective golden generation and promotion (Python 3)."""
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import update_render_goldens


class UpdateRenderGoldensTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.header = self.root / "GoldenChecksums.h"
        self.previous = ('{"RenderGolden_A", "0000000000000001"},\n'
                         '{"RenderGolden_B", "0000000000000002"},\n'
                         '{"RenderGolden_Disabled", "INCOMPLETE"},\n')
        self.header.write_text(self.previous, encoding="utf-8")

    def invoke(self, arguments):
        with mock.patch("sys.argv", ["update_render_goldens.py", "--header", str(self.header)] + arguments):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                update_render_goldens.main()

    def manifest(self, name, rows):
        path = self.root / name
        path.write_text(rows, encoding="utf-8")
        return path

    def test_partial_merge_preserves_unselected_entries_and_placeholders(self):
        manifest = self.manifest("checksums.tsv", "windows-Release\tRenderGolden_A\t0000000000000003\n")
        self.invoke([str(manifest)])
        entries = update_render_goldens.read_entries(self.header.read_text(encoding="utf-8"))
        self.assertEqual(entries, {"RenderGolden_A": "0000000000000003",
                                   "RenderGolden_B": "0000000000000002",
                                   "RenderGolden_Disabled": "INCOMPLETE"})
        self.assertEqual(self.header.read_text(encoding="utf-8"),
                         self.previous.replace("0000000000000001", "0000000000000003"))

    def test_wrapped_names_and_comments_keep_their_formatting(self):
        previous = ('// retained comment\n    {"RenderGolden_"\n'
                    '     "A",\n     "0000000000000001"},\n')
        self.assertEqual(update_render_goldens.revise_existing_entries(
            previous, {"RenderGolden_A": "0000000000000003"}),
            previous.replace("0000000000000001", "0000000000000003"))

    def test_revision_preserves_windows_line_endings(self):
        previous = self.previous.replace("\n", "\r\n").encode("utf-8")
        self.header.write_bytes(previous)
        manifest = self.manifest("checksums.tsv", "windows-Release\tRenderGolden_A\t0000000000000003\n")
        self.invoke([str(manifest)])
        self.assertEqual(self.header.read_bytes(),
                         previous.replace(b"0000000000000001", b"0000000000000003"))

    def test_new_entries_expand_the_shared_table(self):
        manifest = self.manifest("checksums.tsv", "windows-Release\tRenderGolden_New\t0000000000000003\n")
        self.invoke([str(manifest)])
        content = self.header.read_text(encoding="utf-8")
        self.assertIn("std::array<GoldenChecksum, 4>", content)
        entries = update_render_goldens.read_entries(content)
        self.assertEqual(entries["RenderGolden_A"], "0000000000000001")
        self.assertEqual(entries["RenderGolden_New"], "0000000000000003")

    def test_unchanged_candidates_do_not_rewrite_header(self):
        manifest = self.manifest("checksums.tsv", "windows-Release\tRenderGolden_A\t0000000000000001\n")
        self.invoke([str(manifest)])
        self.assertEqual(self.header.read_text(encoding="utf-8"), self.previous)

    def test_conflicting_candidates_and_wrong_profiles_leave_header_untouched(self):
        for rows in ("windows-Release\tRenderGolden_A\t0000000000000003\n"
                     "windows-Release\tRenderGolden_A\t0000000000000004\n",
                     "linux-Release\tRenderGolden_A\t0000000000000003\n",
                     "windows-Release\tRenderGolden_A\t000000000000000g\n", ""):
            with self.subTest(rows=rows):
                manifest = self.manifest("invalid.tsv", rows)
                with self.assertRaises(ValueError):
                    self.invoke([str(manifest)])
                self.assertEqual(self.header.read_text(encoding="utf-8"), self.previous)

    def test_render_requires_filters_and_disallows_pruning(self):
        for arguments in (["--render"], ["--render", "--filter", "RenderGolden_A",
                                       "--inventory", "partial-inventory.txt"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.invoke(arguments)
        self.assertEqual(self.header.read_text(encoding="utf-8"), self.previous)

    def test_disabled_gpu_and_non_golden_selections_rejected(self):
        for inventory in ("RenderGolden_A [disabled: incomplete]", "RenderGolden_A [requires --use-gpu]",
                          "UnitTest_A", ""):
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                update_render_goldens.selected_cases(inventory, False)
        self.assertEqual(update_render_goldens.selected_cases(
            "RenderGolden_A [requires --use-gpu]\n", True), ["RenderGolden_A"])

    def test_multi_image_coverage_and_unselected_candidates(self):
        entries = {"RenderGolden_A_Original": "0000000000000001",
                   "RenderGolden_A_Reset": "0000000000000002",
                   "RenderGolden_B": "0000000000000003"}
        candidates = {"RenderGolden_A_Original": "0000000000000004",
                      "RenderGolden_A_Reset": "0000000000000005"}
        provenance = "profile=windows-Release\ngeneration=1\n"
        output = "1 generated, 0 skipped, 0 disabled\nRESULT: GENERATION COMPLETED (goldens not validated)\n"
        registered = {"RenderGolden_A", "RenderGolden_B"}
        update_render_goldens.validate_generation(
            provenance, output, candidates, ["RenderGolden_A"], entries, registered)
        for invalid in ({"RenderGolden_A_Original": "0000000000000004"},
                        dict(candidates, RenderGolden_B="0000000000000006")):
            with self.subTest(candidates=invalid), self.assertRaises(ValueError):
                update_render_goldens.validate_generation(
                    provenance, output, invalid, ["RenderGolden_A"], entries, registered)
        for invalid_provenance, invalid_output in (
                ("profile=windows-Debug\ngeneration=1\n", output),
                (provenance, output.replace("0 skipped", "1 skipped")),
                (provenance, "")):
            with self.subTest(output=invalid_output), self.assertRaises(ValueError):
                update_render_goldens.validate_generation(
                    invalid_provenance, invalid_output, candidates, ["RenderGolden_A"], entries, registered)

    def test_registered_prefix_siblings_can_be_updated_independently(self):
        cpu = "RenderGolden_AutoSelection_View1"
        gpu = cpu + "_AutoGpu"
        entries = {cpu: "0000000000000001", gpu: "0000000000000002"}
        registered = {cpu, gpu}
        provenance = "profile=windows-Release\ngeneration=1\n"
        output = "1 generated, 0 skipped, 0 disabled\nRESULT: GENERATION COMPLETED (goldens not validated)\n"
        for selected in (cpu, gpu):
            with self.subTest(selected=selected):
                update_render_goldens.validate_generation(
                    provenance, output, {selected: "0000000000000003"}, [selected], entries, registered)
                with self.assertRaises(ValueError):
                    update_render_goldens.validate_generation(
                        provenance, output, entries, [selected], entries, registered)

    def test_longest_registered_parent_still_requires_all_child_artifacts(self):
        parent = "RenderGolden_A"
        sibling = parent + "_Longer"
        entries = {parent + "_Original": "0000000000000001",
                   sibling + "_Original": "0000000000000002",
                   sibling + "_Reset": "0000000000000003"}
        registered = {parent, sibling}
        provenance = "profile=windows-Release\ngeneration=1\n"
        output = "1 generated, 0 skipped, 0 disabled\nRESULT: GENERATION COMPLETED (goldens not validated)\n"
        update_render_goldens.validate_generation(
            provenance, output, {parent + "_Original": "0000000000000004"},
            [parent], entries, registered)
        with self.assertRaises(ValueError):
            update_render_goldens.validate_generation(
                provenance, output, {sibling + "_Original": "0000000000000004"},
                [sibling], entries, registered)

    def test_failed_render_does_not_promote_partial_results(self):
        process = mock.MagicMock()
        process.__enter__.return_value = process
        process.stdout = io.StringIO("partial generation\n")
        process.wait.return_value = 1
        with mock.patch("update_render_goldens.subprocess.run", return_value=mock.Mock(stdout="RenderGolden_A\n")):
            with mock.patch("update_render_goldens.subprocess.Popen", return_value=process):
                with self.assertRaises(subprocess.CalledProcessError):
                    self.invoke(["--render", "--filter", "RenderGolden_A", "--output-dir", str(self.root)])
        self.assertEqual(self.header.read_text(encoding="utf-8"), self.previous)

    def test_render_runs_only_selected_filters_and_merges_results(self):
        output_root = self.root / "output"
        process = mock.MagicMock()
        process.__enter__.return_value = process
        process.wait.return_value = 0
        process.stdout = io.StringIO("1 generated, 0 skipped, 0 disabled\n"
                                    "RESULT: GENERATION COMPLETED (goldens not validated)\n")

        def generated_process(command, **kwargs):
            invocation = Path(command[command.index("--output-dir") + 1])
            run = invocation / "windows-Release" / "test-run"
            run.mkdir(parents=True)
            (run / "checksums.tsv").write_text(
                "windows-Release\tRenderGolden_A\t0000000000000003\n", encoding="utf-8")
            (run / "provenance.txt").write_text("profile=windows-Release\ngeneration=1\n", encoding="utf-8")
            return process

        with mock.patch("update_render_goldens.subprocess.run", return_value=mock.Mock(stdout="RenderGolden_A\n")) as inventory:
            with mock.patch("update_render_goldens.subprocess.Popen", side_effect=generated_process) as render:
                self.invoke(["--render", "--filter", "RenderGolden_A", "--exclude", "RenderGolden_B",
                             "--output-dir", str(output_root)])
        self.assertIn("--list-tests", inventory.call_args.args[0])
        command = render.call_args.args[0]
        self.assertEqual(command[1:5], ["--filter", "RenderGolden_A", "--exclude", "RenderGolden_B"])
        self.assertIn("--generate-goldens", command)
        entries = update_render_goldens.read_entries(self.header.read_text(encoding="utf-8"))
        self.assertEqual(entries["RenderGolden_A"], "0000000000000003")
        self.assertEqual(entries["RenderGolden_B"], "0000000000000002")
        self.assertEqual(entries["RenderGolden_Disabled"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
