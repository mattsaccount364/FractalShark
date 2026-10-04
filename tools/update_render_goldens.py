"""Promote Windows Release render manifests into the shared golden checksum table."""
from __future__ import print_function, unicode_literals

import argparse
import io
import re
import subprocess
import uuid
from pathlib import Path


SHARED_ENTRY = r'\{\s*((?:"[^"]*"\s*)+),\s*"([0-9a-f]{16}|INCOMPLETE)"\s*\}'


def read_entries(previous):
    entries = {}
    for fragments, crc in re.findall(SHARED_ENTRY, previous):
        case = "".join(re.findall(r'"([^"]*)"', fragments))
        entries[case] = crc
    # Migrate earlier platform-specific tables using the canonical Windows Release rows.
    for profile, fragments, crc in re.findall(
            r'\{\s*"([^"]+)",\s*((?:"[^"]*"\s*)+),\s*"([0-9a-f]{16}|INCOMPLETE)"\s*\}', previous):
        if profile == "windows-Release":
            case = "".join(re.findall(r'"([^"]*)"', fragments))
            entries[case] = crc
    return entries


def revise_existing_entries(previous, candidates):
    revised = set()

    def replace(match):
        case = "".join(re.findall(r'"([^"]*)"', match.group(1)))
        if case not in candidates:
            return match.group(0)
        revised.add(case)
        start = match.start(2) - match.start()
        end = match.end(2) - match.start()
        return match.group(0)[:start] + candidates[case] + match.group(0)[end:]

    content = re.sub(SHARED_ENTRY, replace, previous)
    return content if candidates.keys() <= revised else None


def read_candidates(manifests):
    candidates = {}
    for manifest in manifests:
        with io.open(str(manifest), encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                profile, case, crc = line.rstrip("\r\n").split("\t")
                if profile != "windows-Release":
                    raise ValueError("goldens require Windows Release input at {}:{}".format(manifest, number))
                if not re.fullmatch(r"[0-9a-f]{16}", crc) or not case or '"' in case or "\\" in case:
                    raise ValueError("invalid checksum row at {}:{}".format(manifest, number))
                if case in candidates and candidates[case] != crc:
                    raise ValueError("conflicting candidates for {}".format(case))
                candidates[case] = crc
    if not candidates:
        raise ValueError("no candidate checksums found")
    return candidates


def belongs_to(case, parent):
    return case == parent or case.startswith(parent + "_")


def artifact_parent(case, registered):
    return max((name for name in registered if belongs_to(case, name)), key=len, default=None)


def selected_cases(inventory, use_gpu):
    names = []
    for line in inventory.splitlines():
        if not line.strip():
            continue
        name = line.split(" [", 1)[0].strip()
        if not name.startswith("RenderGolden_") or any(char.isspace() for char in name):
            raise ValueError("selection contains a non-golden test: {}".format(line))
        if " [disabled:" in line:
            raise ValueError("selection contains a disabled test: {}".format(line))
        if " [requires --use-gpu]" in line and not use_gpu:
            raise ValueError("selection requires --use-gpu: {}".format(name))
        names.append(name)
    if not names:
        raise ValueError("no golden tests selected")
    return names


def validate_generation(provenance, output, candidates, names, entries, registered):
    metadata = dict(line.split("=", 1) for line in provenance.splitlines() if "=" in line)
    if metadata.get("profile") != "windows-Release" or metadata.get("generation") != "1":
        raise ValueError("generation requires Windows Release provenance")
    summary = re.search(r"(\d+) generated, (\d+) skipped, (\d+) disabled", output)
    if (summary is None or tuple(map(int, summary.groups())) != (len(names), 0, 0)
            or "RESULT: GENERATION COMPLETED (goldens not validated)" not in output):
        raise ValueError("generation did not complete every selected test")
    selected = set(names)
    parents = {case: artifact_parent(case, registered) for case in entries.keys() | candidates.keys()}
    for case in candidates:
        if parents[case] not in selected:
            raise ValueError("unselected candidate: {}".format(case))
    for name in names:
        if not any(parents[case] == name for case in candidates):
            raise ValueError("no checksums generated for {}".format(name))
    expected = {case for case in entries if parents[case] in selected}
    missing = expected - candidates.keys()
    if missing:
        raise ValueError("missing selected artifacts: {}".format(", ".join(sorted(missing))))


def render_selected(args, entries, repository):
    executable = (args.test_exe or repository / "Release" / "FractalSharkTest.exe").resolve()
    registry = subprocess.run([str(executable), "--list-tests"], cwd=str(repository),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding="utf-8", errors="replace", check=True).stdout
    registered = {line.split(" [", 1)[0].strip() for line in registry.splitlines() if line.strip()}
    command = [str(executable)]
    for pattern in args.filter:
        command.extend(["--filter", pattern])
    for pattern in args.exclude:
        command.extend(["--exclude", pattern])
    if args.use_gpu:
        command.append("--use-gpu")
    inventory = subprocess.run(command + ["--list-tests"], cwd=str(repository),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding="utf-8", errors="replace", check=True).stdout
    names = selected_cases(inventory, args.use_gpu)
    print("Selected {} golden test(s):\n{}".format(len(names), "\n".join(names)), flush=True)
    root = (args.output_dir or repository / "validation-render-goldens").resolve()
    invocation = root / ("update-" + uuid.uuid4().hex[:8])
    invocation.mkdir(parents=True)
    generation = command + ["--generate-goldens", "--output-dir", str(invocation)]
    lines = []
    with io.open(str(invocation / "generation.log"), "w", encoding="utf-8") as log:
        with subprocess.Popen(generation, cwd=str(repository), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                lines.append(line)
            return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, generation)
    manifests = list(invocation.glob("windows-Release/*/checksums.tsv"))
    if len(manifests) != 1:
        raise ValueError("expected one Windows Release manifest beneath {}".format(invocation))
    manifest = manifests[0]
    candidates = read_candidates(manifests)
    provenance = (manifest.parent / "provenance.txt").read_text(encoding="utf-8")
    validate_generation(provenance, "".join(lines), candidates, names, entries, registered)
    print("Retained images and artifact manifest: {}".format(manifest.parent))
    print("After rebuilding with the updated header, verify only these cases:\n& {}".format(
        subprocess.list2cmdline(command)))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument("--render", action="store_true", help="render selected tests and merge their goldens")
    parser.add_argument("--filter", action="append", default=[], help="test name or wildcard; repeat to include groups")
    parser.add_argument("--exclude", action="append", default=[], help="exclude matching test names")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--test-exe", type=Path, help="Windows Release test executable")
    parser.add_argument("--output-dir", type=Path, help="root for retained generation artifacts")
    parser.add_argument("--header", type=Path,
                        default=Path(__file__).resolve().parent.parent /
                        "FractalSharkTest" / "GoldenChecksums.h")
    parser.add_argument("--inventory", type=Path,
                        help="prune retired case IDs using a current --list-tests output")
    args = parser.parse_args()
    if args.render:
        if not args.filter or args.manifests or args.inventory:
            parser.error("--render requires --filter and cannot use manifests or --inventory")
    elif (not args.manifests or args.filter or args.exclude or args.use_gpu
          or args.test_exe or args.output_dir):
        parser.error("supply manifests, or use --render with explicit --filter arguments")
    with io.open(str(args.header), encoding="utf-8", newline="") as source:
        previous = source.read()
    entries = read_entries(previous)
    candidates = (render_selected(args, entries, Path(__file__).resolve().parent.parent)
                  if args.render else read_candidates(args.manifests))
    changed = {case: crc for case, crc in candidates.items() if entries.get(case) != crc}
    for case, crc in sorted(changed.items()):
        print("  {}: {} -> {}".format(case, entries.get(case, "NEW"), crc))
    entries.update(candidates)
    if args.inventory is not None:
        with io.open(str(args.inventory), encoding="utf-8-sig") as source:
            names = set(line.split(" [", 1)[0].strip() for line in source)
        names = set(name for name in names if name and not any(char.isspace() for char in name))
        def registered(case):
            return case in names or any(case[:index] in names
                                        for index, char in enumerate(case) if char == "_")
        entries = {case: crc for case, crc in entries.items() if registered(case)}
    rows = ["    {{\"{}\", \"{}\"}},".format(case, crc)
            for case, crc in sorted(entries.items())]
    content = revise_existing_entries(previous, candidates) if args.inventory is None else None
    if content is None:
        content = '''#pragma once

#include <array>
#include <string_view>

namespace RenderTests {
struct GoldenChecksum {
    const char *CaseId;
    const char *Crc;
};

// Shared Windows Release baseline: CRC-64 of decoded RGBA16 PNG bytes or canonical reference text.
inline constexpr std::array<GoldenChecksum, %d> GoldenChecksums{{
%s
}};
inline constexpr std::string_view IncompleteCrc = "INCOMPLETE";
}
''' % (len(entries), "\n".join(rows))
    if changed or args.inventory is not None:
        with io.open(str(args.header), "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
    print("Promoted {} changed checksums; {} unchanged candidates; {} total entries".format(
        len(changed), len(candidates) - len(changed), len(entries)))


if __name__ == "__main__":
    main()
