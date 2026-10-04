"""Promote Windows Release render manifests into the shared golden checksum table."""
from __future__ import print_function, unicode_literals

import argparse
import io
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--header", type=Path,
                        default=Path(__file__).resolve().parent.parent /
                        "FractalSharkTest" / "GoldenChecksums.h")
    parser.add_argument("--inventory", type=Path,
                        help="prune retired case IDs using a current --list-tests output")
    args = parser.parse_args()
    with io.open(str(args.header), encoding="utf-8") as source:
        previous = source.read()
    entries = {}
    for fragments, crc in re.findall(
            r'\{\s*((?:"[^"]*"\s*)+),\s*"([0-9a-f]{16}|INCOMPLETE)"\s*\}', previous):
        case = "".join(re.findall(r'"([^"]*)"', fragments))
        entries[case] = crc
    # Migrate earlier platform-specific tables using the canonical Windows Release rows.
    for profile, fragments, crc in re.findall(
            r'\{\s*"([^"]+)",\s*((?:"[^"]*"\s*)+),\s*"([0-9a-f]{16}|INCOMPLETE)"\s*\}', previous):
        if profile == "windows-Release":
            case = "".join(re.findall(r'"([^"]*)"', fragments))
            entries[case] = crc
    candidates = {}
    for manifest in args.manifests:
        with io.open(str(manifest), encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                profile, case, crc = line.rstrip("\n").split("\t")
                if profile != "windows-Release":
                    raise ValueError("goldens require Windows Release input at {}:{}".format(manifest, number))
                if not re.match(r"^[0-9a-f]{16}$", crc) or '"' in case or "\\" in case:
                    raise ValueError("invalid checksum row at {}:{}".format(manifest, number))
                if case in candidates and candidates[case] != crc:
                    raise ValueError("conflicting candidates for {}".format(case))
                candidates[case] = crc
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
    with io.open(str(args.header), "w", encoding="utf-8", newline="\n") as output:
        output.write(content)
    print("Promoted {} candidate checksums; {} total entries".format(len(candidates), len(entries)))


if __name__ == "__main__":
    main()
