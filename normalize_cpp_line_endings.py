#!/usr/bin/env python3
"""Normalize tracked C++ and CUDA source files to CRLF line endings."""

import os
from pathlib import Path
import subprocess


SOURCE_PATTERNS = ("*.cpp", "*.h", "*.cu", "*.cuh", "*.cc", "*.hh", "*.hpp")


def NormalizeLineEndings(contents: bytes) -> bytes:
    return contents.replace(b"\r\n", b"\n").replace(b"\r", b"\n").replace(b"\n", b"\r\n")


def GetTrackedSourcePaths(repositoryRoot: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(repositoryRoot), "ls-files", "-z", "--", *SOURCE_PATTERNS],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        repositoryRoot / os.fsdecode(relativePath)
        for relativePath in result.stdout.split(b"\0")
        if relativePath
    ]


def main() -> int:
    repositoryRoot = Path(__file__).resolve().parent
    normalizedCount = 0
    missingCount = 0
    sourcePaths = GetTrackedSourcePaths(repositoryRoot)

    for sourcePath in sourcePaths:
        if not sourcePath.is_file():
            missingCount += 1
            continue

        contents = sourcePath.read_bytes()
        normalizedContents = NormalizeLineEndings(contents)
        if contents != normalizedContents:
            sourcePath.write_bytes(normalizedContents)
            normalizedCount += 1

    print(f"Normalized {normalizedCount} of {len(sourcePaths)} tracked C++/CUDA files.")
    if missingCount:
        print(f"Skipped {missingCount} tracked file(s) that are absent from the working tree.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
