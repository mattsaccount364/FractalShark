#!/usr/bin/env python3
"""Report leading coordinate digits shared by successive NR checkpoints."""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


FIELD_PATTERN = re.compile(r"^(c_re|c_im|cand_re|cand_im):\s+(-?\d+)\s+(-?\d+)\s*$")
NATURAL_SORT_PATTERN = re.compile(r"(\d+)")
COORDINATE_FIELDS = ("c_re", "c_im", "cand_re", "cand_im")


@dataclass
class Checkpoint:
    path: Path
    coordinates: dict[str, tuple[str, str]]


def natural_sort_key(path: Path) -> list[object]:
    """Sort names such that Copy (2) comes before Copy (10)."""
    return [int(part) if part.isdigit() else part.casefold()
            for part in NATURAL_SORT_PATTERN.split(path.name)]


def read_checkpoint(path: Path) -> Checkpoint:
    """Read coordinate fields without relying on checkpoint metadata."""
    coordinates: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as checkpoint:
        for line in checkpoint:
            match = FIELD_PATTERN.match(line)
            if match:
                coordinates[match.group(1)] = (match.group(2), match.group(3))
                if len(coordinates) == len(COORDINATE_FIELDS):
                    return Checkpoint(path, coordinates)

    missing = sorted(set(COORDINATE_FIELDS) - coordinates.keys())
    raise ValueError(f"missing {', '.join(missing)}")


def shared_leading_digits(reference: tuple[str, str], value: tuple[str, str]) -> int:
    """Count identical leading decimal digits, after requiring matching exponents/signs."""
    reference_exponent, reference_digits = reference
    value_exponent, value_digits = value
    if reference_exponent != value_exponent or reference_digits.startswith("-") != value_digits.startswith("-"):
        return 0

    reference_digits = reference_digits.removeprefix("-")
    value_digits = value_digits.removeprefix("-")
    return next((index for index, (left, right) in enumerate(zip(reference_digits, value_digits))
                 if left != right), min(len(reference_digits), len(value_digits)))


def coordinates_equal(left: Checkpoint, right: Checkpoint) -> bool:
    """Return whether all serialized coordinate fields are exactly equal."""
    return left.coordinates == right.coordinates


def select_progression_runs(checkpoints: list[Checkpoint]) -> list[list[Checkpoint]]:
    """Collapse exact repeats and split a run when its baseline returns."""
    runs: list[list[Checkpoint]] = []
    for checkpoint in checkpoints:
        if not runs:
            runs.append([checkpoint])
            continue

        current_run = runs[-1]
        if coordinates_equal(current_run[-1], checkpoint):
            continue
        if len(current_run) > 1 and coordinates_equal(current_run[0], checkpoint):
            runs.append([checkpoint])
        else:
            current_run.append(checkpoint)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare coordinate leading digits between successive checkpoints.")
    parser.add_argument("directory", type=Path, help="directory containing NR checkpoint text files")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    checkpoints: list[Checkpoint] = []
    for path in sorted(args.directory.glob("nr_checkpoint*.txt"), key=natural_sort_key):
        try:
            checkpoints.append(read_checkpoint(path))
        except (OSError, UnicodeError, ValueError) as error:
            print(f"warning: skipping {path.name}: {error}", file=sys.stderr)

    if not checkpoints:
        parser.error("no readable checkpoint files containing c_re and c_im")

    runs = select_progression_runs(checkpoints)
    for run_number, run in enumerate(runs, start=1):
        initial = run[0]
        print(f"Run {run_number}: baseline {initial.path.name}")
        print(f"{'File':<68} {'prev c_re':>12} {'prev c_im':>12} "
              f"{'base c_re':>12} {'base c_im':>12} "
              f"{'prev cand_re':>14} {'prev cand_im':>14} "
              f"{'base cand_re':>14} {'base cand_im':>14}")
        print("-" * 178)
        previous = None
        for checkpoint in run:
            if previous is None:
                previous_c_re = previous_c_im = previous_cand_re = previous_cand_im = "-"
            else:
                previous_c_re = str(shared_leading_digits(
                    previous.coordinates['c_re'], checkpoint.coordinates['c_re']))
                previous_c_im = str(shared_leading_digits(
                    previous.coordinates['c_im'], checkpoint.coordinates['c_im']))
                previous_cand_re = str(shared_leading_digits(
                    previous.coordinates['cand_re'], checkpoint.coordinates['cand_re']))
                previous_cand_im = str(shared_leading_digits(
                    previous.coordinates['cand_im'], checkpoint.coordinates['cand_im']))

            print(f"{checkpoint.path.name:<68} {previous_c_re:>12} {previous_c_im:>12} "
                  f"{shared_leading_digits(initial.coordinates['c_re'], checkpoint.coordinates['c_re']):>12} "
                  f"{shared_leading_digits(initial.coordinates['c_im'], checkpoint.coordinates['c_im']):>12} "
                  f"{previous_cand_re:>14} {previous_cand_im:>14} "
                  f"{shared_leading_digits(initial.coordinates['cand_re'], checkpoint.coordinates['cand_re']):>14} "
                  f"{shared_leading_digits(initial.coordinates['cand_im'], checkpoint.coordinates['cand_im']):>14}")
            previous = checkpoint

        candidate_pairs = {
            (checkpoint.coordinates['cand_re'], checkpoint.coordinates['cand_im']) for checkpoint in run
        }
        candidate_status = "unchanged" if len(candidate_pairs) == 1 else "changed"
        print(f"Candidate pair: {candidate_status} ({len(candidate_pairs)} distinct pair(s))")

        if run_number != len(runs):
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
