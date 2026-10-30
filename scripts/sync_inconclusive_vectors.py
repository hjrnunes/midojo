#!/usr/bin/env python3
"""Verify or refresh the byte copies of the inconclusive vectors.

    python3 scripts/sync_inconclusive_vectors.py --check \\
        --consumer ../asago-artifact-generator --garak ../garak
    python3 scripts/sync_inconclusive_vectors.py --write --garak ../garak

MiDojo owns the vectors: ``tests/fixtures/inconclusive_vectors`` holds the sealed cases,
``generate.py`` that re-seals them from ``recording.json``, and ``MANIFEST.json`` that pins
every file's sha256. A copy is a byte copy of the pinned cases and of the manifest, at the
directory named in ``COPIES``. The consumer's MiDojo ``parse``, its Garak report route and
Garak's ``ipi_episode`` probe and detectors read a copy from their own suites, and no test
reads MiDojo's tree at run time.

``--check`` verifies the owner's tree against its own manifest and each copy it is given
against the same manifest, the manifest's own bytes included, writes nothing, and exits 1 on
drift. ``--write`` replaces a copy with the pinned files and then verifies it; it never edits
the owner. Without a checkout argument the script reads no directory outside MiDojo.

The script needs only the standard library. Run it from any interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REFERENCE = "MANIFEST.json"
SCHEMA_VERSION = "inconclusive-vectors-manifest-v1"

# The copy directory of each repository, relative to the checkout root.
COPIES = {
    "consumer": Path("packages/asago-bundle-midojo/tests/fixtures/inconclusive_vectors"),
    "garak": Path("tests/_assets/detectors/inconclusive_vectors"),
}

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/inconclusive_vectors"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> dict[str, Path]:
    """Every vector file under ``root`` by relative path, the manifest excluded."""

    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != REFERENCE and "__pycache__" not in path.parts
    }


def _read_manifest(path: Path) -> dict[str, object] | str:
    """The parsed manifest, or a problem string when it cannot be used."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"{path.name} is unreadable: {exc}"
    if not isinstance(document, dict):
        return f"{path.name} is not a JSON object"
    if document.get("schema_version") != SCHEMA_VERSION:
        return f"{path.name} names schema {document.get('schema_version')!r}, the owner's is {SCHEMA_VERSION!r}"
    for key in ("files", "copies", "cases"):
        if not isinstance(document.get(key), dict | list):
            return f"{path.name} lists no {key}"
    return document


def _recorded(document: dict[str, object]) -> dict[str, str]:
    files = document["files"]
    assert isinstance(files, dict)
    return {str(name): str(digest) for name, digest in files.items()}


def _copied(document: dict[str, object]) -> list[str]:
    copies = document["copies"]
    assert isinstance(copies, list)
    return [str(name) for name in copies]


def _cases(document: dict[str, object]) -> list[str]:
    cases = document["cases"]
    assert isinstance(cases, list)
    return [str(name) for name in cases]


def owner_problems(root: Path = ROOT) -> list[str]:
    """Every way the owner's tree disagrees with its own manifest."""

    reference = root / REFERENCE
    if not reference.is_file():
        return [f"the reference {REFERENCE} is missing"]
    document = _read_manifest(reference)
    if isinstance(document, str):
        return [document]
    problems: list[str] = []
    recorded = _recorded(document)
    on_disk = _files(root)
    for name in sorted(set(recorded) | set(on_disk)):
        if name not in on_disk:
            problems.append(f"{name} is missing")
        elif name not in recorded:
            problems.append(f"{name} is not in {REFERENCE}")
        elif _sha256(on_disk[name]) != recorded[name]:
            problems.append(
                f"{name} differs from {REFERENCE} (recorded {recorded[name]}, found {_sha256(on_disk[name])})"
            )
    for name in _copied(document):
        if name not in recorded:
            problems.append(f"{name} is listed as copied but not pinned")
    cases = set(_cases(document))
    if sorted(_copied(document)) != sorted(name for name in recorded if name.split("/")[0] in cases):
        problems.append("the copied files are not the pinned cases")
    return problems


def copy_problems(name: str, checkout: Path, *, root: Path = ROOT) -> list[str]:
    """Every way one copy disagrees with the owner's manifest."""

    copy = checkout / COPIES[name]
    if not copy.is_dir():
        return [f"copy directory {COPIES[name]} is missing"]
    document = _read_manifest(root / REFERENCE)
    if isinstance(document, str):
        return [f"the owner's manifest is unusable: {document}"]
    recorded = _recorded(document)
    problems: list[str] = []
    reference = copy / REFERENCE
    if not reference.is_file():
        problems.append(f"the reference is missing: {REFERENCE} is not in the copy")
    elif _sha256(reference) != _sha256(root / REFERENCE):
        # Byte for byte, not document for document: a copy of the manifest is its bytes.
        problems.append(f"the reference {REFERENCE} differs from the owner's manifest")
    expected = {file: recorded[file] for file in _copied(document)}
    present = _files(copy)
    for file in sorted(set(expected) | set(present)):
        if file not in present:
            problems.append(f"{file} is missing from the copy")
        elif file not in expected:
            problems.append(f"{file} is not in the owner's vectors")
        elif _sha256(present[file]) != expected[file]:
            problems.append(
                f"{file} differs from the owner's file (expected {expected[file]}, found {_sha256(present[file])})"
            )
    return problems


def refresh_copy(name: str, checkout: Path, *, root: Path = ROOT) -> list[str]:
    """Replace one copy with the owner's pinned files, then return its problems."""

    copy = checkout / COPIES[name]
    document = _read_manifest(root / REFERENCE)
    if isinstance(document, str):
        return [f"the owner's manifest is unusable: {document}"]
    expected = {file: root / file for file in _copied(document)}
    expected[REFERENCE] = root / REFERENCE
    for file, path in _files(copy).items():
        if file not in expected:
            path.unlink()
    copy.mkdir(parents=True, exist_ok=True)
    for file, path in expected.items():
        target = copy / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for directory in sorted((path for path in copy.rglob("*") if path.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    return copy_problems(name, checkout, root=root)


def _report(label: str, problems: list[str], detail: str) -> None:
    if problems:
        print(f"{label}: DIFFERS ({detail})")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print(f"{label}: verified ({detail})")


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check", action="store_true", help="verify the owner's tree and every copy given (the default)"
    )
    modes.add_argument("--write", action="store_true", help="replace each copy given with the owner's files")
    for name in COPIES:
        parser.add_argument(
            f"--{name}", type=Path, metavar="CHECKOUT", help=f"{name}'s checkout root, which holds {COPIES[name]}"
        )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    root = ROOT if root is None else root
    args = _arguments(argv)
    problems = owner_problems(root)
    document = _read_manifest(root / REFERENCE)
    detail = f"{len(_copied(document))} copied files" if isinstance(document, dict) else "unusable manifest"
    _report(f"owner {root}", problems, detail)
    given = [name for name in COPIES if getattr(args, name) is not None]
    if not given:
        print("no copies checked (pass --consumer or --garak)")
        return 1 if problems else 0
    failed = bool(problems)
    for name in given:
        checkout = getattr(args, name)
        label = f"{name} copy {checkout / COPIES[name]}"
        if args.write:
            found = refresh_copy(name, checkout, root=root)
            if found:
                failed = True
                print(f"{label}: DIFFERS after writing")
                for problem in found:
                    print(f"  - {problem}")
            else:
                print(f"{label}: refreshed ({detail})")
            continue
        found = copy_problems(name, checkout, root=root)
        failed = failed or bool(found)
        _report(label, found, detail)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
