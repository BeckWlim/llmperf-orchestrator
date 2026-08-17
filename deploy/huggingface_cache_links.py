#!/usr/bin/env python3
"""Capture, audit, and repair Hugging Face cache snapshot symlinks.

Hugging Face cache snapshots map repository filenames to content-addressed blobs
with relative symlinks. Some copy tools omit those links while retaining both the
snapshot directories and blobs. A link manifest captured on the healthy source is
therefore required for deterministic, network-free repair on the destination.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


MANIFEST_SCHEMA = "llmperf-huggingface-cache-links/v1"


class CacheLinkError(RuntimeError):
    """Raised when a cache or manifest is unsafe or inconsistent."""


def _within(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((str(root), str(candidate))) == str(root)
    except ValueError:
        return False


def _relative_path(raw_path: str) -> PurePosixPath:
    path = PurePosixPath(raw_path)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CacheLinkError(f"Unsafe manifest path: {raw_path!r}")
    if ".locks" in path.parts or "snapshots" not in path.parts:
        raise CacheLinkError(f"Manifest path is not a snapshot entry: {raw_path!r}")
    return path


def _target_path(root: Path, link: Path, raw_target: str) -> Path:
    target = PurePosixPath(raw_target)
    if target.is_absolute():
        raise CacheLinkError(f"Absolute symlink target is not allowed: {raw_target!r}")
    resolved = (link.parent / Path(*target.parts)).resolve()
    if not _within(root, resolved):
        raise CacheLinkError(
            f"Symlink target escapes cache root: {link} -> {raw_target}"
        )
    return resolved


def _snapshot_links(cache_root: Path) -> Iterable[Path]:
    for directory, names, filenames in os.walk(cache_root, followlinks=False):
        names[:] = sorted(name for name in names if name != ".locks")
        current = Path(directory)
        for filename in sorted(filenames):
            candidate = current / filename
            relative = candidate.relative_to(cache_root)
            if "snapshots" in relative.parts and candidate.is_symlink():
                yield candidate


def capture_manifest(cache_root: Path) -> Dict[str, Any]:
    root = cache_root.expanduser().resolve()
    if not root.is_dir():
        raise CacheLinkError(f"Cache root is not a directory: {root}")
    links: List[Dict[str, str]] = []
    for link in _snapshot_links(root):
        raw_target = os.readlink(link)
        target = _target_path(root, link, raw_target)
        if not target.is_file():
            raise CacheLinkError(f"Source cache contains a broken link: {link}")
        links.append(
            {
                "path": link.relative_to(root).as_posix(),
                "target": raw_target,
            }
        )
    if not links:
        raise CacheLinkError(f"No snapshot symlinks found under cache root: {root}")
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "links": links,
    }


def write_manifest(path: Path, manifest: Mapping[str, Any], force: bool) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() and not force:
        raise CacheLinkError(
            f"Manifest already exists: {destination}; pass --force to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_manifest(path: Path) -> List[Tuple[PurePosixPath, str]]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheLinkError(f"Unable to read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise CacheLinkError(f"Unsupported manifest schema in {path}")
    raw_links = payload.get("links")
    if not isinstance(raw_links, list) or not raw_links:
        raise CacheLinkError(f"Manifest contains no links: {path}")
    entries: List[Tuple[PurePosixPath, str]] = []
    seen = set()
    for raw_entry in raw_links:
        if not isinstance(raw_entry, dict):
            raise CacheLinkError("Manifest link entries must be objects")
        raw_path = raw_entry.get("path")
        raw_target = raw_entry.get("target")
        if not isinstance(raw_path, str) or not isinstance(raw_target, str):
            raise CacheLinkError("Manifest links require string path and target")
        relative = _relative_path(raw_path)
        if relative in seen:
            raise CacheLinkError(f"Duplicate manifest path: {raw_path}")
        seen.add(relative)
        entries.append((relative, raw_target))
    return entries


def audit_links(
    cache_root: Path,
    entries: Sequence[Tuple[PurePosixPath, str]],
    repair: bool = False,
) -> Dict[str, int]:
    root = cache_root.expanduser().resolve()
    if not root.is_dir():
        raise CacheLinkError(f"Cache root is not a directory: {root}")
    counts = {
        "healthy": 0,
        "materialized": 0,
        "missing": 0,
        "broken": 0,
        "conflict": 0,
        "repaired": 0,
    }
    for relative, raw_target in entries:
        link = root.joinpath(*relative.parts)
        if not _within(root, link.parent.resolve()):
            raise CacheLinkError(f"Manifest path escapes cache root: {relative}")
        target = _target_path(root, link, raw_target)
        if link.is_symlink():
            if os.readlink(link) != raw_target:
                counts["conflict"] += 1
                print(f"conflict {relative}: existing target={os.readlink(link)!r}")
            elif target.is_file():
                counts["healthy"] += 1
            else:
                counts["broken"] += 1
                print(f"broken {relative}: blob target is missing")
            continue
        if link.exists():
            counts["materialized"] += 1
            continue
        if not target.is_file():
            counts["broken"] += 1
            print(f"broken {relative}: blob target is missing")
            continue
        if not repair:
            counts["missing"] += 1
            print(f"missing {relative}")
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(raw_target, link)
        except FileExistsError:
            counts["conflict"] += 1
            print(f"conflict {relative}: appeared during repair")
        else:
            counts["repaired"] += 1
            print(f"repaired {relative} -> {raw_target}")
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture, audit, or repair Hugging Face cache snapshot links."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture", help="Capture healthy source-cache links into a manifest."
    )
    capture.add_argument("--cache-root", type=Path, required=True)
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("--force", action="store_true")

    for command in ("audit", "repair"):
        operation = subparsers.add_parser(
            command,
            help=(
                "Audit a destination cache without changes."
                if command == "audit"
                else "Create only missing links whose blobs are present."
            ),
        )
        operation.add_argument("--cache-root", type=Path, required=True)
        operation.add_argument("--manifest", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] = ()) -> int:
    parsed = _parser().parse_args(list(arguments) if arguments else None)
    try:
        if parsed.command == "capture":
            manifest = capture_manifest(parsed.cache_root)
            write_manifest(parsed.manifest, manifest, parsed.force)
            print(
                f"captured={len(manifest['links'])} manifest={parsed.manifest.expanduser()}"
            )
            return 0
        entries = load_manifest(parsed.manifest)
        counts = audit_links(
            parsed.cache_root,
            entries,
            repair=parsed.command == "repair",
        )
        print(" ".join(f"{key}={value}" for key, value in counts.items()))
        unresolved = counts["missing"] + counts["broken"] + counts["conflict"]
        return 1 if unresolved else 0
    except CacheLinkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
