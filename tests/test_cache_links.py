import importlib.util
import json
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "deploy" / "huggingface_cache_links.py"
SPEC = importlib.util.spec_from_file_location("huggingface_cache_links", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cache_links = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache_links)

MANIFEST_SCHEMA = cache_links.MANIFEST_SCHEMA
CacheLinkError = cache_links.CacheLinkError
audit_links = cache_links.audit_links
capture_manifest = cache_links.capture_manifest
load_manifest = cache_links.load_manifest


def _cache(tmp_path):
    root = tmp_path / "downloads"
    repository = root / "models--example--tokenizer"
    blob = repository / "blobs" / "content-hash"
    blob.parent.mkdir(parents=True)
    blob.write_text("{}", encoding="utf-8")
    snapshot = repository / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    link = snapshot / "tokenizer.json"
    link.symlink_to("../../blobs/content-hash")
    return root, blob, link


def test_capture_repair(tmp_path):
    source, _, source_link = _cache(tmp_path / "source")
    manifest = capture_manifest(source)
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["links"] == [
        {
            "path": source_link.relative_to(source).as_posix(),
            "target": "../../blobs/content-hash",
        }
    ]

    destination, _, destination_link = _cache(tmp_path / "destination")
    destination_link.unlink()
    entries = [
        (Path(item["path"]), item["target"]) for item in manifest["links"]
    ]

    audit = audit_links(destination, entries)
    repaired = audit_links(destination, entries, repair=True)

    assert audit["missing"] == 1
    assert repaired["repaired"] == 1
    assert destination_link.is_symlink()
    assert destination_link.resolve().is_file()


def test_repair_conflict(tmp_path):
    root, _, link = _cache(tmp_path)
    link.unlink()
    link.write_text("materialized", encoding="utf-8")
    entries = [(link.relative_to(root), "../../blobs/content-hash")]

    result = audit_links(root, entries, repair=True)

    assert result["materialized"] == 1
    assert link.read_text(encoding="utf-8") == "materialized"


def test_partial_repair(tmp_path):
    root, _, missing_link = _cache(tmp_path)
    repository = missing_link.parents[2]
    second_blob = repository / "blobs" / "config-hash"
    second_blob.write_text("{}", encoding="utf-8")
    healthy_link = missing_link.parent / "tokenizer_config.json"
    healthy_link.symlink_to("../../blobs/config-hash")
    entries = [
        (missing_link.relative_to(root), "../../blobs/content-hash"),
        (healthy_link.relative_to(root), "../../blobs/config-hash"),
    ]
    missing_link.unlink()

    result = audit_links(root, entries, repair=True)

    assert result["healthy"] == 1
    assert result["repaired"] == 1
    assert missing_link.resolve().is_file()
    assert healthy_link.resolve() == second_blob


def test_manifest_traversal(tmp_path):
    manifest = tmp_path / "links.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "links": [{"path": "../outside", "target": "target"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CacheLinkError, match="Unsafe manifest path"):
        load_manifest(manifest)


def test_broken_blob(tmp_path):
    root, blob, link = _cache(tmp_path)
    blob.unlink()
    entries = [(link.relative_to(root), os.readlink(link))]

    result = audit_links(root, entries, repair=True)

    assert result["broken"] == 1
