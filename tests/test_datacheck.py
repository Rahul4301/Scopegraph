import hashlib

import yaml
from conftest import ROOT

from memstudy.datacheck import verify


def _manifest(tmp_path, content: bytes, sha: str | None = None):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "f.json").write_bytes(content)
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "files": [
                    {
                        "path": "data/f.json",
                        "bytes": len(content),
                        "sha256": sha or hashlib.sha256(content).hexdigest(),
                    }
                ]
            }
        )
    )
    return manifest


def test_matching_file_is_ok(tmp_path):
    assert verify(_manifest(tmp_path, b"abc"), tmp_path)[0]["status"] == "ok"


def test_changed_file_is_a_mismatch(tmp_path):
    manifest = _manifest(tmp_path, b"abc", sha="0" * 64)
    assert verify(manifest, tmp_path)[0]["status"] == "mismatch"


def test_absent_file_is_missing(tmp_path):
    manifest = _manifest(tmp_path, b"abc")
    (tmp_path / "data" / "f.json").unlink()
    assert verify(manifest, tmp_path)[0]["status"] == "missing"


def test_real_data_files_match_the_committed_manifest():
    rows = verify(ROOT / "configs" / "data_manifest.yaml", ROOT)
    present = [r for r in rows if r["status"] != "missing"]
    assert all(r["status"] == "ok" for r in present)
