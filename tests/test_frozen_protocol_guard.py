"""Historical asset tests must not relax real evaluation's source freeze."""
import hashlib
import json

import pytest

from scripts import build_online_holdout50, evaluate_representative50


@pytest.mark.parametrize("module", [build_online_holdout50, evaluate_representative50])
def test_actual_evaluation_still_rejects_source_drift(tmp_path, monkeypatch, module):
    source = tmp_path / "source.py"
    source.write_text("original", encoding="utf-8")
    manifest = {"hashes": {}, "source_hashes": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
    (tmp_path / "FROZEN.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_representative50, "SUITE", tmp_path)
    def verify(**kw):
        return module.verify(tmp_path, **kw) if module is build_online_holdout50 else module.verify(**kw)
    assert verify() == manifest
    source.write_text("changed", encoding="utf-8")
    assert verify(check_protocol=False) == manifest
    with pytest.raises(RuntimeError, match="[Pp]rotocol changed"):
        verify()
