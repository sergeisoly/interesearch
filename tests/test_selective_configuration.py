"""Scientific settings are explicit, snapshotted before model loading, and bound on resume."""
import copy
import json
from pathlib import Path

import pytest

from interesearch.selective_editing.configuration import resolve, saved_recipe, validate
from interesearch.selective_editing.data import load_protocol
from interesearch.selective_editing.stages import main


def test_stage_saves_recipe_before_loading_a_model(tmp_path, monkeypatch):
    def before_model(args):
        run = json.loads((args.out / "RUN.json").read_text())
        invocation = json.loads(args.invocation.read_text())
        assert run["recipe"] == args.recipe == invocation["recipe"]
        assert run["recipe"]["training"]["checkpoints"] == [50, 100, 150, 200]
        assert invocation["code"]
        raise RuntimeError("model boundary")
    monkeypatch.setattr(main, "context", before_model)
    with pytest.raises(RuntimeError, match="model boundary"):
        main.main(["prefreeze", "assets", "--out", str(tmp_path), "--gate", str(tmp_path / "gate.json"), "--no-ci"])


def test_saved_recipe_has_priority_and_explicit_changes_are_refused(tmp_path):
    published = resolve()
    saved = copy.deepcopy(published)
    saved["calibration"]["tol"] = 0.3
    (tmp_path / "RUN.json").write_text(json.dumps({"format": 2, "recipe": saved}))
    assert saved_recipe(tmp_path, published, False) == saved
    with pytest.raises(ValueError, match="scientific recipe differs"):
        saved_recipe(tmp_path, published, True)
    (tmp_path / "RUN.json").write_text(json.dumps({"format": 2}))
    with pytest.raises(ValueError, match="complete recipe snapshot"):
        saved_recipe(tmp_path, published, False)


def test_new_export_requires_the_complete_snapshot(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps({"format": 2, "recipe": {"calibration": {"tol": 0.2}}}))
    with pytest.raises(ValueError, match="complete recipe snapshot"):
        load_protocol(tmp_path, False)


@pytest.mark.parametrize("section,key,value", [("extended", "validation_every", 0),
    ("selection", "max_walk", 0), ("extended", "seeds", [0, -1]), ("readout", "atlas_top", 65)])
def test_invalid_scientific_domains_fail_before_computation(section, key, value):
    recipe = resolve()
    recipe[section][key] = value
    with pytest.raises(ValueError):
        validate(recipe)
