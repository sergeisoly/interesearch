from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from interesearch.checksums import verify_checksums
from interesearch.recipes import load_recipe
from interesearch.repair_transfer.analysis.recovery import analyze, analyze_target, replay_recipe
from interesearch.repair_transfer.analysis.costs import donor_cost
from interesearch.repair_transfer.data import load_block, load_donor
from interesearch.repair_transfer.method.finetune import FTConfig, select_checkpoint
from interesearch.repair_transfer.method.selection import select
from replay_check import assert_close

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "repair_transfer"


@pytest.fixture
def recipe():
    return load_recipe("interesearch.repair_transfer")


def test_replay_reproduces_committed_results(recipe):
    committed = json.loads((ROOT / "results/repair_transfer/analysis.json").read_text())
    assert_close(json.loads(json.dumps(analyze(DATA, recipe))), committed)


def test_nonfinite_readout_makes_the_cell_and_primary_not_assessable(recipe):
    main, nested = (load_block(DATA / "primary_target" / b) for b in ("main", "nested"))
    key = next(n for n in main.readout if n.startswith("donor_init@2K::"))
    main.readout[key]["nl"]["ce"][0] = math.nan
    donor = donor_cost(load_donor(DATA / "donor"), recipe["costs"]["selection_tokens"])
    res = analyze_target("primary_target", main, nested, donor, recipe)
    assert res["cells"]["donor_init@2K"]["status"] == "technical_invalid"
    assert res["contrasts"]["donor_init@2K - strengths_best@2K"]["status"] == "technical_invalid"
    assert res["primary"]["reading"].startswith("not assessable: donor_init@2K is technical_invalid")
    assert res["cells"]["fresh@2K"]["status"] == "ok"


def test_each_target_fine_tune_record_reselects_its_checkpoint_and_names_the_update_its_blocks_used():
    for target in ("primary_target", "replication_target"):
        d = DATA / target / "finetune"
        verify_checksums(d, ("run.json", "train_log.jsonl", "val.json"))
        run, val = (json.loads((d / f).read_text()) for f in ("run.json", "val.json"))
        chosen = select_checkpoint(val["by_step"], FTConfig.from_dict(run["config"]))["selected_step"]
        assert chosen == val["saved_step"] == run["saved"]["saved_step"]
        for block in ("main", "nested"):
            recorded = json.loads((DATA / target / block / "block.json").read_text())["recorded"]
            assert recorded["finetune_update_sha256"] == run["update_sha256"]



def test_saved_selection_recipe_replays_after_current_profile_changes(recipe, tmp_path):
    """A changed YAML threshold changes the decision, and the recorded threshold survives later defaults."""
    requested = deepcopy(recipe)
    requested["selection"]["phi_max"] = 0.03
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(requested, sort_keys=False))
    saved = load_recipe("interesearch.repair_transfer", path)
    info = {"format": 2, "study": "repair_transfer", "target": "primary_target", "block": "main",
            "run": {"reduced": True}, "protocol": {"resolved_recipe": saved}}

    def values(x):
        return torch.tensor([x], dtype=torch.float64)

    per = {
        "base": {"nl": {"ce": values(1.0)}, "de": {"ce": values(1.0), "kl_ref": values(1.0)}},
        "finetuned": {"nl": {"ce": values(2.0), "kl_base": values(1.0)}, "de": {"ce": values(2.0)}},
        "candidate": {"nl": {"ce": values(1.5), "kl_base": values(0.5)},
                      "de": {"ce": values(2.0), "kl_ref": values(0.04)}},
    }
    for language in recipe["selection"]["retain_languages"]:
        per["finetuned"][language] = {"ce": values(1.0)}
        per["candidate"][language] = {"ce": values(1.005)}
    args = (per, {"candidate": 0.5}, ["candidate"], torch.arange(1))
    assert select(*args, recipe["selection"])["selected"] == "candidate"
    assert select(*args, saved["selection"])["selected"] is None
    assert select(*args, replay_recipe(info, recipe)["selection"])["selected"] is None


@pytest.mark.parametrize("info", [
    {"format": 1, "study": "repair_transfer", "target": "primary_target", "block": "main", "run": {}},
    {"format": 2, "study": "repair_transfer", "target": "primary_target", "block": "main"},
    {"format": 2, "study": "repair_transfer", "record": "donor_repair_fit"},
])
def test_new_export_without_snapshot_cannot_inherit_current_defaults(recipe, info):
    with pytest.raises(ValueError, match="no resolved recipe"):
        replay_recipe(info, recipe)


def test_incomplete_saved_recipe_is_not_filled_from_published_defaults(recipe):
    saved = deepcopy(recipe)
    del saved["selection"]["top_retain"]
    info = {"format": 2, "study": "repair_transfer", "protocol": {"resolved_recipe": saved}}
    with pytest.raises(ValueError, match=r"missing selection\.top_retain"):
        replay_recipe(info, recipe)


def test_recipe_values_control_full_run_permission(recipe, tmp_path, monkeypatch):
    from interesearch.repair_transfer.stages import run

    path = tmp_path / "different-name.yaml"
    path.write_text("# Same values, another file.\n" + yaml.safe_dump(recipe, sort_keys=False))
    captured = []
    monkeypatch.setattr(run, "cmd_block", lambda args: captured.append(args) or 0)
    args = ["block", "--target", "primary_target", "--block", "main", "--recipe", str(path),
            "--assets", "unused", "--finetune", "unused", "--donor-repair", "unused", "--fit", "unused",
            "--selection", "unused", "--readout", "unused", "--out", "unused"]
    assert run.main(args) == 0
    assert not captured[-1].reduced and captured[-1].deviations == {}
    recipe["selection"]["phi_max"] = 0.03
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    with pytest.raises(SystemExit):
        run.main(args)
    args[args.index("primary_target")] = "standin"
    assert run.main(args) == 0
    assert captured[-1].reduced and "selection" in captured[-1].deviations
