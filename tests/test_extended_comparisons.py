import dataclasses
import json
from pathlib import Path

from interesearch.selective_editing.analysis.extended import analyze, label, load
from interesearch.selective_editing.data import load_study
from replay_check import assert_close

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "selective_editing"
COMMITTED = ROOT / "results" / "selective_editing" / "extended_comparisons_analysis.json"


def replay(add=None) -> dict:
    return json.loads(json.dumps(analyze(add or load(DATA / "extended_comparisons"), load_study(DATA))))


def test_replay_reproduces_committed_results():
    assert_close(replay(), json.loads(COMMITTED.read_text()), COMMITTED.name)


def test_svd_contrasts_have_no_estimate_and_no_label():
    res = replay()
    svd = sorted(k for k in res["composition"] if k.endswith("-svd_atom"))
    assert res["contrasts_absent"] == svd and len(svd) == 6
    assert not set(svd) & (set(res["labels"]) | set(res["all_runs"]))
    for k, row in res["methods"].items():
        if k.endswith("|svd_atom"):  # every run matched but failed the guard, or never reached the dose
            assert row["used"] == 0 and all(x.endswith(("matched (guard)", "not reached on the evaluated grid"))
                                            for x in row["excluded"])
    stable = {k.split("->")[0]: v["comparator_runs_stable"] for k, v in res["composition"].items()
              if k.endswith("|tau=1.0|vpd_atom-lora_rank1_long")}
    assert stable == {"de": 8, "es": 6, "fr": 6}


def test_no_stable_run_is_absent_and_changes_both_attributes():
    add = load(DATA / "extended_comparisons")
    training = {r: rec | {"convergence": rec["convergence"] | {"unstable": rec["target"] == "fr"}}
                for r, rec in add.training.items()}
    res = replay(dataclasses.replace(add, training=training))
    fr = [k for k in res["labels_stable_runs"] if k.startswith("fr->")]
    assert len(fr) == 4 and all(res["labels_stable_runs"][k] == "absent (no stable run)" for k in fr)
    assert all(res["stable_changes_a_label"][k] == ["direction", "within_band"] for k in fr)
    assert fr[0] in res["labels"] and fr[0] not in res["stable_runs"]


def test_direction_and_band_are_independent():
    assert label({"lo": -0.015, "hi": -0.005}, 0.02) == {"direction": "below 0", "within_band": True}
    assert label({"lo": -0.03, "hi": 0.01}, 0.02) == {"direction": "includes 0", "within_band": False}


def test_both_extended_families_use_the_recorded_bootstrap():
    add = load(DATA / "extended_comparisons")
    parameters = {"resamples": 23, "seed": 19, "level": 0.5}
    low = replay(dataclasses.replace(add, protocol=add.protocol | {"bootstrap": parameters}))
    high = replay(dataclasses.replace(add, protocol=add.protocol | {"bootstrap": parameters | {"level": 0.95}}))
    for family in ("all_runs", "stable_runs"):
        assert low[family]["_critical_value"] < high[family]["_critical_value"]
        name = next(k for k in low[family] if not k.startswith("_"))
        assert low[family][name]["estimate"] == high[family][name]["estimate"]
        assert low[family][name]["se"] == high[family][name]["se"]
