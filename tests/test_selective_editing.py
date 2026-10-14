import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from interesearch.checksums import verify_checksums
from interesearch.selective_editing.method import fisher
from interesearch.selective_editing.analysis.statistics import analyze, readings
from interesearch.selective_editing.analysis.confirm import reading_table, run
from interesearch.selective_editing.data import Study, load_study
from interesearch.selective_editing.analysis.posthoc import run as run_posthoc
from interesearch.selective_editing.method.calibration import decide, decide_neurons, replay_trace
from replay_check import assert_close

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "selective_editing"
RESULTS = ROOT / "results" / "selective_editing"
LINKS = {"analysis_sha256": "confirm_analysis.json", "dev_analysis_sha256": "dev_analysis.json"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_replay_reproduces_committed_results():
    texts = run(DATA)
    others = ("posthoc.json", "extended_comparisons_analysis.json", "explainer_data.json")  # written by other commands
    assert sorted(texts) == sorted(p.name for p in RESULTS.glob("*.json") if p.name not in others)
    for name, text in texts.items():
        new, committed = json.loads(text), json.loads((RESULTS / name).read_text())
        if name == "confirm_result.json":  # each links the analyses by the sha256 of the bytes written beside it
            for key, f in LINKS.items():
                assert new.pop(key) == sha256(texts[f]), key
                assert committed.pop(key) == sha256((RESULTS / f).read_text()), key
        assert_close(new, committed, name)


def test_posthoc_replay_reproduces_committed_results():
    assert_close(run_posthoc(DATA), json.loads((RESULTS / "posthoc.json").read_text()), "posthoc.json")


def test_nonfinite_readout_is_technical_and_makes_the_analysis_incomplete():
    """readout.py: a non-finite per-document value of a matched edit is a technical state, never a declared
    "unusable". The draw is excluded, the contrast and the whole analysis are incomplete, and the reading table
    gives no verdict on it."""
    study = load_study(DATA)
    study.cells["confirm"][("de", 1)].ce_doc["lora_rank1@0.5"][5] = math.nan
    name = "de->nl|tau=0.5|vpd_atom-lora_rank1"
    a = analyze(study, "confirm")
    assert a["composition"]["claim"][name] == {
        "expected_draws": [0, 1, 2], "used_draws": [0, 2], "status": "incomplete",
        "excluded": {"1": "vpd_atom: ok; lora_rank1: non-finite readout (1 of 58 documents of confirm_nl)"}}
    assert a["status"] == "incomplete"
    assert a["incomplete_contrasts"] == [name]
    assert a["claim"][name]["status"] == "incomplete"
    assert a["composition"]["claim"]["de->nl|tau=0.5|vpd_atom-finetune_24"]["status"] == "complete"
    reads = {c: readings(study.protocol, cell.info, study.cells["dev"][c].info)
             for c, cell in study.cells["confirm"].items()}
    row = next(r for r in reading_table(a, analyze(study, "dev"), reads) if r["contrast"] == name)
    assert (row["verdict"], row["why"]) == ("not established", "status incomplete (technical)")


def test_the_original_curvature_check_passes_the_criteria_of_the_current_code():
    d = DATA / "curvature_check"
    verify_checksums(d, ("curvature_check.json",))
    rec = json.loads((d / "curvature_check.json").read_text())
    pooled = rec["pooled_z"]
    assert rec["criteria"] == fisher.CRITERIA and rec["passed"] and fisher.evaluate(rec) == []
    assert math.isclose(rec["pooled_z"], pooled, rel_tol=1e-12)


@pytest.mark.parametrize("method", ["vpd_atom", "neurons"])
@pytest.mark.parametrize("tol,max_bisect", [(0.05, 0), (0.001, 1)])
def test_calibration_replay_uses_the_saved_settings_for_both_edit_families(method, tol, max_bisect, monkeypatch):
    from interesearch.selective_editing.method import calibration

    grid = [0.25, 1.0, 2.0]
    effect = lambda a: 1.04 * a if a <= 1.0 else 1.0
    parameters = {"tol": tol, "max_bisect": max_bisect, "mono_tol": 0.05}
    ladder = [(1, effect(1.0))]
    if method == "neurons":
        scan, decision = decide_neurons(ladder, lambda k: effect, 1.0, grid, **parameters)
    else:
        scan, decision = decide(effect, 1.0, grid, **parameters)
    assert scan["monotone"]  # the final 0.04 drop is within the saved monotonicity tolerance
    assert len(scan["bisections"]) == max_bisect
    assert decision["status"] == ("matched" if max_bisect == 0 else "bracket found, tolerance not met")
    default_scan, _ = decide(effect, 1.0, grid)
    assert not default_scan["monotone"] and default_scan["bisections"] != scan["bisections"]
    recorded_scan = {k: [list(p) for p in scan[k]] for k in ("grid_evaluations", "bisections")}
    protocol = {"format": 2, "study": "selective_editing", "expected_cells": ["de:0"],
                "calibration": parameters | {"grid": grid, "grid_of_method": {method: "grid"}}}
    rec = {"ladder": ladder, "by_tau": {"1.0": {"scan": recorded_scan, "recorded": decision}}}
    study = Study(protocol, {}, {("de", 0): {"methods": {method: rec}}})
    monkeypatch.setattr(calibration, "TOL", 0.5)
    monkeypatch.setattr(calibration, "MAX_BISECT", 20)
    trace = replay_trace(study)
    assert trace["n_agree"] == trace["n_decisions"] == 1
    assert trace["decisions"][0]["same_points"]


def test_only_recognized_legacy_records_receive_missing_defaults(monkeypatch):
    from interesearch.selective_editing.method import calibration
    from interesearch.selective_editing.analysis import statistics

    monkeypatch.setattr(calibration, "TOL", 0.7)
    monkeypatch.setattr(statistics, "N_BOOT", 7)
    old = {"format": 1, "study": "selective_editing", "calibration": {"tol": 0.03, "max_bisect": 2}}
    assert calibration.calibration_parameters(old) == {"tol": 0.03, "max_bisect": 2, "mono_tol": 0.02}
    assert statistics.bootstrap_parameters(old) == {"n_boot": 2000, "seed": 0, "level": 0.95}
    with pytest.raises(ValueError, match="incomplete calibration"):
        calibration.calibration_parameters(old | {"format": 2})
    with pytest.raises(ValueError, match="incomplete bootstrap"):
        statistics.bootstrap_parameters(old | {"format": 2})
    with pytest.raises(ValueError, match="unrecognized"):
        statistics.bootstrap_parameters(old | {"study": "unknown"})


def test_main_analysis_uses_recorded_bootstrap_settings():
    study = load_study(DATA)
    parameters = {"resamples": 23, "seed": 19, "level": 0.5}
    low = analyze(dataclasses.replace(study, protocol=study.protocol | {"bootstrap": parameters}), "dev")
    high = analyze(dataclasses.replace(study, protocol=study.protocol | {
        "bootstrap": parameters | {"level": 0.95}}), "dev")
    for family in ("claim", "references"):
        assert low[family]["_critical_value"] < high[family]["_critical_value"]
        name = next(k for k in low[family] if not k.startswith("_"))
        assert low[family][name]["estimate"] == high[family][name]["estimate"]
        assert low[family][name]["se"] == high[family][name]["se"]


def test_posthoc_accepts_explicit_settings_and_records_them():
    config = {"resamples": 13, "seed": 29, "level": 0.6, "band": 0.0}
    res = run_posthoc(DATA, config)
    assert res["posthoc_config"] == config
    for row in res["dose_agreement"].values():
        assert row["seed"] == 29 and row["n_boot"] == 13
        assert row["summary"]["band"] == 0.0 and row["summary"]["target_gap_within_band"] == 0
    from interesearch.selective_editing.analysis.posthoc import joint_intervals
    matrix = np.array([[0.1, 0.4, 0.8, -0.2], [0.3, 0.5, -0.1, 0.2]])
    low = joint_intervals(matrix, ["a", "b"], seed=29, n_boot=13, level=0.5)
    high = joint_intervals(matrix, ["a", "b"], seed=29, n_boot=13, level=0.95)
    assert low["critical"] < high["critical"]
