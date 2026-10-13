import hashlib
import json
import math
from pathlib import Path

from interesearch.checksums import verify_checksums
from interesearch.selective_editing import fisher
from interesearch.selective_editing.analysis import analyze, readings
from interesearch.selective_editing.confirm import reading_table, run
from interesearch.selective_editing.data import load_study
from interesearch.selective_editing.posthoc import run as run_posthoc
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
