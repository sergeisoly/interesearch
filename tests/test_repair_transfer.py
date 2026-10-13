import json
import math
from pathlib import Path

from interesearch.checksums import verify_checksums
from interesearch.repair_transfer.analysis import analyze, analyze_target
from interesearch.repair_transfer.costs import donor_cost
from interesearch.repair_transfer.data import load_block, load_donor
from interesearch.repair_transfer.finetune import FTConfig, select_checkpoint
from replay_check import assert_close

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "repair_transfer"


def test_replay_reproduces_committed_results():
    committed = json.loads((ROOT / "results/repair_transfer/analysis.json").read_text())
    assert_close(json.loads(json.dumps(analyze(DATA))), committed)


def test_nonfinite_readout_makes_the_cell_and_primary_not_assessable():
    main, nested = (load_block(DATA / "primary_target" / b) for b in ("main", "nested"))
    key = next(n for n in main.readout if n.startswith("donor_init@2K::"))
    main.readout[key]["nl"]["ce"][0] = math.nan
    res = analyze_target("primary_target", main, nested, donor_cost(load_donor(DATA / "donor")))
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
