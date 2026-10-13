import dataclasses
import json
import math

import pytest
import torch

from interesearch.model import LlamaSimpleMLP, LlamaSimpleMLPConfig
from interesearch.selective_editing import extended_comparisons_run as A
from interesearch.selective_editing import run as R
from interesearch.selective_editing.access import AccessError, write_freeze
from interesearch.selective_editing.extended_comparisons import load
from interesearch.selective_editing.comparators import svd_dictionary, whitened_atom, whitened_dictionary
from test_selective_model import V, gate_record, tokens, toy_dictionary, write_selection


def test_svd_atoms_sum_to_the_weight_in_the_dictionary_orientation():
    W = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    dic = svd_dictionary({"m": W})
    atoms = dic["m"]
    assert dic.name == "svd" and tuple(atoms.U.shape) == (4, 6) and tuple(atoms.V.shape) == (4, 4)
    assert torch.allclose(sum(atoms.atom(c) for c in range(atoms.C)), W.double(), atol=1e-12)
    norms = [float(atoms.atom(c).norm()) for c in range(atoms.C)]
    assert norms == sorted(norms, reverse=True)  # the singular values, largest first


def test_the_whitened_atom_maximizes_the_damped_whitened_gradient_and_takes_the_given_norm():
    g = torch.Generator().manual_seed(1)
    G = torch.randn(5, 3, generator=g, dtype=torch.float64)
    X, Y = torch.randn(3, 3, generator=g, dtype=torch.float64), torch.randn(5, 5, generator=g, dtype=torch.float64)
    A_, B_ = X @ X.T, Y @ Y.T
    gamma = 1e-2
    u, v, best = whitened_atom(G, A_, B_, gamma)
    pi = math.sqrt((float(A_.trace()) / 3) / (float(B_.trace()) / 5))
    Ad, Bd = A_ + pi * gamma ** 0.5 * torch.eye(3, dtype=torch.float64), B_ + gamma ** 0.5 / pi * torch.eye(5, dtype=torch.float64)

    def objective(u, v):
        return float((u @ G @ v) ** 2 / ((u @ Bd @ u) * (v @ Ad @ v)))

    assert math.isclose(objective(u, v), best, rel_tol=1e-10)
    assert all(objective(torch.randn(5, generator=g, dtype=torch.float64), torch.randn(3, generator=g,
                                                                                         dtype=torch.float64)) <= best
               for _ in range(200))
    dic = whitened_dictionary({"m": G}, {"m": (A_, B_)}, gamma, 0.7, "kfac_es_d0")
    assert dic.name == "kfac_es_d0" and math.isclose(float(dic["m"].atom(0).norm()), 0.7, rel_tol=1e-12)


def test_the_whitened_selection_borrows_the_main_norm_ridge_floor_and_grid(tmp_path):
    torch.manual_seed(0)
    cfg = LlamaSimpleMLPConfig(block_size=32, vocab_size=V, n_layer=4, n_head=2, n_embd=32, n_intermediate=64,
                               n_ctx=32, n_key_value_heads=2, flash_attention=False)
    model = LlamaSimpleMLP(cfg).eval().requires_grad_(False)
    main, out = tmp_path / "main", tmp_path / "extended_comparisons"
    R.write_json(R.select_path(main, "es", 0, "vpd"), {"status": "feasible", "accepted": {"norm": 0.5}})
    R.write_json(main / "q" / "vpd.json", {"eps": 1e-3})

    class Stub:
        ctx = R.Context(out, tmp_path, model, toy_dictionary(model), torch.device("cpu"),
                        gate_record(tmp_path / "gate.json"), 4)
        sizes, state_cache = R.REDUCED, {}

        def windows(self, role, lang):
            return tokens(1, 8)

        def draw(self, t, d):
            return tokens(2, 8)

    stub = Stub()
    stub.out, stub.main = out, main
    A.selection(stub, "es", 0, "whitened_rank1")
    rec = json.loads(R.select_path(out, "es", 0, "whitened_rank1").read_text())
    assert (rec["seed_name"], rec["eps"], rec["grid"]) == ("kfac_es_d0", 1e-3, list(A.WHITENED_GRID))
    assert rec["n_candidates"] == 4 and all(math.isclose(w["norm"], 0.5, rel_tol=1e-12) for w in rec["walk"])
    assert (out / "q" / "kfac_es_d0.pt").exists() and (out / "cache" / "kfac_factors.pt").exists()


def test_reduced_extended_comparisons_stages_run_after_a_reduced_main_run_and_export_records_the_analyzer_reads(tmp_path):
    w = tmp_path / "windows"
    write_selection(w)
    torch.manual_seed(0)
    cfg = LlamaSimpleMLPConfig(block_size=256, vocab_size=V, n_layer=4, n_head=2, n_embd=32, n_intermediate=64,
                               n_ctx=256, n_key_value_heads=2, flash_attention=False)
    model = LlamaSimpleMLP(cfg).eval().requires_grad_(False)
    model.wte.weight.mul_(10.0)
    main, out, cells = tmp_path / "main", tmp_path / "extended_comparisons", [("es", 0), ("es", 1)]
    gate = gate_record(tmp_path / "gate.json")
    dic = toy_dictionary(model)
    ctx = R.Context(main, w, model, dic, torch.device("cpu"), gate, 4)
    r = R.Runner(ctx, "prefreeze", True)
    R.prefreeze(r, cells)
    r.finish(["test"])
    actx = R.Context(out, w, model, dic, torch.device("cpu"), gate, 4)
    with pytest.raises(AccessError, match="readout roles stay closed"):  # the main study is not frozen yet
        A.prefreeze(A.Runner(actx, main, "prefreeze", True), cells)
    write_freeze(main, gate, cells, allow_incomplete=True)  # without the CI reference
    r = R.Runner(ctx, "readout", None, True)
    R.readout(r, cells)
    r.finish(["test"])

    for condition, lr in (("matrix", 1e-7), ("lora", 1e-3)):  # draw-0 choices recorded earlier (3 steps reach no tau)
        R.write_json(A.choice_path(out, "es", condition), {"target": "es", "rows": [], "chosen_lr": lr})
    r = A.Runner(actx, main, "prefreeze", True)
    A.prefreeze(r, cells)
    assert r.finish(["test"]) == 0
    for d in (0, 1):  # the main selection found no atom on this model. The check reproduces that
        sel = json.loads(R.select_path(out, "es", d, "vpd_atom").read_text())
        assert sel["reproduces_main_strict"] is True and sel["accepted"] is None
        assert json.loads(R.select_path(out, "es", d, "whitened_rank1").read_text())["status"].startswith("no main")
    train = json.loads((A.run_dir(out, "es", 1, "matrix", 1e-7, 2) / "train.json").read_text())
    assert train["config"]["seed"] == 100 + 1 + 2000 and train["seed_index"] == 2
    cal = json.loads(R.cell_path(out, "calibration", "es", 1).read_text())["methods"]
    assert list(cal) == A.methods_of_cell()
    assert cal["finetune_24_long_s2"]["learned_update"]["training_run"] == "es_draw1_finetune_24_long_lr1e-07_s2"

    binding = out / "select" / "main_study.json"  # the additional comparisons stay bound to the main freeze they started from
    kept = binding.read_text()
    binding.write_text(json.dumps({"freeze_sha256": "0" * 64}))
    with pytest.raises(AccessError, match="not the main study run"):
        A.Runner(actx, main, "prefreeze", True)
    binding.write_text(kept)

    A.freeze(out, gate, cells, True, "")
    r = A.Runner(actx, main, "readout", None, True)
    A.readout(r, cells)
    assert r.finish(["test"]) == 0
    read = json.loads(R.cell_path(out, "readout", "es", 0).read_text())
    assert read["vpd_vs_main_readout_max_abs"] in (0.0, None)
    with pytest.raises(ValueError, match="allow_reduced"):
        load(out / "records")
    add = load(out / "records", allow_reduced=True)
    assert len(add.training) == 2 * 2 * 3  # the recorded choices close the grid: only the chosen rates, draws x seeds
    res = json.loads((out / "analysis/extended_comparisons_analysis.json").read_text())
    assert res["expected_cells"] == ["es:0", "es:1"] and set(res["composition"]) == {
        f"es->pt|tau={tau}|vpd_atom-{m}" for tau in (0.5, 1.0) for m in ("svd_atom", "whitened_rank1",
                                                                          "lora_rank1_long", "finetune_24_long")}


def test_the_extended_comparisons_refuse_another_main_study_and_main_windows_of_other_content(tmp_path):
    gate, out, role = gate_record(tmp_path / "gate.json"), tmp_path / "extended_comparisons", R.SELECTION_ROLES[0]
    R.write_json(out / R.RUN_FILE, {"windows": {role: "w1"}})
    for name, accepted in (("main_a", 1), ("main_b", 2)):  # two frozen main studies with different selections
        R.write_json(tmp_path / name / R.RUN_FILE, {"windows": {role: "w1"}})
        R.write_json(R.select_path(tmp_path / name, "es", 0, "vpd"), {"accepted": accepted})
        write_freeze(tmp_path / name, gate, [("es", 0)], allow_incomplete=True)
    A.bind_main(out, tmp_path / "main_a")
    A.bind_main(out, tmp_path / "main_a")  # a later stage with the same main study
    with pytest.raises(AccessError, match="not the main study run"):
        A.bind_main(out, tmp_path / "main_b")
    R.write_json(tmp_path / "main_a" / R.RUN_FILE, {"windows": {role: "w2"}})
    with pytest.raises(AccessError, match="other windows"):
        A.bind_main(out, tmp_path / "main_a")


def test_a_rate_without_a_training_record_is_a_technical_row_not_a_choice(tmp_path, monkeypatch):
    class Stub:
        out = tmp_path
        ctx = type("C", (), {"model": None, "device": "cpu", "bs": 4})

        def windows(self, role, lang):
            return torch.zeros(4, 9, dtype=torch.long)

    monkeypatch.setattr(A, "LEARNING_RATES", {"lora": (1e-3, 3e-3)})
    stub = Stub()
    stub.sizes = dataclasses.replace(R.REDUCED, n_rates=None)
    (A.run_dir(tmp_path, "es", 0, "lora", 3e-3, 0)).mkdir(parents=True)
    (A.run_dir(tmp_path, "es", 0, "lora", 3e-3, 0) / "train.json").write_text("{}")
    monkeypatch.setattr(A, "load_update", lambda d: {})
    monkeypatch.setattr(A, "choose_lr", lambda *a, **k: {"rows": [{"lr": 3e-3, "status": "matched", "scale": 1.0,
                                                                   "achieved": 1.0, "en_kl": 0.1}]})
    A.choose(stub, "es", "lora")
    rec = json.loads(A.choice_path(tmp_path, "es", "lora").read_text())
    assert rec["rows"][0] == {"lr": 1e-3, "status": A.NO_RECORD, "technical_failure": True}
    assert rec["chosen_lr"] == 3e-3
