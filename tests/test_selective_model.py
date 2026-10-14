import dataclasses
import json
import math
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

import pytest
import torch
from safetensors.torch import save_file

from interesearch.assets import MODULES
from interesearch.checksums import sha256_file
from interesearch.model import LlamaSimpleMLP, LlamaSimpleMLPConfig
from interesearch.selective_editing.method import fisher
from interesearch.selective_editing.stages import readout as RO
from interesearch.selective_editing.stages import main as R
from interesearch.selective_editing.stages import support as RS
from interesearch.selective_editing.method import selector as S
from interesearch.selective_editing.method import runtime as RT
from interesearch.selective_editing.stages.access import Access, AccessError, AlreadyOpened, load_confirm, load_windows, open_confirm, technical_gaps, verify_freeze, window_identity, write_freeze
from interesearch.selective_editing.method.calibration import TECHNICAL
from interesearch.selective_editing.data import load_dev
from interesearch.selective_editing.configuration import resolve
from interesearch.selective_editing.method.dose import GUARD_MAX
from interesearch.selective_editing.method.dictionary import Dictionary, module_generator, randomized_dictionary, tensor_digest
from interesearch.selective_editing.method.neurons import neuron_ranking
from interesearch.selective_editing.method import references
from interesearch.selective_editing.method.references import omp_top1

V, C = 64, 5


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = LlamaSimpleMLPConfig(block_size=32, vocab_size=V, n_layer=4, n_head=2, n_embd=32, n_intermediate=64,
                               n_ctx=32, n_key_value_heads=2, flash_attention=False)
    return LlamaSimpleMLP(cfg).eval().requires_grad_(False)


def tokens(seed, n=4, t=9):
    return torch.randint(0, V, (n, t), generator=torch.Generator().manual_seed(seed))


def toy_dictionary(model, seed=0, modules=MODULES):
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for m in modules:
        d_out, d_in = model.get_submodule(m).weight.shape
        tensors[m] = (torch.randn(C, d_out, generator=g), torch.randn(d_in, C, generator=g))
    return Dictionary.from_tensors(tensors, {"tensor_digest": f"toy{seed}"})


def gate_record(path, **change):
    rec = {"passed": True, "failures": [], "code": RT.code_identity(), "fisher_k": S.FISHER_K, "n_windows": 8,
           "window": 257, "k_big": 32, "dtype": "float64", "device": "cpu",
           "provenance": RT.runtime_identity("cpu"),
           "selection_settings": {"fisher_k": S.FISHER_K, "eps_percentile": S.EPS_PERCENTILE,
                                  "null_seed": S.NULL_SEED[0]}} | change
    path.write_text(json.dumps(rec))
    return path


# --------------------------------------------------------------------------- tensor digest
def write_safetensors(path, tensors, metadata, reverse=False):
    """A safetensors file with the tensors laid out in the given order (``save_file`` picks its own layout)."""
    names = sorted(tensors, reverse=reverse)
    hdr, blobs, off = {"__metadata__": metadata}, [], 0
    for k in names:
        b = tensors[k].contiguous().numpy().tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(tensors[k].shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_tensor_digest_ignores_the_header_layout_and_sees_every_tensor_byte_and_the_metadata(tmp_path):
    g = torch.Generator().manual_seed(0)
    t = {"a.U": torch.randn(3, 4, generator=g), "a.V": torch.randn(4, 3, generator=g),
         "b.U": torch.randn(2, 5, generator=g)}
    meta = {"source": "x.pth", "source_sha256": "0" * 64}
    save_file(t, tmp_path / "lib.safetensors", metadata=meta)
    write_safetensors(tmp_path / "fwd.safetensors", t, meta)
    write_safetensors(tmp_path / "rev.safetensors", t, dict(reversed(meta.items())), reverse=True)
    ref = tensor_digest(tmp_path / "lib.safetensors")
    assert (tmp_path / "fwd.safetensors").read_bytes() != (tmp_path / "rev.safetensors").read_bytes()
    assert tensor_digest(tmp_path / "fwd.safetensors") == tensor_digest(tmp_path / "rev.safetensors") == ref
    flipped = {**t, "b.U": t["b.U"].clone()}
    flipped["b.U"].view(torch.int32)[1, 2] ^= 1  # one bit of one float
    changes = [(flipped, meta), ({**t, "a.U": t["a.U"].reshape(4, 3)}, meta),
               ({("c.U" if k == "b.U" else k): v for k, v in t.items()}, meta), (t, {**meta, "source": "y.pth"})]
    for i, (tt, mm) in enumerate(changes):
        write_safetensors(tmp_path / f"c{i}.safetensors", tt, mm)
        assert tensor_digest(tmp_path / f"c{i}.safetensors") != ref, i


# --------------------------------------------------------------------------- ranking and walk order
def test_ranking_is_score_descending_then_module_order_then_index_and_skips_unusable_atoms():
    modules = ["m1", "m0"]  # module order is the given order, not the name order
    score = {"m0": torch.tensor([2.0, 5.0, -1.0, 2.0]), "m1": torch.tensor([2.0, 0.0, 5.0])}
    assert S.ranking(score, modules) == [("m1", 2, 5.0), ("m0", 1, 5.0), ("m1", 0, 2.0), ("m0", 0, 2.0),
                                         ("m0", 3, 2.0), ("m1", 1, 0.0)]
    assert S.ranking(score, modules, top=2) == [("m1", 2, 5.0), ("m0", 1, 5.0)]


def test_walk_order_uses_the_larger_q_and_keeps_the_ranking_order_on_ties(model):
    m0, m1 = MODULES[0], MODULES[5]
    tensors = {}
    for m in (m0, m1):  # unit atoms: ||A_c|| = 1, so every value below is exact
        d_out, d_in = model.get_submodule(m).weight.shape
        tensors[m] = (torch.eye(C, d_out), torch.eye(d_in, C))
    dic = Dictionary.from_tensors(tensors, {})
    eps = 0.5
    q = {m: torch.ones(C, dtype=torch.float64) for m in (m0, m1)}
    q_check = {m: torch.ones(C, dtype=torch.float64) for m in (m0, m1)}
    q_check[m1][3] = 4.0  # the second pool sees a higher curvature for this atom
    g = {m: torch.full((C,), 3.0, dtype=torch.float64) for m in (m0, m1)}  # 9 / 1.5 = 6 for every other atom
    ranked = [(m1, 3, 9.0), (m0, 4, 8.0), (m1, 0, 7.0), (m0, 1, 6.0), (m0, 2, 1.0)]
    short, order = S.shortlist_order(dic, g, q, q_check, eps, ranked, 4)
    assert short == ranked[:4]
    assert order == [(m0, 4, 6.0), (m1, 0, 6.0), (m0, 1, 6.0), (m1, 3, 2.0)]
    g[m1][3] = 6.0  # 36 / 4.5 = 8: first again
    assert S.shortlist_order(dic, g, q, q_check, eps, ranked, 4)[1][0] == (m1, 3, 8.0)


# --------------------------------------------------------------------------- OMP top-1
def test_omp_top1_takes_the_first_module_in_update_order_on_a_tie_and_the_first_index_within(model):
    dic = toy_dictionary(model, modules=[MODULES[0], MODULES[1]])
    a, b = MODULES[0], MODULES[1]  # q_proj and k_proj have the same shape: give them the same atoms
    dic.modules[b].U, dic.modules[b].V = dic[a].U.clone(), dic[a].V.clone()
    dic.modules[b]._cache.clear()
    D = -3.0 * dic[a].atom(2)
    res = omp_top1(dic, {b: D, a: D.clone()})
    assert (res["module"], res["index"], res["sign"]) == (b, 2, -1.0)
    res = omp_top1(dic, {a: D, b: D.clone()})
    assert (res["module"], res["index"], res["label_key"]) == (a, 2, "0.attn.q:2")
    assert res["energy_k1"] == pytest.approx(0.5, rel=1e-12)  # one of two equal modules explained exactly
    assert omp_top1(dic, {a: D})["energy_k1"] == pytest.approx(1.0, rel=1e-12)
    dic.modules[a].U[3] = dic[a].U[2]  # a duplicate atom later in the same module does not win
    dic.modules[a].V[:, 3] = dic[a].V[:, 2]
    dic.modules[a]._cache.clear()
    assert omp_top1(dic, {a: D})["index"] == 2
    dic.modules[a].usable[2] = False  # an unusable atom is never picked
    assert omp_top1(dic, {a: D})["index"] == 3
    assert [r["index"] for r in res["top10"][:2]] == [2, 2]


# --------------------------------------------------------------------------- gate
def test_selection_and_q_refuse_without_a_full_size_passed_check_of_the_current_code(model, tmp_path):
    dic, retain = toy_dictionary(model), tokens(1)
    gate = tmp_path / "check.json"
    with pytest.raises(RT.GateError, match="no curvature-check record"):
        S.q_record(tmp_path / "q.pt", model, dic, retain, retain, "cpu", gate)
    for change, why in [({"n_windows": 2, "k_big": 4}, "too small"), ({"k_big": 31}, "too small"),
                        ({"window": 9}, "too small"),
                        ({"dtype": "float32"}, "not float64"), ({"passed": False}, "did not pass"),
                        ({"code": {**RT.code_identity(), "fisher.py": "0" * 64}}, "other code"),
                        ({"fisher_k": 2}, "K=2")]:
        with pytest.raises(RT.GateError, match=why):
            S.q_record(tmp_path / "q.pt", model, dic, retain, retain, "cpu", gate_record(gate, **change))
        with pytest.raises(RT.GateError, match=why):
            S.select(model, dic, {}, tokens(2), "cpu", gate)
    assert not (tmp_path / "q.pt").exists()
    RT.require_gate(gate_record(gate))
    other_gpu = RT.runtime_identity("cpu") | {"multi_processor_count": 132}  # production guard, not a gate criterion
    with pytest.raises(RT.GateError, match="multi_processor_count"):
        S.select(model, dic, {}, tokens(2), "cpu", gate_record(gate, provenance=other_gpu))


# --------------------------------------------------------------------------- stored records
@pytest.mark.parametrize("field,value", [("eps_percentile", 2.0), ("null_seeds", [109, 102, 103])])
def test_curvature_gate_binds_percentile_and_first_randomized_seed(tmp_path, field, value):
    gate = gate_record(tmp_path / "check.json")
    recipe = resolve()
    recipe["selection"][field] = value
    with pytest.raises(RT.GateError, match="selection settings"):
        # No model computation is reached: these checks precede q-cache access and selection.
        S.q_record(tmp_path / "q.pt", None, None, None, None, "cpu", gate, recipe=recipe)
    with pytest.raises(RT.GateError, match="selection settings"):
        S.select(None, None, {}, None, "cpu", gate, recipe=recipe)
    saved = json.loads(gate.read_text())
    saved.pop("selection_settings")
    gate.write_text(json.dumps(saved))
    with pytest.raises(RT.GateError, match="missing selection settings"):
        RT.require_gate(gate)


def test_stored_records_are_reused_only_under_equal_provenance(model, tmp_path):
    dic, retain, qcheck, target = toy_dictionary(model), tokens(1), tokens(3), tokens(2)
    gate = gate_record(tmp_path / "check.json")
    q = S.q_record(tmp_path / "q.pt", model, dic, retain, qcheck, "cpu", gate, bs=2)
    assert q["provenance"]["batch_size"] == 2 and q["provenance"]["device"] == "cpu"
    assert q["provenance"]["torch"] == str(torch.__version__)
    again = S.q_record(tmp_path / "q.pt", model, dic, retain, qcheck, "cpu", gate, bs=2)
    assert all(torch.equal(again["q"][m], q["q"][m]) for m in MODULES)
    before = (tmp_path / "q.pt").read_bytes()
    for kw in [dict(bs=3), dict(qcheck=tokens(4)), dict(dictionary=toy_dictionary(model, seed=1)),
               dict(dictionary=randomized_dictionary(dic, 101))]:
        args = dict(model=model, dictionary=dic, retain=retain, qcheck=qcheck, device="cpu", gate=gate, bs=2) | kw
        with pytest.raises(RT.CacheMismatch):
            S.q_record(tmp_path / "q.pt", **args)
    assert (tmp_path / "q.pt").read_bytes() == before
    w = model.get_submodule(MODULES[0]).weight
    w0 = w[0, 0].item()
    with torch.no_grad():
        w[0, 0] += 1.0
    with pytest.raises(RT.CacheMismatch, match="model"):
        S.q_record(tmp_path / "q.pt", model, dic, retain, qcheck, "cpu", gate, bs=2)
    with pytest.raises(RT.CacheMismatch, match="model"):
        S.select(model, dic, q, target, "cpu", gate, bs=2)
    with torch.no_grad():
        w[0, 0] = w0

    rec, _ = S.select(model, dic, q, target, "cpu", gate, bs=2, g_cache=tmp_path / "G.pt", n_short=3)
    assert rec["g_cache"]["cache"] == "computed"
    rec2, _ = S.select(model, dic, q, target, "cpu", gate, bs=2, g_cache=tmp_path / "G.pt", n_short=3)
    assert rec2["g_cache"]["cache"] == "reused" and rec2["top64"] == rec["top64"]
    rec3, _ = S.select(model, dic, q, tokens(5), "cpu", gate, bs=2, g_cache=tmp_path / "G.pt", n_short=3)
    assert rec3["g_cache"]["cache"] == "recomputed: the cached gradient differs in inputs"

    english = tokens(6)
    ranking = neuron_ranking(model, target, english, "cpu", tmp_path / "rates.pt", bs=2)
    assert set(ranking["english_provenance"]["code"]) == {"neurons.py", "model.py"}  # rates come from the forward
    with pytest.raises(RT.CacheMismatch, match="inputs"):
        neuron_ranking(model, target, tokens(7), "cpu", tmp_path / "rates.pt", bs=2)

    (tmp_path / "q.pt").unlink()  # what the refusal says to do. The summary follows the recomputed record
    S.q_record(tmp_path / "q.pt", model, dic, retain, qcheck, "cpu", gate, bs=3)
    assert json.loads((tmp_path / "q.json").read_text())["provenance"]["batch_size"] == 3


def test_q_and_the_target_gradient_are_bound_to_the_code_and_runtime_that_computed_them(model, tmp_path, monkeypatch):
    dic, retain, target = toy_dictionary(model), tokens(1), tokens(2)
    gate = gate_record(tmp_path / "check.json")
    q = S.q_record(tmp_path / "q.pt", model, dic, retain, retain, "cpu", gate, bs=2)
    S.target_gradient(model, list(MODULES), target, "cpu", 2, tmp_path / "G.pt")
    for field, value in [("device", "cuda:0"), ("torch", "0.0"), ("float32_matmul_precision", "medium"),
                         ("multi_processor_count", 132), ("model_config", "0" * 64)]:
        other = q | {"provenance": q["provenance"] | {field: value}}
        with pytest.raises(RT.CacheMismatch, match=field):
            S.select(model, dic, other, target, "cpu", gate, bs=2)
    with pytest.raises(RT.CacheMismatch, match="dictionary"):
        S.select(model, randomized_dictionary(dic, 101), q, target, "cpu", gate, bs=2)
    with pytest.raises(FloatingPointError, match="eps is inf"):
        S.select(model, dic, q | {"eps": math.inf}, target, "cpu", gate, bs=2)
    torch.backends.cudnn.allow_tf32 = True  # a fresh process: torch's default, which selection switches off
    S.select(model, dic, q, target, "cpu", gate, bs=3, n_short=1)  # its own batch size and pools are q's business
    assert torch.backends.cudnn.allow_tf32 is False

    new_code = {f: "0" * 64 for f in RT.GATE_FILES}  # the estimator changed. A fresh check of it passed
    monkeypatch.setattr(RT, "code_identity", lambda: new_code)
    gate = gate_record(tmp_path / "check2.json")
    with pytest.raises(RT.CacheMismatch, match="code"):
        S.q_record(tmp_path / "q.pt", model, dic, retain, retain, "cpu", gate, bs=2)
    with pytest.raises(RT.CacheMismatch, match="code"):
        S.select(model, dic, q, target, "cpu", gate, bs=2)
    _, info = S.target_gradient(model, list(MODULES), target, "cpu", 2, tmp_path / "G.pt")
    assert info["cache"] == "recomputed: the cached gradient differs in code"


def test_english_ci_means_are_bound_to_the_imported_param_decomp_sources(model, tmp_path, monkeypatch):
    pkg = tmp_path / "pd" / "param_decomp"
    (pkg / "models").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models" / "component_model.py").write_text("C = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path / "pd"))
    monkeypatch.delitem(sys.modules, "param_decomp", raising=False)
    english, cache = tokens(6), tmp_path / "ci_en.pt"

    def english_means():
        prov = RT.provenance(model, "cpu", 2, {"english": english}, references.ci_code(), {"ci_fn": "toy"})
        return RT.stored(cache, prov, lambda: {"mean": {}})

    english_means()
    english_means()
    (pkg / "models" / "component_model.py").write_text("C = 2\n")  # a modified checkout
    with pytest.raises(RT.CacheMismatch, match="code"):
        english_means()


def test_the_curvature_check_refuses_fewer_retain_windows_than_it_would_record(tmp_path):
    with pytest.raises(ValueError, match="needs 8 retain windows, got 4"):  # before any asset is read
        fisher.run(tmp_path / "no_assets", tokens(1, n=4), "cpu", n_windows=8, k_big=32)


def test_q_does_not_depend_on_batching(model):
    model, dic, retain = model.double(), toy_dictionary(model), tokens(1, n=5)  # float64: no batching rounding
    one = S.retain_fisher(model, dic, MODULES, retain, "cpu", K=3, seed=7, bs=1)
    full = S.retain_fisher(model, dic, MODULES, retain, "cpu", K=3, seed=7, bs=5)
    for m in MODULES:
        torch.testing.assert_close(one["q"][m], full["q"][m], rtol=1e-10, atol=0)
        torch.testing.assert_close(one["q_half"][m], full["q_half"][m], rtol=1e-10, atol=0)
    other = S.retain_fisher(model, dic, MODULES, retain, "cpu", K=3, seed=8, bs=5)
    assert not torch.equal(other["q"][MODULES[0]], full["q"][MODULES[0]])


# --------------------------------------------------------------------------- randomized control
def test_randomized_dictionary_depends_only_on_seed_and_module(model):
    dic = toy_dictionary(model)
    a, b = randomized_dictionary(dic, 101), randomized_dictionary(dic, 101)
    rev = randomized_dictionary(Dictionary(dict(reversed(dic.modules.items())), dic.excluded, dic.meta), 101)
    other = randomized_dictionary(dic, 102)
    assert a.name == "null101" and dic.name == "vpd" and a.meta["tensor_digest"] == dic.meta["tensor_digest"]
    for m in MODULES:
        for x in (b, rev):
            assert torch.equal(a[m].U, x[m].U) and torch.equal(a[m].V, x[m].V)
        assert not torch.equal(a[m].U, dic[m].U) and not torch.equal(a[m].U, other[m].U)
        torch.testing.assert_close(a[m].norms, dic[m].norms, rtol=1e-14, atol=0)
        assert torch.equal(a[m].usable, dic[m].usable)
        # a signed permutation of the coordinates: each row of U keeps its multiset of absolute values
        assert torch.equal(a[m].U.abs().sort(1).values, dic[m].U.abs().sort(1).values)
    assert module_generator(101, MODULES[0]).initial_seed() != module_generator(101, MODULES[1]).initial_seed()


# --------------------------------------------------------------------------- stage runner
def write_role(root, role, langs, n, seed, dates=True):
    """A synthetic window directory of ``role`` (random tokens)."""
    d = root / role
    d.mkdir(parents=True)
    g = np.random.default_rng(seed)
    meta = {"role": role, "languages": langs, "provenance": "synthetic"}
    meta["dates" if dates else "rows"] = [f"2000-01-{i:04d}" if dates else f"r{i}" for i in range(n)]
    (d / "windows.json").write_text(json.dumps(meta))
    for s in langs:
        np.save(d / f"{s}.npy", g.integers(0, V, (n, 257)).astype(np.int32))
    files = ["windows.json", *(f"{s}.npy" for s in langs)]
    (d / "SHA256SUMS").write_text("".join(f"{sha256_file(d / f)}  {f}\n" for f in files))
    return d


SELECTION = [("train", ["es"], 128, True), ("val", ["es"], 8, True), ("retain_en", ["en"], 196, False),
             ("qcheck_en", ["en"], 8, False), ("val_en", ["en"], 16, False)]


def write_selection(root, rows=None):
    for i, (role, langs, n, dates) in enumerate(SELECTION):
        write_role(root, role, langs, rows or n, i, dates)


def test_runner_items_keep_invocation_provenance_when_sources_change(model, tmp_path, monkeypatch):
    write_selection(tmp_path / "windows", 4)
    ctx = RS.Context(tmp_path / "run", tmp_path / "windows", model, toy_dictionary(model), torch.device("cpu"))
    with pytest.raises(AccessError, match="starts with the pre-freeze stage"):
        RS.Runner(ctx, "readout", None, True)
    r = RS.Runner(ctx, "prefreeze", True)
    done = ctx.out / "done.json"

    def nonfinite():
        raise FloatingPointError("non-finite target rise")

    assert r.item("a", None, nonfinite) == "failed"
    assert r.item("b", None, lambda: None, ["a"]) == "blocked"
    assert r.item("c", done, lambda: done.write_text("{}")) == "ok"
    assert r.item("c_again", done, lambda: 1 / 0) == "ok"  # output exists: not run
    assert r.item("d", None, lambda: None, needs=[ctx.out / "absent.json"]) == "blocked"
    r.declare("e", "no learning rate reached the upper effect point on draw 0")
    assert r.item("f", None, lambda: None, ["e"]) == "ok"  # a declared outcome blocks nothing
    with pytest.raises(ZeroDivisionError):  # anything else is a bug, not a recorded state
        r.item("g", None, lambda: 1 / 0)
    assert r.finish(["test"]) == 3
    rec = json.loads(next((ctx.out / "runs").glob("prefreeze-*.json")).read_text())
    assert rec["not_ok"] == ["a", "b", "d"] and rec["reduced"] is True and rec["code"] == RS.package_identity()
    assert [g.split(":")[0] for g in technical_gaps(ctx.out, [])] == ["item a", "item b", "item d"]
    with pytest.raises(AccessError, match="reduced run folder"):
        RS.Runner(ctx, "prefreeze", False)
    with pytest.raises(AccessError, match="stand-in"):
        RS.Runner(ctx, "readout", None, False)
    monkeypatch.setattr(RS, "package_identity", lambda: {"other.py": "0" * 64})
    assert r.item("h", None, lambda: None) == "ok"  # sources are sampled once per invocation
    resumed = RS.Runner(ctx, "prefreeze", True)
    assert resumed.code == {"other.py": "0" * 64} and r.code != resumed.code
    assert resumed.item("reuse", done, lambda: 1 / 0) == "ok"
    assert resumed.finish(["resume"]) == 0
    resumed_record = json.loads(resumed.invocation.read_text())
    assert resumed_record["items"][0]["output"] == "done.json"
    assert resumed_record["items"][0]["reused"] is True
    assert resumed.invocation != r.invocation


def test_an_interrupted_record_write_leaves_no_record_and_the_item_runs_again(model, tmp_path, monkeypatch):
    write_selection(tmp_path / "windows", 4)
    ctx = RS.Context(tmp_path / "run", tmp_path / "windows", model, toy_dictionary(model), torch.device("cpu"))
    r = RS.Runner(ctx, "prefreeze", True)
    p = RS.cell_path(ctx.out, "neurons", "es", 0)
    write_text = Path.write_text

    def interrupted(self, text, *a, **k):  # the process stops after the first byte
        write_text(self, text[:1])
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(Path, "write_text", interrupted)
        with pytest.raises(KeyboardInterrupt):
            r.item("neurons_es_draw0", p, lambda: R.write_json(p, {"ranked": []}))
    assert not p.exists()
    assert r.item("neurons_es_draw0", p, lambda: R.write_json(p, {"ranked": []})) == "ok"
    assert r.rows[-1]["reason"] == "" and json.loads(p.read_text()) == {"ranked": []}  # run again, not skipped


def test_a_run_folder_and_the_confirm_set_resume_only_on_the_same_windows_and_settings(model, tmp_path):
    w, out = tmp_path / "windows", tmp_path / "run"
    write_selection(w, 4)
    ctx = RS.Context(out, w, model, toy_dictionary(model), torch.device("cpu"))
    RS.Runner(ctx, "prefreeze", True)
    RS.Runner(ctx, "prefreeze", True)  # unchanged windows and settings: accepted
    with pytest.raises(AccessError, match="was run with"):
        RS.Runner(RS.Context(out, w, model, ctx.vpd, torch.device("cpu"), bs=8), "prefreeze", True)
    shutil.rmtree(w / "retain_en")
    write_role(w, "retain_en", ["en"], 4, 99, False)  # another valid export at the same path
    with pytest.raises(AccessError, match=r"windows of \['retain_en'\] differ"):
        RS.Runner(ctx, "prefreeze", True)
    confirm = write_role(tmp_path, "confirm", ["es"], 4, 1)
    rec = {"utc": "t", "code": {}, "freeze_sha256": "f", "confirm_windows": window_identity(confirm, "confirm")}
    open_confirm(out / "confirm", rec)
    open_confirm(out / "confirm", rec | {"utc": "later", "code": {"changed.py": "new"}}, resume=True)  # the same windows: a resume
    shutil.rmtree(confirm)
    write_role(tmp_path, "confirm", ["es"], 4, 2)
    with pytest.raises(AlreadyOpened):
        open_confirm(out / "confirm", rec | {"confirm_windows": window_identity(confirm, "confirm")}, resume=True)


def test_window_roles_are_refused_before_their_files_are_read(model, tmp_path):
    w, out = tmp_path / "windows", tmp_path / "run"
    write_role(w, "train", ["es"], 4, 1)
    write_role(w, "test", ["es"], 4, 2)
    mixed = write_role(tmp_path / "other", "train,test", ["es"], 4, 3)
    (mixed / "SHA256SUMS").write_text("0" * 64 + "  windows.json\n")  # never reached
    with pytest.raises(AccessError, match="may not be opened here"):
        load_windows(mixed, "train,test", refused=("test",))
    with pytest.raises(AccessError, match="expected 'train'"):
        load_windows(w / "test", "train")
    with pytest.raises(AccessError, match="stand-in run reads no readout role"):
        Access(w, out, standin=True).windows("test")
    a = Access(w, out)
    assert a.windows("train", ["es"]).tokens["es"].shape == (4, 257)
    with pytest.raises(AccessError, match="until the freeze"):
        a.windows("test")
    with pytest.raises(AccessError, match="only after"):
        load_confirm(mixed, out, ["es"])


def test_reduced_stages_run_in_order_on_stand_in_sets_and_open_the_confirm_set_once(tmp_path):
    w = tmp_path / "windows"
    write_selection(w)
    torch.manual_seed(0)
    cfg = LlamaSimpleMLPConfig(block_size=256, vocab_size=V, n_layer=4, n_head=2, n_embd=32, n_intermediate=64,
                               n_ctx=256, n_key_value_heads=2, flash_attention=False)
    model = LlamaSimpleMLP(cfg).eval().requires_grad_(False)
    model.wte.weight.mul_(10.0)  # sharp enough for the neuron ablation to reach both effect points
    out, cells = tmp_path / "run", [("es", 0), ("es", 1)]
    gate = gate_record(tmp_path / "gate.json")
    ctx = RS.Context(out, w, model, toy_dictionary(model), torch.device("cpu"), gate, 4, recipe=resolve())
    # a draw-0 rate choice recorded earlier: the later draw trains with it
    R.write_json(RS.choice_path(out, "es", "matrix"), {"target": "es", "condition": "matrix", "chosen_lr": 1e-6})

    r = RS.Runner(ctx, "prefreeze", True)
    R.prefreeze(r, cells)
    assert r.finish(["test"]) == 0
    assert r.state["lr_choice_es_matrix"] == "ok" and r.state["finetune_es_draw1_lora"] == "declared"
    train1 = json.loads((RS.run_dir(out, "es", 1, "matrix", 1e-6) / "train.json").read_text())
    assert (train1["lr"], train1["draw"]) == (1e-6, 1)
    assert [RS.select_path(out, "es", d, RS.null_name(d)).exists() for d in (0, 1)] == [True, True]
    assert RS.null_name(0) == "null101" and RS.null_name(1) == "null102"
    cal = json.loads(RS.cell_path(out, "calibration", "es", 1).read_text())["methods"]
    assert cal["ci_difference"]["technical_failure"] is True  # not run: a technical gap
    assert cal["lora_rank1"]["technical_failure"] is False and not cal["lora_rank1"]["by_tau"]  # declared
    assert cal["finetune_24"]["learned_update"]["learning_rate"] == 1e-6
    assert {row["recorded"]["status"] for row in cal["neurons"]["by_tau"].values()} == {"matched"}

    def ci(target, english, cache, dst):  # a stand-in for the reference env
        dst.write_text(json.dumps({"module": MODULES[0], "index": 0, "sign": 1.0}))

    r = RS.Runner(dataclasses.replace(ctx, ci=ci), "prefreeze", True)
    R.prefreeze(r, [("es", 1)])  # the CI reference added on a rerun: its calibration is made again
    assert r.finish(["test"]) == 0 and r.rows[-1]["item"] == "calibration_es_draw1" and r.rows[-1]["reason"] == ""
    cal = json.loads(RS.cell_path(out, "calibration", "es", 1).read_text())["methods"]
    assert "technical_failure" not in cal["ci_difference"] and cal["ci_difference"]["by_tau"]
    with pytest.raises(AccessError, match="1 technical gap"):  # es draw 0 still lacks the CI reference
        write_freeze(out, gate, cells)
    assert write_freeze(out, gate, cells, allow_incomplete=True)["allow_incomplete"] is True
    with pytest.raises(AccessError, match=r"exists\. Selection, training and calibration are closed"):
        R.prefreeze(RS.Runner(ctx, "prefreeze", True), cells)

    r = RS.Runner(ctx, "readout", None, True)
    R.readout(r, cells)
    assert r.finish(["test"]) == 0 and "test" not in r.access.opened
    with pytest.raises(ValueError, match="allow_reduced"):
        load_dev(out / "records")
    dev = load_dev(out / "records", allow_reduced=True)
    assert sorted(dev.cells["dev"]) == cells
    assert json.loads((out / "analysis/dev_analysis.json").read_text())["status"] == "incomplete"

    r = RS.Runner(ctx, "confirm", None, True)
    R.confirm_readout(r, cells, None)
    assert r.finish(["test"]) == 0
    dev_edits = json.loads(RS.cell_path(out, "readout", "es", 0).read_text())["edits"]
    conf = json.loads(RS.cell_path(out, "confirm/readout", "es", 0).read_text())
    assert conf["guards"] == ["confirm_en"] and conf["edits"].keys() == dev_edits.keys()
    assert any(e["status"] == "matched" for e in conf["edits"].values())
    for name, e in conf["edits"].items():
        d = dev_edits[name]
        if d["status"] == "matched":  # the frozen dev edit at its dose, dev usability carried
            assert (e["scale"], e["k"], e["dev_usable"]) == (d["scale"], d["k"], d["usable"])
            assert e["usable"] == (d["usable"] and e["guards"]["confirm_en"] <= GUARD_MAX)
        else:
            assert e["status"] == d["status"]
    with pytest.raises(AlreadyOpened):
        R.confirm_readout(RS.Runner(ctx, "confirm", None, True), cells, None)
    r = RS.Runner(ctx, "confirm", None, True)
    R.confirm_readout(r, cells, None, resume=True)
    assert r.state["confirm_es_draw0"] == "ok" and (out / "confirm/resume_log.jsonl").exists()
    RS.cell_path(out, "calibration", "es", 0).write_text("{}")
    with pytest.raises(AccessError, match="changed after the freeze"):
        verify_freeze(out)


def test_the_atlas_does_not_rank_atoms_with_a_non_finite_effect_or_protected_rise(monkeypatch):
    val_t, protected = torch.zeros(2, 3), torch.ones(2, 3)

    def ce_docs(model, base, tokens, update, device, bs):
        if update is None:
            return torch.zeros(2, dtype=torch.float64)
        c, a = update
        if (c == 0 and tokens is protected) or (c == 1 and a == 2.0):
            return torch.full((2,), math.nan, dtype=torch.float64)
        return torch.full((2,), a * (0.1 if tokens is protected else 1.0), dtype=torch.float64)

    monkeypatch.setattr(RO, "ce_docs", ce_docs)
    monkeypatch.setattr(RO, "atom_update", lambda dic, m, c, a, device: (c, abs(a)))
    top = [(MODULES[0], c, 1.0, f"m:{c}", 0.1, 1.0, 1.0) for c in range(3)]
    res = RO.atlas(None, None, None, {"top64": top}, val_t, protected, "cpu")
    assert [(row["status"], row.get("technical_failure")) for row in res["rows"]] == [
        (RO.PROTECTED_TECHNICAL, True), (TECHNICAL, True), ("matched", None)]
    assert res["most_selective_rank"] == 2 and res["rows"][2]["protected_delta_ce"] == pytest.approx(0.05)
