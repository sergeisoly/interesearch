import json
import math
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from interesearch.assets import MODULES
from interesearch.checksums import sha256_file
from interesearch.edits import capture_base
from interesearch.model import LlamaSimpleMLP, LlamaSimpleMLPConfig
from interesearch.repair_transfer import finetune, fitting, run, selection
from interesearch.repair_transfer.data import load_block, load_windows
from interesearch.repair_transfer.evaluate import PARITY_TOL, parity

V = 64
LANGS = run.LANGS
IDENTITY = {"target/model_step_99999.pt": "w" * 64, "target/model_config.yaml": "c" * 64}


def tiny_model():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        cfg = LlamaSimpleMLPConfig(block_size=512, vocab_size=V, n_layer=4, n_head=2, n_embd=32, n_intermediate=64,
                                   rotary_dim=16, n_ctx=512, n_key_value_heads=2, flash_attention=False)
        return LlamaSimpleMLP(cfg).eval().requires_grad_(False)


MODEL = tiny_model()
BASE = capture_base(MODEL, MODULES)
SHAPES = {m: tuple(BASE[m].shape) for m in MODULES}


def tokens(role, lang, rows):
    g = torch.Generator().manual_seed(zlib.crc32(f"{role}|{lang}".encode()))
    return torch.randint(0, V, (rows, 257), generator=g)


def make_D(scale=0.05, seed=7):
    g = torch.Generator().manual_seed(seed)
    return {m: scale * torch.randn(SHAPES[m], generator=g) for m in MODULES}


def make_donor(seed=3):
    g = torch.Generator().manual_seed(seed)
    return {m: (0.3 * torch.randn(SHAPES[m][0], generator=g), 0.3 * torch.randn(SHAPES[m][1], generator=g))
            for m in MODULES}


def cap_of(D):
    return math.sqrt(sum(float(D[m].double().norm() ** 2) for m in MODULES))


def write_windows(d: Path, role: str, langs, rows: int, name: str | None = None, source=None) -> Path:
    """A window directory: windows.json, int32 <lang>.npy [rows, 257] and SHA256SUMS."""
    d.mkdir(parents=True)
    meta = {"role": role, "dates": [f"2000-01-{i + 1:02d}" for i in range(rows)], "languages": list(langs)}
    if source is not None:
        meta["source"] = source
    (d / "windows.json").write_text(json.dumps(meta))
    for lg in langs:
        np.save(d / f"{lg}.npy", tokens(name or role, lg, rows).numpy().astype(np.int32))
    files = ["windows.json", *(f"{lg}.npy" for lg in langs)]
    (d / "SHA256SUMS").write_text("".join(f"{sha256_file(d / f)}  {f}\n" for f in files))
    return d


def write_finetune(d: Path, D: dict, seed=2, rows=None) -> Path:
    """A fine-tune directory as finetune.py writes it (update.pt and the fields of run.json that runs read)."""
    d.mkdir(parents=True)
    torch.save({"modules": D, "meta": {"step": 200}}, d / "update.pt")
    rec = {"config": {"condition": "matrix", "lr": 3e-6, "seed": seed, "steps": 200},
           "selection": {"usable": True, "selected_step": 200}, "gates": {"passed": True, "failed": []},
           "saved": {"update_sha256": sha256_file(d / "update.pt"), "saved_step": 200}, "target_model": IDENTITY,
           "data": {"train": {"role": finetune.TRAIN_ROLE, "source": {"role": "donor_readout", "rows": rows}}}}
    (d / "run.json").write_text(json.dumps(rec))
    return d


def pick_last(per, removed_fro, names, idx):
    """Plumbing stand-in for the selection rule: the tiny model has no feasible candidate, so the end-to-end tests
    choose the last candidate of each group to exercise the readout, parity and direction paths."""
    return {"selected": names[-1], "n_feasible": len(names), "tried": []}


@pytest.fixture
def toy(monkeypatch, tmp_path):
    D, donor = make_D(), make_donor()
    donor_path = tmp_path / "donor_repair.pt"
    torch.save({"factors": donor, "name": "toy donor", "cap_donor": 1.0}, donor_path)
    opened = []
    real_load = run.load_windows

    def load_windows_spy(directory, *args, **kwargs):
        opened.append(json.loads((Path(directory) / "windows.json").read_text())["role"])
        return real_load(directory, *args, **kwargs)

    monkeypatch.setattr(run, "load_windows", load_windows_spy)
    monkeypatch.setattr(run, "setup", lambda a: (MODEL, BASE, {"toy": True}))
    monkeypatch.setattr(run, "target_model_identity", lambda d: (IDENTITY, []))
    monkeypatch.setattr(run, "DONOR_REPAIR_SHA256", sha256_file(donor_path))
    monkeypatch.setattr(run, "DAMAGE_FLOOR", -1e9)
    return SimpleNamespace(
        tmp=tmp_path, D=D, donor=donor, donor_path=donor_path, opened=opened,
        ft=write_finetune(tmp_path / "ft", D),
        fit=write_windows(tmp_path / "w" / "fit", "fit", run.FIT_LANGS, 32),
        sel=write_windows(tmp_path / "w" / "selection", "selection", LANGS, 32),
        standin=write_windows(tmp_path / "w" / "standin", finetune.VAL_ROLE, LANGS, 32),
        readout=write_windows(tmp_path / "w" / "readout", "readout", LANGS, 64),
    )


def block_args(t, block, out, readout=None, *extra):
    return ["block", "--target", "standin", "--block", block, "--assets", str(t.tmp / "assets"),
            "--finetune", str(t.ft), "--donor-repair", str(t.donor_path), "--fit", str(t.fit),
            "--selection", str(t.sel), "--readout", str(readout or t.standin), "--out", str(out), "--device", "cpu",
            "--threads", "1", "--steps", "20", "--every", "10", "--step-caps", "10", "--polish-iters", "40",
            "--rows", "6", "--eval-batch-size", "8", *extra]


def record(d):
    return json.loads((Path(d) / "run.json").read_text())


def few_families(monkeypatch, families):
    """A main block of a few families at one learning rate each (exit-code tests)."""
    monkeypatch.setitem(run.BLOCKS, "main", {**run.BLOCKS["main"], "families": families})
    monkeypatch.setattr(run, "LRS", {f: lrs[:1] for f, lrs in fitting.LRS.items()})


def fit_windows(t):
    return load_windows(t.fit, run.FIT_LANGS).tokens


# ---------------------------------------------------------------- families, forward paths, polish
def test_family_starts():
    D, donor = make_D(), make_donor()
    cap = cap_of(D)
    rd = {"random_init": fitting.randomized(donor, 20261011), "random_init_seed2": fitting.randomized(donor, 20261012)}
    S = fitting.make_method("strengths", SHAPES, "cpu", cap, donor, rd)
    assert fitting.c0_of(donor, cap) == pytest.approx(S.c0, rel=1e-12)
    E_start = {m: e.detach().double() for m, e in S.E().items()}
    meths = {f: fitting.make_method(f, SHAPES, "cpu", cap, donor, rd) for f in run.FAMILIES}
    for fam, m in meths.items():
        ic = fitting.init_check(fam, m, E_start, cap)
        assert ic["ok"], (fam, ic)
        assert m.norm() <= cap * (1 + fitting.INIT_TOL)
    W, Wo, F1, F2 = (meths[f] for f in ("donor_init", "donor_init_output", "fresh", "fresh_seed2"))
    assert len(W.params) == 48 and len(Wo.params) == 24 and all(p is Wo.b[m] for p, m in zip(Wo.params, MODULES))
    assert not any(Wo.a[m].requires_grad for m in MODULES)
    for m in MODULES:
        assert torch.equal(W.a[m], donor[m][1]) and float(F1.b[m].detach().abs().max()) == 0.0
    assert not torch.equal(F1.a[MODULES[0]], F2.a[MODULES[0]])
    assert not torch.equal(rd["random_init"][MODULES[0]][0], rd["random_init_seed2"][MODULES[0]][0])


def test_donor_init_output_trains_only_output_vectors(toy):
    cap = cap_of(toy.D)
    fs = fitting.fit_sets(fit_windows(toy), "256")
    m = fitting.make_method("donor_init_output", SHAPES, "cpu", cap, toy.donor, {})
    a0 = {k: v.clone() for k, v in m.a.items()}
    b0 = {k: v.detach().clone() for k, v in m.b.items()}
    ck, lg, st = fitting.train(MODEL, BASE, toy.D, m, fs, 1e-3, "cpu", lambda s: None, "x", 12, 6)
    assert st["finite"] and sorted(ck) == [6, 12] and st["updates_applied"] == 12
    assert all(torch.equal(m.a[k], a0[k]) for k in MODULES)
    assert any(not torch.equal(m.b[k].detach(), b0[k]) for k in MODULES)
    assert st["max_post_over_cap"] <= 1 + fitting.CAP_TOL


def test_fit_sets_are_nested(toy):
    fs = {B: fitting.fit_sets(fit_windows(toy), B) for B in ("256", "512", "2K")}
    tot = {B: fitting.fit_accounting(fs[B])["total_unique_predicted_tokens"] for B in fs}
    assert tot == {"256": 256, "512": 512, "2K": 2048}
    for small, big in (("256", "512"), ("512", "2K")):
        for g in ("de", "nl"):
            n, L1 = fs[small][g].shape
            assert torch.equal(fs[small][g], fs[big][g][:n, :L1])
        for lg in fitting.RETAIN:
            n, L1 = fs[small]["ret"][lg].shape
            assert torch.equal(fs[small]["ret"][lg], fs[big]["ret"][lg][:n, :L1])
    with pytest.raises(ValueError, match="shorter than the prefixes of budget 2K"):  # never 1,024 tokens as 2K
        fitting.fit_sets({lg: w[:1] for lg, w in fit_windows(toy).items()}, "2K")


@pytest.mark.parametrize("budget", ["256", "2K"])
def test_training_and_export_paths_agree(toy, budget):
    """The fit loss of the training path equals the training loss. The export path is within FIT_PATH_TOL and the
    logits of the two paths within PARITY_TOL."""
    D, donor = toy.D, toy.donor
    cap = cap_of(D)
    fs = fitting.fit_sets(fit_windows(toy), budget)
    m1, m2 = (fitting.make_method("donor_init", SHAPES, "cpu", cap, donor, {}) for _ in range(2))
    _, lg, _ = fitting.train(MODEL, BASE, D, m1, fs, 1e-4, "cpu", lambda s: None, "x", 1, 1)
    fl = fitting.FitLoss(MODEL, BASE, D, fs, "cpu")
    comps = fl.train_path(m2.E())
    L = float((comps["de"] + comps["nl"] + comps["ret"]).detach())
    assert L == pytest.approx(lg[0]["loss"], rel=1e-5)
    spec = m2.spec(m2.snapshot())
    assert abs(fl.export(spec)["total"] - L) <= fitting.FIT_PATH_TOL
    assert parity(MODEL, BASE, D, spec, fs["nl"], "cpu") <= PARITY_TOL
    Dd = dict(D)
    zero = fitting.update_of({"kind": "zero"}, Dd, "cpu")
    assert all(torch.equal(zero[m], D[m]) for m in MODULES) and fitting.fro_of({"kind": "zero"}, Dd, "cpu") == 0.0
    raw = fitting.fro_of({"kind": "scaled", "k": 1.0, "factors": donor}, Dd, "cpu")
    assert fitting.fro_of({"kind": "scaled", "k": 0.5, "factors": donor}, Dd, "cpu") == pytest.approx(raw / 2)
    S = fitting.Method.strengths(donor, cap, "cpu")
    assert fitting.fro_of(S.spec(S.snapshot(), donor), Dd, "cpu") == pytest.approx(S.norm(), rel=1e-12)


def test_polish_decreases_respects_cap_and_converges(toy):
    D, donor = toy.D, toy.donor
    cap = cap_of(D)
    fl = fitting.FitLoss(MODEL, BASE, D, fitting.fit_sets(fit_windows(toy), "2K"), "cpu")
    S = fitting.Method.strengths(donor, cap, "cpu")
    logs = []
    rec, snap = fitting.polish(fl, S, torch.full((24,), S.c0), cap, 300, logs.append, "c0")
    assert rec["finite"] and snap is not None and bool(torch.isfinite(snap["c"]).all())
    assert rec["L_end"] < rec["L_start"] and rec["norm_over_cap"] <= 1 + 1e-6
    assert rec["path_parity_abs"] <= fitting.FIT_PATH_TOL
    assert rec["converged"], rec
    assert fl.export(S.spec(snap, donor))["total"] == pytest.approx(rec["L_end"], abs=1e-5)
    # the second polish starts at the optimum, measures stationarity on the c0 scale and stays converged
    ref = {"stat": rec["stationarity_start"], "lam0": rec["lam0"]}
    rec2, _ = fitting.polish(fl, S, snap["c"], cap, 300, logs.append, "best", ref)
    assert rec2["ref"] == "c0 polish" and rec2["stationarity_ref"] == rec["stationarity_start"]
    assert rec2["converged"] and rec2["L_end"] <= rec["L_end"] + 1e-6
    # a short polish is not declared converged
    rec3, _ = fitting.polish(fl, S, torch.full((24,), S.c0), cap, 2, logs.append, "c0")
    assert not rec3["converged"] and rec3["stop"] == "max_iter"


def test_polish_nonfinite_trial_is_not_finite(toy, monkeypatch):
    cap = cap_of(toy.D)
    fl = fitting.FitLoss(MODEL, BASE, toy.D, fitting.fit_sets(fit_windows(toy), "2K"), "cpu")
    S = fitting.Method.strengths(toy.donor, cap, "cpu")
    real, calls = fl.train_path, {"n": 0}

    def train_path(E):
        out = real(E)
        if not torch.is_grad_enabled():  # value-only evaluations are line-search trials
            calls["n"] += 1
            if calls["n"] == 1:
                return {k: v * math.nan for k, v in out.items()}
        return out

    monkeypatch.setattr(fl, "train_path", train_path)
    rec, snap = fitting.polish(fl, S, torch.full((24,), S.c0), cap, 300, lambda s: None, "c0")
    assert calls["n"] >= 1 and rec["finite"] is False and rec["converged"] is False and snap is None
    assert "non-finite" in rec["stop"] and rec["nonfinite_trial"]["iter"] == 1


# ---------------------------------------------------------------- blocks end to end
def test_block_main_and_nested(toy, monkeypatch):
    monkeypatch.setattr(run, "select", pick_last)
    out_m, out_n = toy.tmp / "main", toy.tmp / "nested"
    assert run.main(block_args(toy, "main", out_m)) == 0
    info = record(out_m)
    assert info["status"] == "ok" and info["technical"] == {} and info["failed_families"] == {}
    assert info["reduced"] and set(info["deviations"]) == {"steps", "every", "step_caps", "rows", "polish_iters"}
    assert all(v["ok"] for v in info["checks"]["init"].values())
    g = info["selection_groups"]
    assert g["strengths@2K"] == 8 and g["strengths_best@2K"] == 10 and g["donor_init@2K:le10"] == 4
    assert {k for k in g if ":le" in k} == {f"{f}@2K:le10" for f in run.CAPPED}
    assert set(info["polish"]) == {"c0", "best"} and info["polish"]["best"]["ref"] == "c0 polish"
    assert all(r["L_end"] <= r["L_start"] for r in info["polish"].values())
    assert info["selections"]["strengths_best@2K"]["selected"].startswith("strengths_converged@2K:start=")
    assert max(info["checks"]["parity"].values()) <= PARITY_TOL
    assert "readout" not in toy.opened and toy.opened[-1] == finetune.VAL_ROLE
    w = info["inputs"]["windows"]  # what was read: role, checked hashes, dates, rows, fit prefixes
    for arg, d in (("fit", toy.fit), ("selection", toy.sel), ("readout", toy.standin)):
        assert w[arg]["sha256"] == load_windows(d).sha256 and w[arg]["role"] == json.loads(
            (d / "windows.json").read_text())["role"]
    assert w["fit"]["rows"] == 32 and w["fit"]["prefixes"] == {"2K": {"nl": [4, 256], "de": [2, 256], "ret": [1, 128]}}
    assert w["selection"]["rows"] == w["readout"]["rows"] == 6 and w["readout"]["dates"] == info["readout_dates"]
    assert info["inputs"]["donor_repair"] == {"sha256": sha256_file(toy.donor_path), "source": "pinned"}
    with pytest.raises(ValueError, match="reduced"):
        load_block(out_m)
    blk = load_block(out_m, allow_reduced=True)
    assert blk.info["run"]["reduced"] and blk.info["source"]["run_record_sha256"] == sha256_file(out_m / "run.json")
    names = blk.info["readout"]["names"]
    assert names[:6] == ["base", "finetuned", "donor_raw", "donor_fixed", "strengths_converged@2K:start=c0",
                         "strengths_converged@2K:start=best"] and all("::" in n for n in names[6:])
    assert len(blk.info["readout"]["dates"]) == 6 and blk.info["selection"]["rows"] == 6
    assert {g: len(n) for g, n in selection.groups(blk.info).items()} == blk.info["recorded"]["group_sizes"]
    for n, per in blk.selection.items():  # the npz round trip is exact
        for lg in LANGS:
            assert per[lg]["ce"].dtype == torch.float64 and per[lg]["ce"].shape == (6,)
    assert run.main(block_args(toy, "main", out_m)) == run.RC_REFUSED  # never overwritten

    monkeypatch.setattr(run, "select", selection.select)  # the frozen rule: nothing is feasible on the toy model
    assert run.main(block_args(toy, "nested", out_n)) == 0
    ni = record(out_n)
    assert set(ni["budgets"]) == {"256", "512"} and ni["polish"] == {} and ni["s_adequacy_verified"] is False
    assert set(ni["selection_groups"]) == {f"{f}@{B}{c}" for f in run.FAMILIES[:5] for B in ("256", "512")
                                           for c in ("", ":le10")}
    nb = load_block(out_n, allow_reduced=True)
    assert nb.info["recorded"]["fit_tokens"] == {"256": 256, "512": 512}
    assert selection.reselect(nb) == {g: {k: s[k] for k in ("selected", "n_feasible", "tried")}
                                      for g, s in ni["selections"].items()}


def test_pre_readout_stop_opens_no_readout(toy, monkeypatch):
    monkeypatch.setattr(run, "DAMAGE_FLOOR", 1e9)
    out = toy.tmp / "stop"
    assert run.main(block_args(toy, "main", out)) == run.RC_PRE_READOUT_STOP
    info = record(out)
    assert info["status"] == "pre_readout_stop" and info["pre_readout_stop"] and info["runs"] == {}
    assert toy.opened == ["selection"] and sorted(p.name for p in out.iterdir()) == ["run.json"]
    assert list(info["inputs"]["windows"]) == ["selection"]  # recorded when read, and nothing else was read
    with pytest.raises(ValueError, match="pre_readout_stop"):
        load_block(out)


def test_nonfinite_damage_is_technical_not_a_stop(toy, monkeypatch):
    real_eval = run.evaluate

    def evaluate(model, base, D, cands, sets, device, bs=16):
        per = real_eval(model, base, D, cands, sets, device, bs)
        if not cands:  # the damage evaluation of base and finetuned on the selection windows
            per["finetuned"]["nl"]["ce"][0] = math.nan
        return per

    monkeypatch.setattr(run, "evaluate", evaluate)
    monkeypatch.setattr(run, "DAMAGE_FLOOR", 1e9)
    out = toy.tmp / "damage"
    assert run.main(block_args(toy, "main", out)) == run.RC_TECHNICAL
    info = record(out)
    assert "pre_readout_stop" not in info and info["pre_readout_halt"]["rc"] == run.RC_TECHNICAL
    assert info["pre_readout_halt"]["stop_reasons_not_applied"] and "target" in info["technical"]
    assert info["runs"] == {} and toy.opened == ["selection"]


def test_nonfinite_update_is_technical_before_any_evaluation(toy):
    toy.D[MODULES[3]][0, 0] = math.nan
    ft = write_finetune(toy.tmp / "ft_nan", toy.D)
    out = toy.tmp / "nan"
    args = block_args(toy, "main", out)
    args[args.index("--finetune") + 1] = str(ft)
    assert run.main(args) == run.RC_TECHNICAL
    info = record(out)
    assert info["status"] == "technical" and "damage_selection" not in info and info["runs"] == {}
    assert any("non-finite value in the fine-tune update" in r for r in info["technical"]["target"])
    zero = write_finetune(toy.tmp / "ft_zero", {m: torch.zeros_like(d) for m, d in make_D().items()})
    args[args.index("--finetune") + 1] = str(zero)
    args[args.index("--out") + 1] = str(toy.tmp / "zero")
    assert run.main(args) == run.RC_TECHNICAL  # a zero fine-tune: cap 0, a record and no crash
    assert record(toy.tmp / "zero")["technical"]["target"] == ["cap ||D|| = 0.0"]


def test_nonfinite_training_is_technical_with_a_reason(toy, monkeypatch):
    monkeypatch.setattr(run, "select", pick_last)
    few_families(monkeypatch, ("strengths", "donor_init", "random_init_seed2"))
    real_train, real_eval = run.train, run.evaluate

    def train(model, base, D, meth, fs, lr, device, log, tag, steps, every):
        if tag.startswith("random_init_seed2@"):
            return {}, [], {"finite": False, "steps_planned": steps, "nonfinite_step": 3, "batches_presented": 3,
                            "updates_applied": 2}
        return real_train(model, base, D, meth, fs, lr, device, log, tag, steps, every)

    def evaluate(model, base, D, cands, sets, device, bs=16):
        per = real_eval(model, base, D, cands, sets, device, bs)
        for n in per:
            if n.startswith("donor_init@2K::"):
                per[n]["nl"]["ce"][0] = math.nan
        return per

    monkeypatch.setattr(run, "train", train)
    monkeypatch.setattr(run, "evaluate", evaluate)
    out = toy.tmp / "technical"
    assert run.main(block_args(toy, "main", out)) == run.RC_TECHNICAL
    info = record(out)
    assert {"random_init_seed2@2K", "donor_init@2K"} <= set(info["technical"])
    assert "random_init_seed2@2K" in info["failed_families"] and info["nonfinite_runs"]
    assert not (out / "block.json").exists()
    with pytest.raises(ValueError, match="technical"):
        load_block(out)


def test_nonfinite_checkpoint_is_left_out(toy, monkeypatch):
    few_families(monkeypatch, ("strengths", "fresh"))
    real_train = run.train

    def train(model, base, D, meth, fs, lr, device, log, tag, steps, every):
        ck, lg, st = real_train(model, base, D, meth, fs, lr, device, log, tag, steps, every)
        if tag.startswith("fresh@"):
            ck[10]["factors"][MODULES[0]][0][0] = math.nan
        return ck, lg, st

    monkeypatch.setattr(run, "train", train)
    out = toy.tmp / "nan_ckpt"
    assert run.main(block_args(toy, "main", out)) == run.RC_TECHNICAL
    info = record(out)
    assert any("non-finite parameters at checkpoint 10" in r for r in info["technical"]["fresh@2K"])
    assert info["selection_groups"]["fresh@2K"] == 1


def test_failed_family_exits_8(toy, monkeypatch):
    few_families(monkeypatch, ("strengths", "fresh"))
    real_train = run.train

    def train(model, base, D, meth, fs, lr, device, log, tag, steps, every):
        if tag.startswith("fresh@"):
            return real_train(model, base, D, meth, fs, lr, device, log, tag, every - 1, every)
        return real_train(model, base, D, meth, fs, lr, device, log, tag, steps, every)

    monkeypatch.setattr(run, "train", train)
    out = toy.tmp / "failed"
    assert run.main(block_args(toy, "main", out)) == run.RC_FAILED_FAMILY
    info = record(out)
    assert info["status"] == "failed_family" and info["technical"] == {}
    assert set(info["failed_families"]) == {"fresh@2K"} and "no checkpoint saved" in info["failed_families"]["fresh@2K"]
    assert info["selections"]["fresh@2K"]["selected"] is None and not (out / "block.json").exists()


# ---------------------------------------------------------------- arguments, roles, inputs
def test_argument_refusals(toy):
    args = block_args(toy, "main", toy.tmp / "x")
    with pytest.raises(SystemExit):  # values away from the frozen ones need the stand-in target
        run.main([("primary_target" if a == "standin" else a) for a in args])
    with pytest.raises(SystemExit):
        run.main(args + ["--step-caps", "10,x"])
    with pytest.raises(SystemExit):
        run.main([a for a in args if a != "standin"])


def test_reduced_runs_refuse_readout_roles_before_loading(toy, tmp_path):
    for roles in ("readout", "donor_readout", "selection,readout"):
        d = tmp_path / "bare" / roles.replace(",", "_")
        d.mkdir(parents=True)  # no arrays and no SHA256SUMS: refused on windows.json alone
        (d / "windows.json").write_text(json.dumps({"role": roles, "dates": [], "languages": list(LANGS)}))
        with pytest.raises(ValueError, match="may not be opened"):
            load_windows(d, refused=("readout", "donor_readout"))
        out = tmp_path / "out" / d.name
        assert run.main(block_args(toy, "main", out, d)) == run.RC_REFUSED
        assert not out.exists()
    with pytest.raises(ValueError, match="expected 'fit'"):
        load_windows(toy.sel, role="fit")
    assert toy.opened == []


def test_window_manifest(toy):
    w = load_windows(toy.sel, ("de", "nl"), rows=5)
    assert w.role == "selection" and len(w.dates) == 5 and w.tokens["de"].dtype == torch.int64
    assert torch.equal(w.tokens["nl"], tokens("selection", "nl", 32)[:5])
    with pytest.raises(ValueError, match="SHA256SUMS lists"):  # a requested language the directory does not list
        load_windows(toy.fit, ("pt",))
    np.save(toy.sel / "pt.npy", np.zeros((32, 257), dtype=np.int32))
    with pytest.raises(ValueError, match="sha256"):
        load_windows(toy.sel)


def test_input_identity_refusals(toy, monkeypatch):
    out = toy.tmp / "refused"
    monkeypatch.setattr(run, "target_model_identity", lambda d: ({}, ["target/model_config.yaml: sha256 differs"]))
    assert run.main(block_args(toy, "main", out)) == run.RC_REFUSED
    monkeypatch.setattr(run, "target_model_identity", lambda d: (IDENTITY, []))
    short = write_windows(toy.tmp / "w" / "short_fit", "fit", run.FIT_LANGS, 2)  # 2K takes 4 Dutch windows
    assert run.main([str(short) if a == str(toy.fit) else a for a in block_args(toy, "main", out)]) == run.RC_REFUSED
    monkeypatch.setattr(run, "DONOR_REPAIR_SHA256", "0" * 64)
    assert run.main(block_args(toy, "main", out)) == run.RC_REFUSED
    assert not out.exists()
    (out / "x").mkdir(parents=True)
    assert run.main(block_args(toy, "main", out)) == run.RC_REFUSED


def test_finetune_binding(toy):
    """A target's fine-tune is bound to its recipe and to the window directories it was prepared from."""
    train = write_windows(toy.tmp / "w" / "ft_train", finetune.TRAIN_ROLE, finetune.TRAIN_LANGS, 8,
                          source={"role": "donor_readout", "rows": [0, 64]})
    val = write_windows(toy.tmp / "w" / "ft_val", finetune.VAL_ROLE, finetune.VAL_LANGS, 8)
    ft = json.loads((write_finetune(toy.tmp / "ft14", toy.D, seed=14, rows=[0, 64]) / "run.json").read_text())
    ft["data"] = {"train": {"role": finetune.TRAIN_ROLE, "source": {"role": "donor_readout", "rows": [0, 64]},
                            "sha256": load_windows(train).sha256},
                  "val": {"role": finetune.VAL_ROLE, "sha256": load_windows(val).sha256}}
    assert run.binding_errors("primary_target", ft, str(train), str(val)) == []
    assert len(run.binding_errors("replication_target", ft, str(train), str(val))) == 2  # another seed and draw
    assert len(run.binding_errors("primary_target", ft, None, str(val))) == 1  # no prepared training windows
    del ft["data"]["val"]["sha256"]  # no hashes recorded
    assert run.binding_errors("primary_target", ft, str(train), str(val)) == [
        f"fine-tune val data: the recorded role and hashes are not those of {val}"]
    ft["data"]["train"]["source"]["role"] = "selection"  # an unrelated source with the right rows
    ft["config"]["lr"] = 1e-5
    assert len(run.binding_errors("primary_target", ft, str(train), str(val))) == 3
    arr = np.load(train / "fr.npy")
    np.save(train / "fr.npy", arr + 1)  # a prepared array tampered after the fine-tune
    assert any("sha256 does not match" in e for e in run.binding_errors("primary_target", ft, str(train), str(val)))
    (toy.ft / "update.pt").write_bytes(b"changed")
    assert run.finetune_errors(toy.ft)[1]
    with pytest.raises(ValueError, match="sha256"):
        finetune.load_update(toy.ft)


def test_donor_command(toy, monkeypatch):
    monkeypatch.setattr(run, "select", pick_last)
    out = toy.tmp / "donor"
    args = ["donor", "--assets", str(toy.tmp / "assets"), "--finetune", str(toy.ft), "--fit", str(toy.fit),
            "--selection", str(toy.sel), "--readout", str(toy.standin), "--out", str(out), "--device", "cpu",
            "--threads", "1", "--steps", "10", "--every", "5", "--rows", "6", "--eval-batch-size", "8"]
    assert run.main(args) == run.RC_REFUSED  # the fit and selection roles of the donor stage
    fit = write_windows(toy.tmp / "w" / "donor_fit", "donor_fit", run.FIT_LANGS, 8)
    val = write_windows(toy.tmp / "w" / "val", finetune.VAL_ROLE, LANGS, 8)
    args[args.index("--fit") + 1], args[args.index("--selection") + 1] = str(fit), str(val)
    other = write_finetune(toy.tmp / "ft_other", toy.D)
    (other / "run.json").write_text(json.dumps({**record(other), "target_model": {**IDENTITY, "x": "y"}}))
    wrong = [str(other) if a == str(toy.ft) else a for a in args]
    assert run.main(wrong) == run.RC_REFUSED and not out.exists()  # a fine-tune of another target model
    assert run.main(args) == 0
    info = record(out)
    assert info["status"] == "ok" and info["reduced"] and len(info["runs"]) == 3
    assert info["fit"]["windows"]["de"] == 6  # --rows also cuts the donor fit windows
    assert info["selection"]["selected"] == "donor_repair:lr=0.003:step=10" and info["gate"]["n_primary"] == 6
    assert sorted(p.name for p in out.iterdir()) == ["donor_repair.pt", "run.json"]  # reduced: no fit record
    saved = torch.load(out / "donor_repair.pt", weights_only=True)
    assert saved["name"] == info["selection"]["selected"] and set(saved["factors"]) == set(MODULES)
    assert info["readout"]["role"] == finetune.VAL_ROLE and "readout" not in toy.opened
    real_train = run.train

    def train(*a):  # every run stops non-finite after its first checkpoint
        ck, lg, st = real_train(*a)
        return {5: ck[5]}, lg[:6], {**st, "finite": False, "nonfinite_step": 6, "checkpoints_saved": [5]}

    monkeypatch.setattr(run, "train", train)
    args[args.index("--out") + 1] = str(toy.tmp / "donor_nan")
    assert run.main(args) == run.RC_TECHNICAL
    bad = record(toy.tmp / "donor_nan")
    assert len(bad["technical"]["donor"]) == 3 and "selection" not in bad
    assert sorted(p.name for p in (toy.tmp / "donor_nan").iterdir()) == ["run.json"]
    monkeypatch.setattr(run, "train", real_train)
    real_eval = run.evaluate

    def nan_eval(model, base, D, cands, *a):  # every candidate's Dutch selection values non-finite
        per = real_eval(model, base, D, cands, *a)
        for n in cands:
            per[n]["nl"]["ce"] = torch.full_like(per[n]["nl"]["ce"], math.nan)
        return per

    for name, attr, fake, why in (("donor_sel", "evaluate", nan_eval, "non-finite selection values"),
                                  ("donor_parity", "parity", lambda *a: math.nan, "parity of")):
        with monkeypatch.context() as m:
            m.setattr(run, attr, fake)
            args[args.index("--out") + 1] = str(toy.tmp / name)
            assert run.main(args) == run.RC_TECHNICAL  # not a run without a selection, nor one whose gate failed
        bad = record(toy.tmp / name)
        assert any(why in r for r in bad["technical"]["donor"]) and "readout" not in bad["inputs"]["windows"]
        assert sorted(p.name for p in (toy.tmp / name).iterdir()) == ["run.json"]

    def flat_eval(model, base, D, cands, *a):  # finite readout tensors, zero German damage: rho_de undefined
        per = real_eval(model, base, D, cands, *a)
        if "donor_repair" in cands:
            per["finetuned"]["de"]["ce"] = per["base"]["de"]["ce"].clone()
        return per

    monkeypatch.setattr(run, "evaluate", flat_eval)
    args[args.index("--out") + 1] = str(toy.tmp / "donor_flat")
    assert run.main(args) == run.RC_TECHNICAL  # not a finished run whose gate failed
    bad = record(toy.tmp / "donor_flat")
    assert any("non-finite readout estimate primary/rho_de" in r for r in bad["technical"]["donor"])
    assert sorted(p.name for p in (toy.tmp / "donor_flat").iterdir()) == ["run.json"]
    monkeypatch.setattr(run, "evaluate", real_eval)
    assert {k: v["role"] for k, v in info["inputs"]["windows"].items()} == {
        "fit": "donor_fit", "selection": finetune.VAL_ROLE, "readout": finetune.VAL_ROLE}


def test_new_donor_repair_feeds_a_block(toy, monkeypatch):
    """A donor repair the donor command wrote is accepted by a block only through its run record: a full run that
    ended ok and passed its gate, on the same model, with this file as its chosen candidate."""
    monkeypatch.setattr(run, "select", pick_last)
    fit = write_windows(toy.tmp / "w" / "donor_fit", "donor_fit", run.FIT_LANGS, 8)
    val = write_windows(toy.tmp / "w" / "val", finetune.VAL_ROLE, LANGS, 8)
    donor = toy.tmp / "donor"
    assert run.main(["donor", "--assets", str(toy.tmp / "assets"), "--finetune", str(toy.ft), "--fit", str(fit),
                     "--selection", str(val), "--readout", str(toy.standin), "--out", str(donor), "--device", "cpu",
                     "--threads", "1", "--steps", "10", "--every", "5", "--rows", "6", "--eval-batch-size", "8"]) == 0
    written = record(donor)
    few_families(monkeypatch, ("strengths",))
    args = block_args(toy, "main", toy.tmp / "b0")
    args[args.index("--donor-repair") + 1] = str(donor / "donor_repair.pt")
    assert run.main(args) == run.RC_REFUSED  # a reduced donor run (and the toy donor does not pass its gate)

    def as_written(**change):  # the writer's record, with the fields a full run that passed its gate would hold
        rec = {**written, "reduced": False, "gate": {**written["gate"], "pass": True}, **change}
        (donor / "run.json").write_text(json.dumps(rec))

    for i, change in enumerate(({"status": "technical"}, {"gate": {"pass": False}}, {"command": "block"},
                                {"inputs": {**written["inputs"], "target_model": {}}},
                                {"selection": {**written["selection"], "selected": "donor_repair:lr=0.001:step=10"}},
                                {"donor_repair_sha256": "0" * 64})):
        as_written(**change)
        args[args.index("--out") + 1] = str(toy.tmp / f"b{i + 1}")
        assert run.main(args) == run.RC_REFUSED, change
    as_written()
    args[args.index("--out") + 1] = str(toy.tmp / "ok")
    assert run.main(args) == 0
    got = record(toy.tmp / "ok")["donor_repair"]
    assert got["source"] == "donor run" and got["sha256"] == written["donor_repair_sha256"]
    assert got["name"] == written["selection"]["selected"]

    def failed_export(*a):
        raise OSError("disk full")

    monkeypatch.setattr(run, "write_block", failed_export)  # the block files fail: no record says ok
    args[args.index("--out") + 1] = str(toy.tmp / "export")
    with pytest.raises(OSError):
        run.main(args)
    assert not (toy.tmp / "export" / "run.json").exists()


# ---------------------------------------------------------------- target fine-tune
def ft_data(rows=4):
    return finetune.FTData(target=tokens("ft", "de", rows),
                           retain={lg: tokens("ft", lg, rows) for lg in fitting.RETAIN},
                           val={lg: tokens("val", lg, 2) for lg in finetune.VAL_LANGS})


def test_finetune_toy_run(tmp_path):
    cfg = finetune.FTConfig(lr=1e-3, seed=0, steps=4, warmup=2, checkpoints=(2,), n_target=2, n_retain_per_lang=1,
                            micro_batch=2)
    assert cfg.checkpoints == (2, 4)
    data = ft_data()
    res = finetune.train(MODEL, MODULES, data, cfg, "cpu", lambda s: None, val_batch_size=2)
    assert res.checks["zero_update_max_abs_logit_diff"] == 0.0 and res.checks["frozen_parameters_unchanged"]
    assert sorted(res.updates) == [2, 4] and res.saved_step in (2, 4) and len(res.log) == 4
    assert any(float(D.abs().max()) > 0 for D in res.updates[4].values())
    again = finetune.train(MODEL, MODULES, data, cfg, "cpu", lambda s: None, val_batch_size=2)
    assert all(torch.equal(res.updates[4][m], again.updates[4][m]) for m in MODULES)  # no global seed involved
    step = res.saved_step
    res.checks["eval_path_vs_training_path_de_val"] = finetune.eval_path_check(
        MODEL, MODULES, res.updates[step], data.val["de"], res.val_by_step[step]["de"]["ce_doc"], "cpu", 2)
    assert finetune.apply_gates(res.checks)["passed"]
    saved = finetune.save_result(res, tmp_path, MODULES, IDENTITY)
    (tmp_path / "run.json").write_text(json.dumps({"saved": saved}))
    D, _ = finetune.load_update(tmp_path)
    assert all(torch.equal(D[m], res.updates[step][m]) for m in MODULES)
    val = json.loads((tmp_path / "val.json").read_text())
    other = {**val, "config": {**val["config"], "lr": 3e-3, "track": [], "objective": "hinge"}}
    across = finetune.select_across({"a": val, "b": other})
    assert len(across["rows"]) == 4 and {r["run"] for r in across["rows"]} == {"a", "b"}
    assert finetune.FTConfig.from_dict(other["config"]).lr == 3e-3


def test_checkpoint_rule():
    cfg = finetune.FTConfig(lr=1e-3, seed=0, steps=4, checkpoints=(2,))

    def v(dt, dr, kl):
        row = {lg: {"delta_ce": dr, "kl_base": kl} for lg in fitting.RETAIN}
        return {"de": {"delta_ce": dt, "kl_base": 0.0}, **row}

    sel = finetune.select_checkpoint({0: v(0, 0, 0), 2: v(1.2, 0.05, 0.1), 4: v(0.8, 0.05, 0.01)}, cfg)
    assert sel["selected_step"] == 4 and sel["usable"]  # 0.2 from the preferred rise either way: lower KL wins
    sel = finetune.select_checkpoint({2: v(1.0, 0.2, 0.0), 4: v(0.4, 0.0, 0.0)}, cfg)
    assert sel["selected_step"] is None and not sel["usable"]


def test_finetune_command_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(finetune, "target_model_identity", lambda d: (IDENTITY, []))
    train_dir = write_windows(tmp_path / "train", finetune.TRAIN_ROLE, finetune.TRAIN_LANGS, 4)
    sel = write_windows(tmp_path / "sel", "selection", LANGS, 4)
    out = tmp_path / "out"
    args = ["train", "--lr", "3e-6", "--seed", "14", "--assets", str(tmp_path), "--train", str(train_dir),
            "--val", str(sel), "--out", str(out), "--device", "cpu"]
    assert finetune.main(args) == 3 and not out.exists()  # validation windows of another role
    val_dir = write_windows(tmp_path / "val", finetune.VAL_ROLE, finetune.VAL_LANGS, 4)
    args[args.index("--val") + 1] = str(val_dir)
    (out / "x").mkdir(parents=True)
    assert finetune.main(args) == 3
