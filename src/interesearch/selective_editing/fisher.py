"""The curvature check of q, a hard gate before any selection, on English retain windows only.
Rules: papers/selective-editing/index.qmd#sec-app-methods.

On the first ``n_windows`` retain windows, in float64, for a few atoms of the VPD dictionary and of its first
randomized control:

1. q of the production estimator (K label sets) and a long-run estimate with ``k_big`` label sets and its Monte
   Carlo standard error.
2. the finite-difference curvature ``[KL(p_0 || p_{+hA}) + KL(p_0 || p_{-hA})] / h^2`` (mean per prediction) for
   ``h`` from 1e-4 to 0.3. Over a stable range of ``h`` it equals the model-Fisher curvature, so it must agree with
   the long-run q within its standard error.
3. contraction: the per-sequence projection from the hooks against ``u^T (dL/dW) v`` from autograd.
4. batch-split independence: q with batch size 1 and with the full batch.
5. scale: with ``u -> a u``, ``v -> b v`` the projection scales by ``ab``, g by ``ab`` and ``||A||^2`` by
   ``(ab)^2``, so the score is unchanged.

Atoms: the historical atoms 0.attn.o:268, 0.attn.v:271 and 3.attn.v:513 (VPD only) and, in each dictionary, the
usable atoms with the largest q, the median q and the 1st-percentile ``q/||A||^2``. ``CRITERIA`` decide ``passed``.
The record names the code it checked (``selector.code_identity``) and the runtime it ran on. Selection accepts only
a passed record of the current code with at least 8 windows of 257 tokens x 32 label sets
(``selector.require_gate``), so a reduced run never opens it, and, as a production guard of this implementation
rather than a gate criterion, only one made on the backend, GPU model, SM count and torch/CUDA version of the
selection (``selector.select``). Per atom, the record also gives the relative difference between the production q
and its replay from the first K long-run label sets (``diagnostic_rel_diff_q_production_vs_replayed``). It is a
diagnostic and does not enter ``passed``.

    uv run python -m interesearch.selective_editing.fisher assets --retain retain_en.npy --out fisher_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..assets import MODULES, load_target
from ..edits import applied_update, capture_base, logits_of
from ..training import disable_tf32
from . import selector as S
from .dictionary import atom_update, label_key, load_dictionary, parse_label_key, randomized_dictionary

HISTORICAL_ATOMS = ("0.attn.o:268", "0.attn.v:271", "3.attn.v:513")
EPS_LIST = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
SCALE_AB = (3.0, 0.25)  # u -> a u, v -> b v in the scale check
CRITERIA = {"contraction_rel_err_max": 1e-8, "batch_split_rel_diff_max": 1e-8, "scale_rel_err_max": 1e-8,
            "fd_stable_min_eps": 3, "fd_stable_tol": 0.05, "z_abs_max": 4.0, "ratio_min": 0.5, "ratio_max": 2.0,
            "pooled_z_abs_max": 3.0}


def pick_atoms(dic, q, extra=()) -> list[tuple[str, int, str]]:
    out = [(*parse_label_key(lk), f"historical {lk}") for lk in extra]
    keys, qs, rs = [], [], []
    for m in dic.modules:
        for c in torch.nonzero(dic[m].usable).flatten().tolist():
            keys.append((m, c))
            qs.append(float(q[m][c]))
            rs.append(float(q[m][c]) / float(dic[m].norms[c]) ** 2)
    order_q = torch.tensor(qs).argsort()
    order_r = torch.tensor(rs).argsort()
    out.append((*keys[int(order_q[-1])], "max q"))
    out.append((*keys[int(order_q[len(order_q) // 2])], "median q"))
    out.append((*keys[int(order_r[int(0.01 * len(order_r))])], "p1 q/||A||^2"))
    return out


def per_sample_q(model, dic, atoms, tokens, device, K: int, seed: int) -> torch.Tensor:
    """``[n_windows, K, n_atoms]``: squared per-sequence projection / T per window and label set (the summands of
    ``retain_fisher``, same label generator)."""
    mods = sorted({m for m, _, _ in atoms})
    dtype = next(model.parameters()).dtype
    out = torch.zeros(tokens.shape[0], K, len(atoms), dtype=torch.float64)
    with S.Capture(model, mods) as cap:
        for i in range(tokens.shape[0]):
            w = tokens[i:i + 1].to(device)
            T = w.shape[1] - 1
            lp = F.log_softmax(logits_of(model(w[:, :-1])), -1)
            probs = lp.detach().exp()
            for k in range(K):
                y = S.sample_labels(probs, [i], k, seed)
                (-lp.gather(-1, y.unsqueeze(-1)).sum()).backward(retain_graph=k < K - 1)
                for j, (m, c, _) in enumerate(atoms):
                    U = dic[m].U[c:c + 1].to(device=device, dtype=dtype)
                    V = dic[m].V[:, c:c + 1].to(device=device, dtype=dtype)
                    out[i, k, j] = float(S.per_seq_projection(cap.x[m], cap.dy[m], U, V)[0, 0]) ** 2 / T
    return out


@torch.no_grad()
def fd_curvature(model, base, dic, atoms, tokens, device, eps_list=EPS_LIST, bs: int = 8) -> list[dict]:
    lp0 = []
    for s in range(0, tokens.shape[0], bs):
        w = tokens[s:s + bs].to(device)
        lp0.append(F.log_softmax(logits_of(model(w[:, :-1])), -1))
    rows = []
    for m, c, why in atoms:
        r = {"label_key": label_key(m, c), "why": why, "fd": {}}
        for e in eps_list:
            kl = 0.0
            for sign in (1.0, -1.0):
                tot, n = 0.0, 0
                with applied_update(model, base, atom_update(dic, m, c, sign * e, device, dtype=torch.float64)):
                    for bi, s in enumerate(range(0, tokens.shape[0], bs)):
                        w = tokens[s:s + bs].to(device)
                        lp = F.log_softmax(logits_of(model(w[:, :-1])), -1)
                        tot += float((lp0[bi].exp() * (lp0[bi] - lp)).sum(-1).sum())
                        n += lp.shape[0] * lp.shape[1]
                kl += tot / n
            r["fd"][f"{e:g}"] = kl / e ** 2
        rows.append(r)
    return rows


def stable_range(fd: dict[str, float], tol: float) -> tuple[list[str], float | None]:
    """Longest run of consecutive h whose FD curvature stays within ``tol`` of the run's median."""
    ks = list(fd)
    best = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks) + 1):
            run = ks[i:j]
            v = torch.tensor([fd[k] for k in run])
            med = float(v.median())
            if med > 0 and float((v - med).abs().max()) <= tol * med and len(run) > len(best):
                best = run
    if len(best) < 2:
        return [], None
    return best, float(torch.tensor([fd[k] for k in best]).median())


def contraction_check(model, dic, atoms, tokens, device, seed: int) -> list[dict]:
    """Hooks against autograd for one window and one label set."""
    w = tokens[:1].to(device)
    mods = sorted({m for m, _, _ in atoms})
    with S.Capture(model, mods) as cap:
        lp = F.log_softmax(logits_of(model(w[:, :-1])), -1)
        y = S.sample_labels(lp.detach().exp(), [0], 0, seed)
        (-lp.gather(-1, y.unsqueeze(-1)).sum()).backward()
        hooks = {(m, c): float(S.per_seq_projection(cap.x[m], cap.dy[m], dic[m].U[c:c + 1].to(cap.x[m]),
                                                    dic[m].V[:, c:c + 1].to(cap.x[m]))[0, 0]) for m, c, _ in atoms}
    for m in mods:
        model.get_submodule(m).weight.requires_grad_(True)
    lp = F.log_softmax(logits_of(model(w[:, :-1])), -1)
    (-lp.gather(-1, y.unsqueeze(-1)).sum()).backward()
    rows = []
    for m, c, _ in atoms:
        W = model.get_submodule(m).weight
        ag = float(dic[m].U[c].to(W.grad) @ W.grad @ dic[m].V[:, c].to(W.grad))
        rows.append({"label_key": label_key(m, c), "hooks": hooks[(m, c)], "autograd": ag,
                     "rel_err": abs(hooks[(m, c)] - ag) / max(abs(ag), 1e-300)})
    for m in mods:
        W = model.get_submodule(m).weight
        W.grad = None
        W.requires_grad_(False)
    return rows


def scale_check(model, dic, atoms, tokens, device, seed: int, q_prod, eps: float) -> list[dict]:
    """Rescale u by a and v by b (one window, one label set, g on two retain windows)."""
    a, b = SCALE_AB
    s = a * b
    mods = sorted({m for m, _, _ in atoms})
    w = tokens[:1].to(device)
    dtype = next(model.parameters()).dtype
    with S.Capture(model, mods) as cap:
        lp = F.log_softmax(logits_of(model(w[:, :-1])), -1)
        y = S.sample_labels(lp.detach().exp(), [0], 0, seed)
        (-lp.gather(-1, y.unsqueeze(-1)).sum()).backward()
        proj = {}
        for m, c, _ in atoms:
            U = dic[m].U[c:c + 1].to(device=device, dtype=dtype)
            V = dic[m].V[:, c:c + 1].to(device=device, dtype=dtype)
            proj[(m, c)] = (float(S.per_seq_projection(cap.x[m], cap.dy[m], U, V)[0, 0]),
                            float(S.per_seq_projection(cap.x[m], cap.dy[m], a * U, b * V)[0, 0]))
    G = S.target_weight_gradient(model, mods, tokens[:2], device, bs=2)

    def rel(x, y):
        return abs(x - y) / max(abs(y), 1e-300)

    rows = []
    for m, c, _ in atoms:
        u, v = dic[m].U[c].double(), dic[m].V[:, c].double()
        g, g_s = float(u @ G[m] @ v), float((a * u) @ G[m] @ (b * v))
        n2, n2_s = float(torch.outer(u, v).pow(2).sum()), float(torch.outer(a * u, b * v).pow(2).sum())
        q = float(q_prod[m][c])
        sc = g * g / (q + eps * n2)
        sc_s = g_s * g_s / (q * s ** 2 + eps * n2_s)  # q of s A is s^2 q (projection check)
        p0, p1 = proj[(m, c)]
        rows.append({"label_key": label_key(m, c), "s": s, "projection_rel_err": rel(p1, s * p0),
                     "g_rel_err": rel(g_s, s * g), "norm2_rel_err": rel(n2_s, s * s * n2),
                     "score_rel_err": rel(sc_s, sc)})
    return rows


def evaluate(res: dict) -> list[str]:
    """Failures against ``CRITERIA`` (empty: passed). Sets ``res["pooled_z"]``."""
    k = CRITERIA
    fails, zs = [], []

    def fin(x):
        return x is not None and math.isfinite(x)

    for dname, d in res["dicts"].items():
        for r in d["contraction"]:
            if not (fin(r["hooks"]) and fin(r["autograd"]) and r["rel_err"] <= k["contraction_rel_err_max"]):
                fails.append(f"{dname} contraction {r['label_key']}: rel_err {r['rel_err']:.3g}")
        if not (fin(d["batch_split_max_rel_diff"]) and d["batch_split_max_rel_diff"] <= k["batch_split_rel_diff_max"]):
            fails.append(f"{dname} batch split: {d['batch_split_max_rel_diff']:.3g}")
        for r in d["scale"]:
            worst = max(r["projection_rel_err"], r["g_rel_err"], r["norm2_rel_err"], r["score_rel_err"])
            if not (fin(worst) and worst <= k["scale_rel_err_max"]):
                fails.append(f"{dname} scale {r['label_key']}: {worst:.3g}")
        for a in d["atoms"]:
            name = f"{dname} {a['label_key']} ({a['why']})"
            if not (all(fin(v) and v > 0 for v in a["fd"].values()) and fin(a["q_big"]) and a["q_big"] > 0
                    and fin(a["q_big_se"]) and a["q_big_se"] > 0 and fin(a["q_production"])):
                fails.append(f"{name}: non-finite or non-positive value")
                continue
            if len(a["fd_stable_eps"]) < k["fd_stable_min_eps"]:
                fails.append(f"{name}: FD stable over {len(a['fd_stable_eps'])} eps only")
                continue
            z, ratio = a["z_q_big_vs_fd"], a["q_big"] / a["fd_stable_value"]
            zs.append(z)
            if abs(z) > k["z_abs_max"]:
                fails.append(f"{name}: z {z:.2f}")
            if not k["ratio_min"] <= ratio <= k["ratio_max"]:
                fails.append(f"{name}: q_big/FD {ratio:.3f}")
    if not zs:
        fails.append("no atom with a stable FD range")
    else:
        pooled = sum(zs) / math.sqrt(len(zs))
        res["pooled_z"] = pooled
        if abs(pooled) > k["pooled_z_abs_max"]:
            fails.append(f"pooled z {pooled:.2f}")
    return fails


def check_dictionary(model, base, dic, atoms_extra, toks, device, k_big: int) -> dict:
    """Every check for one dictionary on ``toks`` (the model in float64)."""
    n_windows = toks.shape[0]
    seed = S.estimator_seed("fisher_check", dic.name)
    mods = list(dic.modules)
    t_start = time.time()
    prod = S.retain_fisher(model, dic, mods, toks, device, K=S.FISHER_K, seed=seed, bs=n_windows)
    t_prod = time.time() - t_start
    atoms = pick_atoms(dic, prod["q"], atoms_extra)
    ps = per_sample_q(model, dic, atoms, toks, device, k_big, seed)  # [N, K, A]
    per_win = ps.mean(1)  # windows are the independent units
    q_big = per_win.mean(0)
    se_windows = per_win.std(0) / math.sqrt(n_windows)  # includes the window-to-window spread
    # the FD curvature is computed on the same windows, so only the label-sampling noise separates the two
    se_big = (ps.var(1).sum(0) / k_big).sqrt() / n_windows
    q_prod_replay = ps[:, :S.FISHER_K].mean((0, 1))
    fd = fd_curvature(model, base, dic, atoms, toks, device)
    rows = []
    for j, (m, c, _) in enumerate(atoms):
        stable, fd_val = stable_range(fd[j]["fd"], CRITERIA["fd_stable_tol"])
        z = None if fd_val is None else (float(q_big[j]) - fd_val) / max(float(se_big[j]), 1e-300)
        q_prod, q_replay = float(prod["q"][m][c]), float(q_prod_replay[j])
        rows.append({**fd[j], "q_production": q_prod, "q_production_replayed": q_replay,
                     "diagnostic_rel_diff_q_production_vs_replayed": q_prod / q_replay - 1.0,  # not a criterion
                     "q_big": float(q_big[j]), "q_big_se": float(se_big[j]),
                     "q_big_se_with_window_spread": float(se_windows[j]),
                     "per_sample_rel_sd": float(ps[:, :, j].std() / ps[:, :, j].mean().clamp_min(1e-300)),
                     "fd_stable_eps": stable, "fd_stable_value": fd_val, "z_q_big_vs_fd": z,
                     "rel_diff_q_big_vs_fd": None if fd_val is None else float(q_big[j]) / fd_val - 1.0,
                     "rel_diff_q_production_vs_fd": None if fd_val is None else q_prod / fd_val - 1.0})
    one = S.retain_fisher(model, dic, mods, toks[:4], device, K=2, seed=seed, bs=1)
    full = S.retain_fisher(model, dic, mods, toks[:4], device, K=2, seed=seed, bs=4)
    split = max(float(((one["q"][m] - full["q"][m]).abs() / full["q"][m].abs().clamp_min(1e-300)).max()) for m in mods)
    eps = S.ridge_floor(dic, prod["q"])
    return {"atoms": rows, "contraction": contraction_check(model, dic, atoms, toks, device, seed),
            "batch_split_max_rel_diff": split,
            "scale": scale_check(model, dic, atoms, toks, device, seed, prod["q"], eps),
            "production_q_seconds": t_prod}


def run(assets_dir: str | Path, retain: torch.Tensor, device, n_windows: int = S.GATE_WINDOWS,
        k_big: int = S.GATE_K_BIG) -> dict:
    """The check on the first ``n_windows`` rows of ``retain`` for the VPD dictionary and its first randomized
    control. Returns the record."""
    if retain.shape[0] < n_windows:
        raise ValueError(f"the check needs {n_windows} retain windows, got {retain.shape[0]}")
    t0 = time.time()
    disable_tf32()
    model = load_target(assets_dir, device, dtype=torch.float64)
    toks = retain[:n_windows]
    base = capture_base(model, MODULES)
    vpd = load_dictionary(assets_dir)
    res = {"n_windows": n_windows, "window": int(toks.shape[1]), "k_big": k_big, "fisher_k": S.FISHER_K,
           "dtype": "float64", "eps_list": EPS_LIST, "dicts": {}}
    for dic, extra in ((vpd, HISTORICAL_ATOMS), (randomized_dictionary(vpd, S.NULL_SEED[0]), ())):
        res["dicts"][dic.name] = check_dictionary(model, base, dic, extra, toks, device, k_big)
    res["seconds"] = time.time() - t0
    res["criteria"] = CRITERIA
    res["failures"] = evaluate(res)
    res["passed"] = not res["failures"]
    res["code"] = S.code_identity()
    res["provenance"] = S.provenance(model, device, n_windows, {"retain": toks}, res["code"], dictionary=vpd.meta)
    res["device"] = str(torch.device(device))
    res["torch"] = torch.__version__
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The curvature check of q (the gate before selection).")
    ap.add_argument("assets_dir", type=Path)
    ap.add_argument("--retain", type=Path, required=True, help="int windows [N, 257] of the English retain pool (.npy)")
    ap.add_argument("--out", type=Path, required=True, help="record to write (.json)")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--n-windows", type=int, default=S.GATE_WINDOWS)
    ap.add_argument("--k-big", type=int, default=S.GATE_K_BIG)
    a = ap.parse_args(argv)
    retain = torch.from_numpy(np.load(a.retain).astype(np.int64))
    rec = run(a.assets_dir, retain, torch.device(a.device) if a.device else S.default_device(), a.n_windows, a.k_big)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(a.out, "passed" if rec["passed"] else f"FAILED: {rec['failures']}")
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
