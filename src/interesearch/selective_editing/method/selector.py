"""The selection rule: rank every usable atom by g^2 / (q + eps ||A||_F^2), shortlist, re-order by a second q
estimate, and walk the shortlist until an atom reaches the upper effect point on the target train draw.
Rules: papers/selective-editing/index.qmd#sec-selection and papers/selective-editing/index.qmd#sec-app-methods.

* ``g_c = u_c^T G_t v_c``, ``G_t`` the gradient at W_0 of the mean next-token CE over all predictions of the draw.
* ``q_c = 1/(N_pred K) sum_{b,k} s_bk^2`` with ``s_bk = sum_t (u_c . dy_bt)(v_c . x_bt)``: the model-Fisher curvature
  of the mask coordinate on the English retain pool, ``x`` the module input and ``dy`` the gradient at its output
  under the k-th label set sampled from W_0. One backward per label set gives the scores of every atom. The labels
  of window ``i`` under set ``k`` come from a generator on the model's device seeded by ``(seed, i, k)`` only, so q
  does not depend on batching. The device's generator is part of the procedure (q on CUDA and on the CPU differ).
* ``eps`` is the 1st percentile of ``q_c / ||A_c||_F^2`` over usable atoms (retain data only).
* Ranking: score descending, ties by module order, then atom index. The top 64 are re-ordered by
  ``g^2 / (max(q, q_check) + eps ||A||^2)``, ``q_check`` the same estimator on the q-check pool with its own seed.
  Ties keep the ranking order.
* Walk: in that order, atoms with ``g = 0`` skipped. ``alpha = sign(g) a`` for ``a`` on the atom grid until the mean
  CE rise of the draw reaches the upper effect point. The first atom that reaches it is accepted, else the first
  that reached the lower point ("tau_lo only"), else "no feasible atom". A non-finite g, q, q_check, eps, score or
  rise is a technical failure, never an empty ranking or "no feasible atom".

Selection and q refuse to run without a passed full-size curvature-check record of the current code of q, its
check, the dictionary and the model forward (``require_gate``). As a production guard of this implementation (not
one of the dated gate criteria), selection also refuses a check made on another backend, GPU model, SM count or
torch/CUDA version. A stored record (q here, the English means of ``references`` and ``neurons``) is reused only
when its provenance (the code that computed it, device, label generator, SM count, torch and precision settings,
batch size, seeds, the model weights and configuration, the dictionary and the input tokens) equals the current
one, and refused otherwise. A cached target gradient under other conditions is recomputed. Selection also refuses
a q record made with other code, on another device or runtime, or for another model or dictionary than the
selection itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from interesearch.edits import applied_update, capture_base, logits_of
from interesearch.training import disable_tf32
from interesearch.selective_editing.method import runtime
from interesearch.selective_editing.method.runtime import FISHER_K
from interesearch.selective_editing.method.measure import doc_ce
from interesearch.selective_editing.method.dictionary import Dictionary, atom_update, label_key

Tensor = torch.Tensor

# Original study seed namespace. Changing its bytes changes the published draws and Fisher samples.
SELECTION_SALT = "e-prime-v1"
DRAW_ROWS = 64
NULL_SEED = {0: 101, 1: 102, 2: 103}  # randomized dictionary per draw, the same for every target
TAUS = (0.5, 1.0)  # nats of target CE rise
A_MAX = 2.0
ATOM_GRID = tuple(0.0625 * 2 ** (k / 2) for k in range(11))  # 0.0625 ... 2.0, x sqrt(2)
EPS_PERCENTILE = 1.0
MAX_WALK = 64
BATCH = 16


def sha_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def draw_rows(dates: Sequence[str], d: int, n_rows: int = DRAW_ROWS) -> list[int]:
    """Rows of draw ``d``: the first ``n_rows`` train windows in salted-sha256 order of their session date, sorted.
    One window per date, the same rows in every language."""
    if len(set(dates)) != len(dates):
        raise ValueError("expected one train window per date")
    order = sorted(range(len(dates)), key=lambda i: sha_int(f"{SELECTION_SALT}|draw|{d}|{dates[i]}"))
    return sorted(order[:n_rows])


def estimator_seed(tag: str, dictionary_name: str) -> int:
    """Run seed of a q estimate: ``tag`` is ``q``, ``qcheck`` or ``fisher_check``."""
    return sha_int(f"{SELECTION_SALT}|{tag}|{dictionary_name}") % (2**31)


def default_device() -> torch.device:
    """CUDA when available, else the CPU (never MPS)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- module inputs and output gradients
class Capture:
    """Forward hooks on the given ``nn.Linear`` modules: their input ``x`` and the gradient ``dy`` at their output.
    The token-embedding output is made a leaf that requires grad, so a backward reaches every module while the
    model's own parameters stay frozen."""

    def __init__(self, model: nn.Module, modules: Sequence[str]) -> None:
        self.x: dict[str, Tensor] = {}
        self.dy: dict[str, Tensor] = {}
        self.handles = [model.get_submodule("wte").register_forward_hook(self._emb)]
        for m in modules:
            self.handles.append(model.get_submodule(m).register_forward_hook(self._make(m)))

    @staticmethod
    def _emb(_mod, _inp, out):
        return out if out.requires_grad else out.detach().requires_grad_(True)

    def _make(self, m: str):
        def hook(_mod, inp, out):
            self.x[m] = inp[0].detach()
            if out.requires_grad:
                out.register_hook(lambda g: self.dy.__setitem__(m, g.detach()))
        return hook

    def __enter__(self) -> Capture:
        return self

    def __exit__(self, *exc) -> bool:
        for h in self.handles:
            h.remove()
        return False


def per_seq_projection(x: Tensor, dy: Tensor, U: Tensor, V: Tensor) -> Tensor:
    """``[B, C]``: ``sum_t (u_c . dy_bt)(v_c . x_bt)`` for ``x [B, T, d_in]``, ``dy [B, T, d_out]``, ``U [C, d_out]``,
    ``V [d_in, C]``."""
    return ((dy @ U.T) * (x @ V)).sum(1)


def factors_on(dictionary: Dictionary, modules: Sequence[str], device,
               dtype: torch.dtype) -> dict[str, tuple[Tensor, Tensor]]:
    return {m: (dictionary[m].U.to(device=device, dtype=dtype), dictionary[m].V.to(device=device, dtype=dtype))
            for m in modules}


# --------------------------------------------------------------------------- g and q
def target_weight_gradient(model: nn.Module, modules: Sequence[str], tokens: Tensor, device,
                           bs: int = BATCH) -> dict[str, Tensor]:
    """``G_t`` per module (float64, CPU): gradient of the mean CE over all predictions of ``tokens`` at W_0."""
    acc = dict.fromkeys(modules)
    n_pred = 0
    with Capture(model, modules) as cap:
        for s in range(0, tokens.shape[0], bs):
            w = tokens[s:s + bs].to(device)
            logits = logits_of(model(w[:, :-1]))
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), w[:, 1:].reshape(-1), reduction="sum").backward()
            for m in modules:
                x, dy = cap.x[m], cap.dy[m]
                G = (dy.reshape(-1, dy.shape[-1]).T @ x.reshape(-1, x.shape[-1])).double().cpu()
                acc[m] = G if acc[m] is None else acc[m] + G
            n_pred += w[:, 1:].numel()
    return {m: G / n_pred for m, G in acc.items()}


def atom_gradients(dictionary: Dictionary, G: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """``g_c = u_c^T G v_c`` (atoms at their own norm) for every atom of every module in ``G``."""
    return {m: dictionary[m].correlations(G[m], normalized=False) for m in G}


def sample_labels(probs: Tensor, window_ids, k: int, seed: int) -> Tensor:
    """``[B, T]`` labels drawn from ``probs [B, T, V]``. Window ``i`` uses a generator on ``probs.device`` seeded by
    ``(seed, i, k)`` only."""
    out = []
    for b, i in enumerate(window_ids):
        label_seed = sha_int(f"{SELECTION_SALT}|labels|{seed}|{int(i)}|{k}") % (2**63 - 1)
        gen = torch.Generator(device=probs.device).manual_seed(label_seed)
        out.append(torch.multinomial(probs[b], 1, generator=gen).squeeze(-1))
    return torch.stack(out)


def retain_fisher(model: nn.Module, dictionary: Dictionary, modules: Sequence[str], tokens: Tensor, device,
                  K: int = FISHER_K, seed: int = 0, bs: int = BATCH) -> dict:
    """Sampled-label Fisher ``q_c`` per prediction, and the two half estimates on even and odd windows."""
    dtype = next(model.parameters()).dtype
    fac = factors_on(dictionary, modules, device, dtype)
    acc = {m: torch.zeros(dictionary[m].C, dtype=torch.float64) for m in modules}
    half = {m: torch.zeros(2, dictionary[m].C, dtype=torch.float64) for m in modules}
    n_half = [0, 0]
    n_pred = 0
    with Capture(model, modules) as cap:
        for s in range(0, tokens.shape[0], bs):
            w = tokens[s:s + bs].to(device)
            B, T = w.shape[0], w.shape[1] - 1
            lp = F.log_softmax(logits_of(model(w[:, :-1])), -1)
            probs = lp.detach().exp().reshape(-1, lp.shape[-1])
            parity = torch.arange(s, s + B) % 2
            for k in range(K):
                y = sample_labels(probs.view(B, T, -1), range(s, s + B), k, seed)
                (-lp.gather(-1, y.unsqueeze(-1)).sum()).backward(retain_graph=k < K - 1)
                for m in modules:
                    sq = per_seq_projection(cap.x[m], cap.dy[m], *fac[m]).double().pow(2).cpu()
                    acc[m] += sq.sum(0)
                    for h in (0, 1):
                        half[m][h] += sq[parity == h].sum(0)
            n_pred += B * T
            for h in (0, 1):
                n_half[h] += int((parity == h).sum()) * T
    q = {m: v / (n_pred * K) for m, v in acc.items()}
    q_half = {m: torch.stack([half[m][h] / max(1, n_half[h] * K) for h in (0, 1)]) for m in modules}
    return {"q": q, "q_half": q_half, "n_pred": n_pred, "K": K, "seed": seed, "n_windows": int(tokens.shape[0])}


# --------------------------------------------------------------------------- score and ranking
def nonfinite_atoms(dictionary: Dictionary, vals: Mapping[str, Tensor], what: str) -> str | None:
    """The reason when ``vals`` is NaN or infinite on any usable atom, else None."""
    bad = {m: int((~torch.isfinite(vals[m]))[dictionary[m].usable].sum()) for m in vals}
    n = sum(bad.values())
    if not n:
        return None
    worst = max(bad, key=bad.get)
    return f"non-finite {what} on {n} usable atom(s) (most in {worst}: {bad[worst]})"


def require_finite(dictionary: Dictionary, named: Mapping[str, Mapping[str, Tensor]], stage: str) -> None:
    why = [r for what, vals in named.items() if (r := nonfinite_atoms(dictionary, vals, what))]
    if why:
        raise FloatingPointError(f"technical: {stage}: " + "; ".join(why))


def ridge_floor(dictionary: Dictionary, q: Mapping[str, Tensor], percentile: float = EPS_PERCENTILE) -> float:
    """``eps``: the given percentile of ``q_c / ||A_c||_F^2`` over usable atoms."""
    r = torch.cat([(q[m] / dictionary[m].norms.pow(2))[dictionary[m].usable] for m in q])
    if not bool(torch.isfinite(r).all()):
        raise FloatingPointError(f"technical: ridge floor: {int((~torch.isfinite(r)).sum())} non-finite "
                                 "q / ||A||^2 value(s) over usable atoms")
    eps = float(torch.quantile(r, percentile / 100.0))
    if not math.isfinite(eps):
        raise FloatingPointError(f"technical: ridge floor is {eps}")
    return eps


def scores(dictionary: Dictionary, g: Mapping[str, Tensor], q: Mapping[str, Tensor], eps: float) -> dict[str, Tensor]:
    """``g^2 / (q + eps ||A||^2)``. Unusable atoms get -1 (never ranked)."""
    out = {}
    for m in g:
        s = g[m].pow(2) / (q[m] + eps * dictionary[m].norms.pow(2))
        s[~dictionary[m].usable] = -1.0
        out[m] = s
    return out


def ranking(score: Mapping[str, Tensor], modules: Sequence[str],
            top: int | None = None) -> list[tuple[str, int, float]]:
    """``(module, index, score)`` in descending score. Ties by module order, then index."""
    rows = []
    for mi, m in enumerate(modules):
        s = score[m]
        for c in range(s.numel()):
            if float(s[c]) >= 0:
                rows.append((-float(s[c]), mi, c))
    rows.sort()
    rows = rows if top is None else rows[:top]
    return [(modules[mi], c, -neg) for neg, mi, c in rows]


def shortlist_order(dictionary: Dictionary, g, q, q_check, eps: float, ranked, n: int):
    """Top ``n`` of ``ranked``, re-sorted by ``g^2 / (max(q, q_check) + eps ||A||^2)`` in Python floats. Ties keep the
    ranking order. Returns (shortlist, walk order) as ``(module, index, score)`` lists."""
    short = list(ranked[:n])
    rows = []
    for r, (m, c, _) in enumerate(short):
        n2 = float(dictionary[m].norms[c]) ** 2
        cons = float(g[m][c]) ** 2 / (max(float(q[m][c]), float(q_check[m][c])) + eps * n2)
        rows.append((-cons, r, m, c))
    rows.sort()
    return short, [(m, c, -neg) for neg, _, m, c in rows]


# --------------------------------------------------------------------------- reachability walk


@torch.no_grad()
def reachability_walk(model: nn.Module, base: Mapping[str, Tensor], dictionary: Dictionary,
                      ranked: Sequence[tuple[str, int, float]], g: Mapping[str, Tensor], tokens: Tensor, device,
                      taus: Sequence[float] = TAUS, grid: Sequence[float] = ATOM_GRID, max_walk: int = MAX_WALK,
                      bs: int = BATCH) -> dict:
    """Walk ``ranked`` until a candidate's mean CE rise on ``tokens`` reaches ``max(taus)``. ``base`` holds W_0."""
    tau_lo, tau_hi = min(taus), max(taus)
    with applied_update(model, base, None):
        ce0 = doc_ce(model, tokens, device, bs)
    rows, n_fwd = [], 1
    accepted, accepted_lo = None, None
    for rank, (m, c, sc) in enumerate(ranked[:max_walk]):
        gc = float(g[m][c])
        row = {"rank": rank, "module": m, "index": c, "label_key": label_key(m, c), "score": sc, "g": gc,
               "norm": float(dictionary[m].norms[c]), "curve": [], "reach_lo": None, "reach_hi": None}
        if gc == 0.0:
            row["skipped"] = "g == 0"
            rows.append(row)
            continue
        sign = 1.0 if gc > 0 else -1.0
        row["sign"] = sign
        for a in grid:
            with applied_update(model, base, atom_update(dictionary, m, c, sign * a, device)):
                d = float((doc_ce(model, tokens, device, bs) - ce0).mean())
            n_fwd += 1
            row["curve"].append((a, d, abs(gc) * a))  # (a, measured rise, first-order prediction)
            if not math.isfinite(d):
                row["nonfinite_at"] = a
                break
            if row["reach_lo"] is None and d >= tau_lo:
                row["reach_lo"] = a
            if d >= tau_hi:
                row["reach_hi"] = a
                break
        rows.append(row)
        if "nonfinite_at" in row:  # a technical failure, not "does not reach": the walk stops
            return {"status": f"technical: non-finite rise (rank {rank}, a={row['nonfinite_at']})", "accepted": None,
                    "technical_failure": True, "taus_reachable": [], "walk": rows, "n_candidates": len(rows),
                    "n_forward_passes": n_fwd, "n_windows": int(tokens.shape[0])}
        if accepted_lo is None and row["reach_lo"] is not None:
            accepted_lo = row
        if row["reach_hi"] is not None:
            accepted = row
            break
    status = "feasible" if accepted else ("tau_lo only" if accepted_lo else "no feasible atom")
    pick = accepted or accepted_lo
    return {"status": status, "accepted": None if pick is None else {k: pick[k] for k in
            ("rank", "module", "index", "label_key", "score", "g", "sign", "norm", "reach_lo", "reach_hi")},
            "taus_reachable": sorted(taus) if accepted else ([tau_lo] if accepted_lo else []),
            "walk": rows, "n_candidates": len(rows), "n_forward_passes": n_fwd, "n_windows": int(tokens.shape[0])}


# --------------------------------------------------------------------------- curvature-check gate


# --------------------------------------------------------------------------- provenance of stored records


# --------------------------------------------------------------------------- q and selection


def q_provenance(model: nn.Module, dictionary: Dictionary, device, bs: int, inputs: Mapping[str, Tensor], recipe: dict | None = None) -> dict:
    settings = {} if recipe is None else recipe["selection"]
    name = dictionary.name
    return runtime.provenance(model, device, bs, inputs, runtime.code_identity(), dictionary=dictionary.meta, fisher_k=settings.get("fisher_k", FISHER_K),
                      eps_percentile=settings.get("eps_percentile", EPS_PERCENTILE),
                      seeds={"q": estimator_seed("q", name), "q_check": estimator_seed("qcheck", name)})


def q_record(path: str | Path, model: nn.Module, dictionary: Dictionary, retain: Tensor, qcheck: Tensor, device,
             gate: str | Path, bs: int = BATCH, *, recipe: dict | None = None) -> dict:
    """q, q_check and eps of one dictionary (shared by every target and draw), stored at ``path`` (``.pt``, with a
    ``.json`` summary of the returned record, rewritten on every call) and reused only under equal provenance
    (``stored``)."""
    settings = {} if recipe is None else recipe["selection"]
    fisher_k = settings.get("fisher_k", FISHER_K)
    percentile = settings.get("eps_percentile", EPS_PERCENTILE)
    runtime.require_gate(gate, fisher_k, eps_percentile=settings.get("eps_percentile", EPS_PERCENTILE),
                         null_seed=settings.get("null_seeds", NULL_SEED)[0])
    disable_tf32()
    name = dictionary.name
    prov = q_provenance(model, dictionary, device, bs, {"retain": retain, "qcheck": qcheck}, recipe)
    seeds = prov["seeds"]

    def compute() -> dict:
        t0 = time.time()
        modules = list(dictionary.modules)
        rec = retain_fisher(model, dictionary, modules, retain, device, K=fisher_k, seed=seeds["q"], bs=bs)
        chk = retain_fisher(model, dictionary, modules, qcheck, device, K=fisher_k, seed=seeds["q_check"], bs=bs)
        require_finite(dictionary, {"q": rec["q"], "q_check": chk["q"]}, f"q of {name}")
        return rec | {"q_check": chk["q"], "q_check_seed": chk["seed"], "q_check_n_windows": chk["n_windows"],
                      "eps": ridge_floor(dictionary, rec["q"], percentile), "eps_percentile": percentile, "dict": name,
                      "n_usable": sum(int(dictionary[m].usable.sum()) for m in modules), "gate": runtime.code_identity(),
                      "seconds": time.time() - t0}

    rec = runtime.stored(path, prov, compute)
    Path(path).with_suffix(".json").write_text(
        json.dumps({k: v for k, v in rec.items() if k not in ("q", "q_half", "q_check")}, indent=1))
    return rec


def target_gradient(model: nn.Module, modules: Sequence[str], tokens: Tensor, device, bs: int = BATCH,
                    cache: str | Path | None = None) -> tuple[dict[str, Tensor], dict]:
    """``G_t`` of a draw, reused from ``cache`` only when its provenance equals the current one. Otherwise it is
    recomputed and the cache rewritten. The VPD and the randomized dictionary of a cell share it."""
    disable_tf32()
    prov = runtime.provenance(model, device, bs, {"target": tokens}, runtime.code_identity(), modules=list(modules))
    note = "computed"
    if cache is not None and Path(cache).exists():
        rec = torch.load(cache, weights_only=True)
        if rec["provenance"] == prov:
            return rec["G"], {"cache": "reused", "provenance": prov}
        note = "recomputed: the cached gradient differs in " + ", ".join(runtime._differs(prov, rec["provenance"]))
    G = target_weight_gradient(model, modules, tokens, device, bs)
    bad = [m for m, v in G.items() if not bool(torch.isfinite(v).all())]
    if bad:
        raise FloatingPointError(f"technical: non-finite target gradient in {bad[:4]} ({len(bad)} module(s))")
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"G": G, "provenance": prov}, cache)
    return G, {"cache": note, "provenance": prov}


def select(model: nn.Module, dictionary: Dictionary, q_rec: dict, tokens: Tensor, device, gate: str | Path,
           bs: int = BATCH, g_cache: str | Path | None = None, n_short: int | None = None,
           grid: Sequence[float] | None = None, eps: float | None = None, *,
           recipe: dict | None = None) -> tuple[dict, dict]:
    """The selection rule on one target draw: (record, ``{"g", "score"}`` per module). ``q_rec`` comes from
    ``q_record``. It must have been computed by the current code for the same model and dictionary, on the same
    device and runtime (its pools and batch size are its own). ``grid`` and ``eps`` (default: the ridge floor of
    ``q_rec``) are changed only by the additional whitened comparator."""
    settings = {} if recipe is None else recipe["selection"]
    fisher_k = settings.get("fisher_k", FISHER_K)
    n_short = settings.get("max_walk", MAX_WALK) if n_short is None else n_short
    taus = TAUS if recipe is None else recipe["taus"]
    grid = settings.get("atom_grid", ATOM_GRID) if grid is None else grid
    gate_rec = runtime.require_gate(gate, fisher_k, eps_percentile=settings.get("eps_percentile", EPS_PERCENTILE),
                         null_seed=settings.get("null_seeds", NULL_SEED)[0])
    disable_tf32()
    want = q_provenance(model, dictionary, device, bs, {}, recipe)
    # production guard of this implementation, not one of the dated gate criteria: the check ran on the backend,
    # hardware and runtime of this selection (its dtype, batch and device index are its own)
    other = [k for k, v in runtime.runtime_identity(device).items() if gate_rec["provenance"].get(k) != v]
    if other:
        raise runtime.GateError(f"the curvature check at {gate} ran on another backend, hardware or runtime than this "
                        f"selection ({', '.join(other)})")
    differs = [k for k in runtime._differs(want, q_rec["provenance"]) if k not in ("inputs", "batch_size")]
    if differs:
        raise runtime.CacheMismatch(f"the q record of {q_rec['dict']} was computed under other conditions than this "
                            f"selection ({', '.join(differs)})")
    t0 = time.time()
    modules = list(dictionary.modules)
    G, g_info = target_gradient(model, modules, tokens, device, bs, g_cache)
    g = atom_gradients(dictionary, G)
    stage = f"select {dictionary.name}"
    eps = q_rec["eps"] if eps is None else eps
    if not math.isfinite(eps):
        raise FloatingPointError(f"technical: {stage}: ridge floor eps is {eps}")
    require_finite(dictionary, {"g": g, "q": q_rec["q"], "q_check": q_rec["q_check"]}, stage)
    sc = scores(dictionary, g, q_rec["q"], eps)
    require_finite(dictionary, {"score": sc}, stage)
    ranked = ranking(sc, modules)
    short, order = shortlist_order(dictionary, g, q_rec["q"], q_rec["q_check"], eps, ranked, n_short)
    walk = reachability_walk(model, capture_base(model, modules), dictionary, order, g, tokens, device, taus=taus, grid=grid,
                             max_walk=n_short, bs=bs)
    rank_q = {(m, c): r for r, (m, c, _) in enumerate(short)}
    for row in walk["walk"]:
        m, c = row["module"], row["index"]
        row.update(rank_by_q=rank_q[(m, c)], q=float(q_rec["q"][m][c]), q_check=float(q_rec["q_check"][m][c]))
    if walk["accepted"] is not None:
        walk["accepted"]["rank_by_q"] = rank_q[(walk["accepted"]["module"], walk["accepted"]["index"])]
    rec = {"dict": dictionary.name, "dictionary": dictionary.meta, "n_pred": int(tokens[:, 1:].numel()),
           "eps": eps, "eps_percentile": settings.get("eps_percentile", EPS_PERCENTILE), "grid": list(grid),
           "a_max": max(grid), "taus": taus, "fisher_k": fisher_k,
           "walk_order_rule": "top MAX_WALK by g^2/(q + eps||A||^2), re-sorted by g^2/(max(q, q_check) + eps||A||^2)",
           "top64": [(m, c, s, label_key(m, c), float(g[m][c]), float(q_rec["q"][m][c]), float(dictionary[m].norms[c]))
                     for m, c, s in ranked[:64]],
           **walk, "g_cache": g_info, "gate": {"code": gate_rec["code"], "device": gate_rec["device"]},
           "seconds": time.time() - t0}
    return rec, {"g": g, "score": sc}
