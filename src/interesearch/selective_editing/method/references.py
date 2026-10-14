"""The reference rules: OMP top-1 of a learned update and the causal-importance difference (``ci_difference``).
Rules: papers/selective-editing/index.qmd#sec-comparators (reference rows) and papers/selective-editing/index.qmd#sec-setup.

* OMP top-1: per module the correlations ``b_c = <A_c / ||A_c||, D>_F`` with the update ``D`` (float64). The usable
  atom with the largest ``|b_c|``, the first such atom within a module and, across modules, the first module in the
  update's order (strict ``>``). The edit's sign is the sign of ``b_c``. ``energy_k1`` is the share of ``||D||_F^2``
  explained by the least-squares multiple of that one atom.
* ``ci_difference``: mean causal importance of every usable atom on the target draw minus its mean on the English
  retain pool, over the 256 input positions of each window. Top-1, ties by module order, then index. Ablation side
  (sign -1). The top 16 and the firing rates (CI > 0) are recorded, not used. CI is the decomposition's
  ``lower_leaky`` importance with continuous sampling. The CI function attends bidirectionally, so a position's CI
  depends on its whole window. It runs in a separate reference env with param-decomp (Goodfire) at the commit of
  the published decomposition. It needs about 6 GB of RAM:

    git clone https://github.com/goodfire-ai/param-decomp vendor/param-decomp
    git -C vendor/param-decomp checkout 74146b555f3eb3cd2bc639c3789fd3b7a099094a
    (cd vendor/param-decomp && uv sync --frozen --no-install-project --no-dev)
    PYTHONPATH=src:vendor/param-decomp vendor/param-decomp/.venv/bin/python \\
        -m interesearch.selective_editing.method.references ci assets --target draw.npy --english retain_en.npy \\
        --english-cache ci_en.pt --out ci.json

No W&B access: the decomposition is loaded from the asset files through a layout of symbolic links and a copy of
its ``final_config.yaml`` whose ``pretrained_model_name`` points at the local target checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from interesearch.assets import ASSETS, MODULES, check
from interesearch.checksums import sha256_file
from interesearch.selective_editing.method.dictionary import Dictionary, label_key, load_dictionary
from interesearch.selective_editing.method.runtime import (
    config_digest,
    provenance,
    source_identity,
    stored,
    tokens_identity,
)
from interesearch.selective_editing.method.selector import default_device

Tensor = torch.Tensor

CI_BATCH = 8
CI_TOP = 16
CI_RULE = "mean CI(target draw) - mean CI(English retain pool), top-1, ablation side"


# --------------------------------------------------------------------------- OMP top-1
def omp_top1(dictionary: Dictionary, update: Mapping[str, Tensor]) -> dict:
    """OMP top-1 of ``update`` (``{module: D}``) over the usable atoms of ``dictionary``."""
    b_all, d_sq, best = {}, {}, None
    for m, D in update.items():
        D = D.to(torch.float64)
        d_sq[m] = float((D * D).sum())
        b_all[m] = dictionary[m].correlations(D, normalized=True)
        score = b_all[m].abs().clone()
        score[~dictionary[m].usable] = -math.inf
        j = int(score.argmax())
        if math.isfinite(float(score[j])) and (best is None or float(score[j]) > best[2]):
            best = (m, j, float(score[j]))
    m, j, _ = best
    Uh, Vh = dictionary[m].unit
    S = torch.as_tensor([j])
    G_S = ((Uh @ Uh[S].T) * (Vh.T @ Vh[:, S]))[S]  # the atom's Gram entry, [1, 1]
    b_S = b_all[m][S]
    alpha = b_S / G_S[0]
    resid = dict(d_sq)
    resid[m] = max(float(d_sq[m] - 2 * alpha @ b_S + alpha @ (G_S @ alpha)), 0.0)
    total = sum(d_sq.values())
    corr = float(b_all[m][j])
    top = []
    for mm, b in b_all.items():
        b = b.clone()
        b[~dictionary[mm].usable] = 0.0
        for cc in b.abs().topk(5).indices.tolist():
            top.append((abs(float(b[cc])), mm, cc, float(b[cc])))
    top.sort(reverse=True)
    return {"module": m, "index": j, "label_key": label_key(m, j), "sign": 1.0 if corr > 0 else -1.0,
            "normalized_corr": corr, "energy_k1": 1.0 - sum(resid.values()) / total if total > 0 else 0.0,
            "top10": [{"module": mm, "index": cc, "label_key": label_key(mm, cc), "normalized_corr": v}
                      for _, mm, cc, v in top[:10]]}


# --------------------------------------------------------------------------- ci_difference (reference env)
CI_ASSETS = ("target/model_step_99999.pt", "target/model_config.yaml", "target/tokenizer.json",
             "target/final_config.yaml", "decomposition/model_400000.pth", "decomposition/final_config.yaml")


def _link(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_layout(assets_dir: str | Path, layout_dir: str | Path) -> Path:
    """The run-folder layout param-decomp reads, under ``layout_dir``. Returns the path for
    ``ComponentModel.from_pretrained``. The target's run info is read from ``ckpt.parent.parent`` and the
    decomposition's from ``pth.parent``."""
    assets, layout = Path(assets_dir), Path(layout_dir)
    t = layout / "t-9d2b8f02"
    (t / "ckpt").mkdir(parents=True, exist_ok=True)
    for name in ("final_config.yaml", "model_config.yaml", "tokenizer.json"):
        _link(t / name, assets / "target" / name)
    _link(t / "ckpt" / "model_step_99999.pt", assets / "target" / "model_step_99999.pt")
    s = layout / "s-55ea3f9b"
    s.mkdir(parents=True, exist_ok=True)
    _link(s / "model_400000.pth", assets / "decomposition" / "model_400000.pth")
    old = "pretrained_model_name: goodfire/spd/runs/t-9d2b8f02"
    text = (assets / "decomposition" / "final_config.yaml").read_text()
    if text.count(old) != 1:
        raise ValueError(f"decomposition final_config.yaml does not name the target run once: {old!r}")
    target_ckpt = t / "ckpt" / "model_step_99999.pt"
    (s / "final_config.yaml").write_text(text.replace(old, f"pretrained_model_name: {target_ckpt}"))
    return s / "model_400000.pth"


def load_component_model(assets_dir: str | Path, layout_dir: str | Path, device):
    """param-decomp's ComponentModel of the published decomposition (strict state-dict load), eval mode, frozen.
    The asset files must match their sha256. Param-decomp's output folder is kept inside ``layout_dir``."""
    bad = [p for a in ASSETS if a.path in CI_ASSETS and (p := check(assets_dir, a)) is not None]
    if bad:
        raise ValueError(f"asset files differ from their pins: {bad}")
    os.environ["PARAM_DECOMP_OUT_DIR"] = str(Path(layout_dir).resolve() / "param_decomp_out")
    from param_decomp.models.component_model import ComponentModel

    cm = ComponentModel.from_pretrained(str(build_layout(assets_dir, layout_dir)))
    return cm.eval().requires_grad_(False).to(device)


def param_decomp_identity() -> str:
    """sha256 over the source files of the imported param-decomp package (each ``.py`` file's relative path and
    sha256), so another revision or a modified checkout differs."""
    import param_decomp

    root = Path(param_decomp.__file__).parent
    return config_digest({f.relative_to(root).as_posix(): sha256_file(f) for f in sorted(root.rglob("*.py"))})


def ci_code() -> dict[str, str]:
    """The identity of the code that computes the English CI means: this file and the imported param-decomp."""
    return source_identity(__file__) | {"param_decomp": param_decomp_identity()}


def component_model_config(cm) -> dict:
    """The ComponentModel's effective configuration: the target's config, C per module, the CI function's structure
    and the sigmoids."""
    return {"target": cm.target_model.config.model_dump(mode="json"), "module_to_c": dict(cm.module_to_c),
            "ci_fn": repr(cm.ci_fn), "sigmoids": [cm.lower_leaky_fn.__name__, cm.upper_leaky_fn.__name__]}


@torch.no_grad()
def causal_importance(cm, tokens: Tensor) -> dict[str, Tensor]:
    """CI (``lower_leaky``, continuous sampling as in the run config) per module, ``[B, T, C]``, for int64 tokens."""
    out = cm(tokens.to(next(cm.parameters()).device), cache_type="input")
    return cm.calc_causal_importances(out.cache, sampling="continuous").lower_leaky


@torch.no_grad()
def mean_ci(cm, tokens: Tensor, modules=MODULES, bs: int = CI_BATCH) -> tuple[dict, dict]:
    """Mean CI and firing rate (CI > 0) per atom over every input position of ``tokens`` (float64, CPU)."""
    tot = dict.fromkeys(modules)
    fire = dict.fromkeys(modules)
    n = 0
    for s in range(0, tokens.shape[0], bs):
        w = tokens[s:s + bs, :-1]
        ci = causal_importance(cm, w)
        for m in modules:
            v = ci[m].double().sum((0, 1)).cpu()
            f = (ci[m] > 0).double().sum((0, 1)).cpu()
            tot[m] = v if tot[m] is None else tot[m] + v
            fire[m] = f if fire[m] is None else fire[m] + f
        n += w.numel()
    return {m: v / n for m, v in tot.items()}, {m: v / n for m, v in fire.items()}


def ci_difference(cm, dictionary: Dictionary, target: Tensor, english: Tensor, device, english_cache: str | Path,
                  bs: int = CI_BATCH) -> dict:
    """The ``ci_difference`` record of one target draw. The English means are stored at ``english_cache`` and
    reused only under equal provenance, which includes the param-decomp sources and the model's configuration."""
    t0 = time.time()
    prov = provenance(cm, device, bs, {"english": english}, ci_code(), component_model_config(cm),
                      dictionary=dictionary.meta)
    en = stored(english_cache, prov, lambda: dict(zip(("mean", "fire"), mean_ci(cm, english, MODULES, bs))))
    tm, tf = mean_ci(cm, target, MODULES, bs)
    rows = []
    for mi, m in enumerate(MODULES):
        diff = tm[m] - en["mean"][m]
        for c in torch.nonzero(dictionary[m].usable).flatten().tolist():
            rows.append((-float(diff[c]), mi, c))
    rows.sort()
    top = [(MODULES[mi], c, -s) for s, mi, c in rows[:CI_TOP]]
    m, c, s = top[0]
    return {"rule": CI_RULE, "module": m, "index": c, "label_key": label_key(m, c), "sign": -1.0, "score": s,
            "ci_target": float(tm[m][c]), "ci_en": float(en["mean"][m][c]), "fire_target": float(tf[m][c]),
            "fire_en": float(en["fire"][m][c]),
            "top16": [{"module": mm, "index": cc, "label_key": label_key(mm, cc), "score": ss} for mm, cc, ss in top],
            "target": tokens_identity(target), "english_provenance": prov, "device": str(torch.device(device)),
            "torch": str(torch.__version__), "seconds": time.time() - t0}


def _windows(path: Path) -> Tensor:
    return torch.from_numpy(np.load(path).astype(np.int64))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ci_difference of one target draw (param-decomp reference env).")
    ap.add_argument("command", choices=("ci",))
    ap.add_argument("assets_dir", type=Path)
    ap.add_argument("--target", type=Path, required=True, help="int windows [N, 257] of the target draw (.npy)")
    ap.add_argument("--english", type=Path, required=True, help="int windows of the English retain pool (.npy)")
    ap.add_argument("--english-cache", type=Path, required=True, help="stored English means (.pt)")
    ap.add_argument("--out", type=Path, required=True, help="record to write (.json)")
    ap.add_argument("--layout", type=Path, default=None, help="default: <assets_dir>/official_layout")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--bs", type=int, default=CI_BATCH)
    a = ap.parse_args(argv)
    device = torch.device(a.device) if a.device else default_device()
    cm = load_component_model(a.assets_dir, a.layout or a.assets_dir / "official_layout", device)
    rec = ci_difference(cm, load_dictionary(a.assets_dir), _windows(a.target), _windows(a.english), device,
                        a.english_cache, a.bs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(a.out, rec["label_key"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
