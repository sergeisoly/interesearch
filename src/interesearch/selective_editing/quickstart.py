"""See how one German-targeted weight edit changes predictions in German, Dutch and English.
Rules: papers/selective-editing/index.qmd#sec-setup and papers/selective-editing/index.qmd#sec-calibration.

The atom is the de draw-0 selection of the VPD dictionary, component 268 of ``h.0.attn.o_proj`` with sign -1 (the es
and fr draw-0 selections are components 676 and 640 of the same module). Its dose ``a`` at the effect point ``--tau``
is the matched dose of the published calibration record. The edit adds ``sign a A_c = -a u v^T`` to the full weight
``W_0``: the outer product and the scaling in float64, the update cast to float32 and added to the float32 weight. The
atom's mask coefficient changes by ``delta_scale = -a`` and becomes ``resulting_mask = 1 - a``.

On the windows of ``data/quickstart`` (role ``quickstart``: 4 Europarl session windows, the same turns in de, nl and
English) the command reports, per language, the CE of the original and of the edited model (per-window mean
next-token CE, then the mean over windows) and the KL to the original. It checks that every parameter is restored
bit-exactly after the edit.

    uv run python -m interesearch.assets fetch assets
    uv run python -m interesearch.selective_editing.dictionary extract assets
    uv run python -m interesearch.selective_editing.quickstart assets
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from ..assets import ASSETS, MODULES, check, load_target
from ..edits import assert_frozen, capture_base, snapshot_params
from ..training import disable_tf32
from .access import Windows, load_windows
from .dictionary import COMPONENTS, Dictionary, atom_update, load_dictionary, parse_label_key
from .measure import ce_docs, edit_readout

DATA = Path("data/quickstart")
CALIBRATION = Path("data/selective_editing/calibration/de_draw0.json")
ROLES = {"de": "target", "nl": "protected", "en": "English"}
DOWNLOADS = ("target/model_step_99999.pt", "target/model_config.yaml", "decomposition/model_400000.pth")


def edit(model: nn.Module, dictionary: Dictionary, windows: Windows, label: str, sign: float, a: float,
         device) -> dict:
    """The atom edit ``sign a A_c`` of ``label`` measured on every stem of ``windows``."""
    m, c = parse_label_key(label)
    update = atom_update(dictionary, m, c, sign * a, device)
    base = capture_base(model, MODULES)
    snapshot = snapshot_params(model)
    ce0 = {s: ce_docs(model, base, x, None, device) for s, x in windows.tokens.items()}
    r = edit_readout(model, base, update, windows.tokens, device, ce0)
    assert_frozen(model, snapshot)  # every parameter equals its value before the edit
    return {"atom": {"label": label, "module": m, "component": c, "sign": sign,
                     "norm": float(dictionary[m].norms[c])},
            "a": a, "delta_scale": sign * a, "resulting_mask": 1.0 + sign * a,
            "update_dtype": str(update[m].dtype), "update_norm": float(update[m].double().norm()),
            "ce": {s: {"role": ROLES.get(s, s), "original": v["ce_base"], "edited": v["ce"], "rise": v["delta_ce"],
                       "kl": v["kl"]} for s, v in r.items()},
            "weights_restored": True}


def run(assets_dir: Path, data: Path = DATA, calibration: Path = CALIBRATION, tau: float = 1.0,
        device: str = "cpu") -> dict:
    t0 = time.time()
    bad = [p for x in ASSETS if x.path.startswith("target/model") and (p := check(assets_dir, x)) is not None]
    if bad:
        raise ValueError(f"asset files differ from their pins: {bad}")
    disable_tf32()
    model = load_target(assets_dir, device)
    dictionary = load_dictionary(assets_dir)
    windows = load_windows(data, "quickstart")
    method = json.loads(Path(calibration).read_text())["methods"]["vpd_atom"]
    got = method["by_tau"][str(tau)]["recorded"]
    res = edit(model, dictionary, windows, method["atom"]["label"], method["atom"]["sign"], got["scale"], device)
    sizes = {x.path: x.size for x in ASSETS if x.path in DOWNLOADS}
    return {"tau": tau, "calibration": {"path": str(calibration), "achieved_on_target_validation": got["achieved"]},
            **res, "model": {x.path: x.sha256 for x in ASSETS if x.path.startswith("target/model")},
            "dictionary": dictionary.meta["tensor_digest"],
            "windows": {"dates": windows.groups, "sha256": windows.sha256, "tokens_per_window": 257},
            "downloads_bytes": sizes | {"total": sum(sizes.values())},
            "extracted_bytes": {COMPONENTS: (Path(assets_dir) / COMPONENTS).stat().st_size},
            "device": str(device), "torch": torch.__version__, "seconds": time.time() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="See how one German-targeted weight edit changes predictions in German, Dutch and English.")
    ap.add_argument("assets_dir", type=Path)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--calibration", type=Path, default=CALIBRATION)
    ap.add_argument("--tau", type=float, default=1.0, choices=(0.5, 1.0),
                    help="target-language CE rise used to calibrate the saved edit, in nats/token")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, help="also write the result here (.json)")
    a = ap.parse_args(argv)
    res = run(a.assets_dir, a.data, a.calibration, a.tau, a.device)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    for s, v in res["ce"].items():
        print(f"  {s} ({v['role']}): CE {v['original']:.6f} -> {v['edited']:.6f} (rise {v['rise']:+.6f}, "
              f"KL {v['kl']:.6f})")
    at = res["atom"]
    print(f"Edit: subcomponent {at['label']} ({at['module']} component {at['component']}), dose a = {res['a']:.6f}, "
          f"coefficient change {res['delta_scale']:+.6f}, resulting coefficient {res['resulting_mask']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
