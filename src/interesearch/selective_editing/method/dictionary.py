"""VPD rank-1 atoms and their randomized controls.
Rules: papers/selective-editing/index.qmd#sec-selection and
papers/selective-editing/index.qmd#sec-setup.

Orientation (param-decomp ``models/components.py``, tag vpd-paper): ``V [d_in, C]`` is the read side and
``U [C, d_out]`` the write side. Atom ``A_c = outer(U[c], V[:, c])`` has the ``nn.Linear`` weight layout and
``sum_c A_c = (V @ U).T``. U and V are held in float64 on the CPU. An atom is usable when
``||A_c||_F > 1e-12 * max_c ||A_c||_F`` within its module.

The U/V file ``decomposition/components_fp32.safetensors`` is extracted from the published decomposition
``decomposition/model_400000.pth`` (48 float32 tensors ``h.L.<module>.{U,V}``). It is pinned by its tensor digest
(sorted names, dtype, shape, tensor bytes and the metadata, which names the source file and its sha256), not by the
file bytes, so the safetensors header layout does not matter. ``extract`` checks the source sha256 first and keeps
the written file only if its digest matches the pin. ``load_dictionary`` checks the digest at every load.

    uv run python -m interesearch.selective_editing.method.dictionary extract assets
    uv run python -m interesearch.selective_editing.method.dictionary verify assets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch

from interesearch.assets import ASSETS, MODULES
from interesearch.checksums import sha256_file

Tensor = torch.Tensor

SOURCE = "decomposition/model_400000.pth"
SOURCE_SHA256 = next(a.sha256 for a in ASSETS if a.path == SOURCE)
COMPONENTS = "decomposition/components_fp32.safetensors"
TENSOR_DIGEST = "c2e02133ce4314f7727980aa21c2186f81729cb5973ae121534b954771636db2"
EXCLUSION_REL = 1e-12
# Original randomized-dictionary seed namespace, retained to reproduce the published controls.
RANDOMIZED_SALT = "vpd-diffs-randomized"

# label keys of labels.json name the MLP up-projection c_fc "mlp.up": "3.attn.v:513" <-> ("h.3.attn.v_proj", 513)
LABEL_SUFFIX = {"attn.q_proj": "attn.q", "attn.k_proj": "attn.k", "attn.v_proj": "attn.v", "attn.o_proj": "attn.o",
                "mlp.c_fc": "mlp.up", "mlp.down_proj": "mlp.down"}
MODULE_SUFFIX = {v: k for k, v in LABEL_SUFFIX.items()}


def label_key(module: str, c: int) -> str:
    """``("h.3.attn.v_proj", 513)`` -> ``"3.attn.v:513"``."""
    _, layer, suffix = module.split(".", 2)
    return f"{layer}.{LABEL_SUFFIX[suffix]}:{c}"


def parse_label_key(key: str) -> tuple[str, int]:
    """``"3.attn.v:513"`` -> ``("h.3.attn.v_proj", 513)``."""
    head, c = key.rsplit(":", 1)
    layer, short = head.split(".", 1)
    return f"h.{int(layer)}.{MODULE_SUFFIX[short]}", int(c)


# --------------------------------------------------------------------------- atoms
@dataclass
class ModuleAtoms:
    name: str
    U: Tensor  # [C, d_out]
    V: Tensor  # [d_in, C]
    usable: Tensor  # bool [C]
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.U = self.U.detach().to(torch.float64).contiguous()
        self.V = self.V.detach().to(torch.float64).contiguous()
        if self.U.ndim != 2 or self.V.ndim != 2 or self.U.shape[0] != self.V.shape[1]:
            raise ValueError(f"{self.name}: U {tuple(self.U.shape)} and V {tuple(self.V.shape)} do not agree")

    @property
    def C(self) -> int:
        return self.U.shape[0]

    @property
    def weight_shape(self) -> tuple[int, int]:
        return self.U.shape[1], self.V.shape[0]

    @property
    def norms(self) -> Tensor:
        """``||A_c||_F = ||u_c|| ||v_c||``."""
        if "norms" not in self._cache:
            self._cache["norms"] = self.U.norm(dim=1) * self.V.norm(dim=0)
        return self._cache["norms"]

    @property
    def unit(self) -> tuple[Tensor, Tensor]:
        """``(Uhat, Vhat)``: U and V with unit rows and columns (a zero vector stays zero)."""
        if "unit" not in self._cache:
            nu, nv = self.U.norm(dim=1), self.V.norm(dim=0)
            nu[nu == 0] = 1.0
            nv[nv == 0] = 1.0
            self._cache["unit"] = (self.U / nu[:, None], self.V / nv[None, :])
        return self._cache["unit"]

    def atom(self, c: int) -> Tensor:
        return torch.outer(self.U[c], self.V[:, c])

    def correlations(self, D: Tensor, normalized: bool) -> Tensor:
        """``<A_c, D>_F = u_c^T D v_c`` for every atom (of the unit atoms with ``normalized``), via ``D @ V``."""
        D = D.to(torch.float64)
        if tuple(D.shape) != self.weight_shape:
            raise ValueError(f"{self.name}: update has shape {tuple(D.shape)}, expected {self.weight_shape}")
        U, V = self.unit if normalized else (self.U, self.V)
        return ((D @ V) * U.T).sum(dim=0)


class Dictionary:
    """``ModuleAtoms`` per module in ``MODULES`` order. ``excluded`` lists the atoms below the norm threshold."""

    def __init__(self, modules: Mapping[str, ModuleAtoms], excluded: Mapping[str, list[int]], meta: dict) -> None:
        self.modules = dict(modules)
        self.excluded = dict(excluded)
        self.meta = dict(meta)

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, tuple[Tensor, Tensor]], meta: dict) -> Dictionary:
        modules, excluded = {}, {}
        for name, (U, V) in tensors.items():
            atoms = ModuleAtoms(name, U, V, torch.ones(U.shape[0], dtype=torch.bool))
            atoms.usable = atoms.norms > EXCLUSION_REL * atoms.norms.max()
            modules[name] = atoms
            excluded[name] = (~atoms.usable).nonzero().flatten().tolist()
        return cls(modules, excluded, {**meta, "exclusion_rel": EXCLUSION_REL})

    def __getitem__(self, name: str) -> ModuleAtoms:
        return self.modules[name]

    def __iter__(self):
        return iter(self.modules)

    @property
    def name(self) -> str:
        """``vpd``, ``null<seed>`` for the randomized control, or the ``name`` its meta gives (the additional
        comparators). The estimator seeds are derived from it."""
        seed = self.meta.get("randomized_seed")
        return self.meta.get("name") or ("vpd" if seed is None else f"null{seed}")


def atom_update(dictionary: Dictionary, module: str, c: int, alpha: float, device,
                dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
    """``{module: alpha * A_c}`` (outer product and scaling in float64, then cast): the atom's mask moves from 1 to
    ``1 + alpha``."""
    return {module: (float(alpha) * dictionary[module].atom(int(c))).to(device=device, dtype=dtype)}


# --------------------------------------------------------------------------- randomized control
def module_generator(seed: int, module: str) -> torch.Generator:
    """A CPU generator seeded by ``(seed, module)`` only, so the draw does not depend on module order."""
    digest = hashlib.sha256(f"{RANDOMIZED_SALT}|{int(seed)}|{module}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))


def randomized_dictionary(dictionary: Dictionary, seed: int) -> Dictionary:
    """Per module, signed permutations of the output coordinates (on U) and the input coordinates (on V). Atom norms,
    the usable mask and the exclusions are those of ``dictionary``."""
    modules = {}
    for m, a in dictionary.modules.items():
        gen = module_generator(seed, m)
        d_out, d_in = a.weight_shape
        perm_out = torch.randperm(d_out, generator=gen)
        sign_out = torch.randint(0, 2, (d_out,), generator=gen).to(torch.float64) * 2 - 1
        perm_in = torch.randperm(d_in, generator=gen)
        sign_in = torch.randint(0, 2, (d_in,), generator=gen).to(torch.float64) * 2 - 1
        modules[m] = ModuleAtoms(m, a.U[:, perm_out] * sign_out[None, :], a.V[perm_in, :] * sign_in[:, None],
                                 a.usable.clone())
    return Dictionary(modules, dictionary.excluded, {**dictionary.meta, "randomized_seed": int(seed)})


# --------------------------------------------------------------------------- the U/V file
def _header(f) -> tuple[int, dict]:
    n = struct.unpack("<Q", f.read(8))[0]
    return n, json.loads(f.read(n))


def tensor_digest(path: str | Path) -> str:
    """sha256 over a safetensors file's metadata and its tensors (sorted names, dtype, shape, bytes)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n, hdr = _header(f)
        h.update(json.dumps(hdr.pop("__metadata__", {}), sort_keys=True).encode())
        for k in sorted(hdr):
            a, b = hdr[k]["data_offsets"]
            f.seek(8 + n + a)
            h.update(json.dumps([k, hdr[k]["dtype"], hdr[k]["shape"]]).encode())
            h.update(f.read(b - a))
    return h.hexdigest()


def components_metadata(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return _header(f)[1].get("__metadata__", {})


def load_dictionary(assets_dir: str | Path) -> Dictionary:
    """The VPD dictionary from ``<assets_dir>/decomposition/components_fp32.safetensors``, refused unless its tensor
    digest matches the pin."""
    from safetensors.torch import load_file

    path = Path(assets_dir) / COMPONENTS
    digest = tensor_digest(path)
    if digest != TENSOR_DIGEST:
        raise ValueError(f"{path}: tensor digest {digest} differs from the pinned {TENSOR_DIGEST}")
    sd = load_file(str(path))
    return Dictionary.from_tensors({m: (sd[f"{m}.U"], sd[f"{m}.V"]) for m in MODULES}, {"tensor_digest": digest})


def extract(assets_dir: str | Path) -> Path:
    """Write the U/V file from the decomposition checkpoint (target copy, U/V and CI function, 159 float32 tensors).

    The source must match its sha256. Its keys are the current param-decomp names (``_components.h-L-<module>.U``).
    The file is written next to the source as ``.part`` and renamed only if its tensor digest matches the pin.
    """
    from safetensors.torch import save_file

    src, out = Path(assets_dir) / SOURCE, Path(assets_dir) / COMPONENTS
    if sha256_file(src) != SOURCE_SHA256:
        raise ValueError(f"{src}: sha256 differs from the expected {SOURCE_SHA256}")
    sd = torch.load(src, map_location="cpu", mmap=True, weights_only=True)
    tensors = {}
    for m in MODULES:
        key = "_components." + m.replace(".", "-")
        U, V, W = sd[f"{key}.U"], sd[f"{key}.V"], sd[f"target_model.{m}.weight"]
        if U.dtype != torch.float32 or V.dtype != torch.float32 or U.shape[0] != V.shape[1] \
                or (U.shape[1], V.shape[0]) != tuple(W.shape):
            raise ValueError(f"{m}: U {U.dtype} {tuple(U.shape)}, V {V.dtype} {tuple(V.shape)}, W {tuple(W.shape)}")
        tensors[f"{m}.U"] = U.contiguous().clone()
        tensors[f"{m}.V"] = V.contiguous().clone()
    groups = {"target_model.": 0, "_components.": 0, "ci_fn.": 0}
    numel = dict.fromkeys(groups, 0)
    for k, v in sd.items():
        for g in groups:
            if k.startswith(g):
                groups[g] += 1
                numel[g] += 0 if k == "target_model.lm_head.weight" else v.numel()  # tied to wte: counted once
    meta = {"source": Path(SOURCE).name, "source_sha256": SOURCE_SHA256, "wandb_run": "goodfire/spd/s-55ea3f9b",
            "layout": "U: (C, d_out) write side; V: (d_in, C) read side; component weight (V @ U).T",
            "group_counts": json.dumps(groups), "group_numel": json.dumps(numel)}
    part = out.with_suffix(out.suffix + ".part")
    save_file(tensors, str(part), metadata=meta)
    digest = tensor_digest(part)
    if digest != TENSOR_DIGEST:
        part.unlink()
        raise ValueError(f"extracted tensor digest {digest} differs from the pinned {TENSOR_DIGEST}. Deleted it")
    part.rename(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract or verify the decomposition's U/V file.")
    ap.add_argument("command", choices=("extract", "verify"))
    ap.add_argument("assets_dir", type=Path, help="assets directory, e.g. assets")
    a = ap.parse_args(argv)
    path = a.assets_dir / COMPONENTS
    if a.command == "extract":
        t0 = time.time()
        extract(a.assets_dir)
        print(f"wrote {path} ({path.stat().st_size:,} B) in {time.time() - t0:.0f} s")
    digest = tensor_digest(path)
    meta = components_metadata(path)
    ok = digest == TENSOR_DIGEST and meta.get("source_sha256") == SOURCE_SHA256
    print(f"{'ok ' if ok else 'BAD'} {path}: tensor digest {digest}, source sha256 {meta.get('source_sha256')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
