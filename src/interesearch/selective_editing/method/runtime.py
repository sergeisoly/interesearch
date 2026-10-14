"""Curvature-check validity and provenance of reusable numerical records."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from pathlib import Path
from typing import Callable, Mapping
import torch
import torch.nn as nn
from interesearch.checksums import sha256_file

Tensor = torch.Tensor
FISHER_K = 8
GATE_FILES = ("selector.py", "fisher.py", "dictionary.py", "../../model.py")
GATE_WINDOWS, GATE_K_BIG = 8, 32
WINDOW = 257

class GateError(RuntimeError):
    """The curvature check is missing, failed, stale or too small."""


class CacheMismatch(RuntimeError):
    """A stored record was computed under other conditions than the current call."""


def source_identity(*paths: str | Path) -> dict[str, str]:
    """sha256 of source files, read now, by file name."""
    return {Path(p).name: sha256_file(p) for p in paths}


def code_identity() -> dict[str, str]:
    """sha256 of the q estimator, its check, the dictionary and the model forward."""
    return source_identity(*(Path(__file__).parent / f for f in GATE_FILES))


def require_gate(path: str | Path, fisher_k: int = FISHER_K, *, eps_percentile: float = 1.0,
                 null_seed: int = 101) -> dict:
    """Require the current curvature computation, its three selection settings, and the full check size.

    Records without settings must be regenerated; they are never attributed to the caller's current recipe.
    """
    path = Path(path)
    if not path.is_file():
        raise GateError(f"no curvature-check record at {path}. Run interesearch.selective_editing.method.fisher first")
    rec = json.loads(path.read_text())
    if rec.get("passed") is not True:
        raise GateError(f"curvature check at {path} did not pass: {rec.get('failures')}")
    if rec.get("code") != code_identity():
        raise GateError(f"curvature check at {path} was made with other code than the current {GATE_FILES}")
    if rec.get("fisher_k") != fisher_k:
        raise GateError(f"curvature check at {path} used K={rec.get('fisher_k')}, the estimator uses {fisher_k}")
    expected = {"fisher_k": fisher_k, "eps_percentile": eps_percentile, "null_seed": null_seed}
    if rec.get("selection_settings") != expected:
        raise GateError(f"curvature check at {path} has different or missing selection settings "
                        f"({rec.get('selection_settings')}); expected {expected}. Run the Fisher check again")
    if (rec.get("n_windows", 0) < GATE_WINDOWS or rec.get("window") != WINDOW or rec.get("k_big", 0) < GATE_K_BIG
            or rec.get("dtype") != "float64"):
        raise GateError(f"curvature check at {path} is too small or not float64 (n_windows {rec.get('n_windows')}, "
                        f"window {rec.get('window')}, k_big {rec.get('k_big')}, dtype {rec.get('dtype')}. Needs >= "
                        f"{GATE_WINDOWS} windows of {WINDOW} tokens x {GATE_K_BIG}, float64)")
    return rec


def weights_digest(model: nn.Module) -> str:
    """sha256 over the model's state dict (names, dtype, shape, bytes on the CPU)."""
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        t = t.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.flatten().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tokens_identity(tokens: Tensor) -> dict:
    t = tokens.to(torch.int64).contiguous()
    return {"sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(), "shape": list(t.shape)}


def config_digest(config: Mapping) -> str:
    """sha256 of a configuration's canonical JSON."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime_identity(device) -> dict:
    """Backend, hardware and runtime of ``device``. The labels of q come from ``torch.multinomial`` on the device,
    whose CUDA draws depend on the number of streaming multiprocessors."""
    dev = torch.device(device)
    cuda = dev.type == "cuda"
    return {"label_generator": dev.type,
            "device_name": torch.cuda.get_device_name(dev) if cuda else platform.machine(),
            "multi_processor_count": torch.cuda.get_device_properties(dev).multi_processor_count if cuda else None,
            "torch": str(torch.__version__), "cuda": torch.version.cuda}


def provenance(model: nn.Module, device, bs: int, inputs: Mapping[str, Tensor], code: Mapping[str, str],
               model_config: Mapping | None = None, **config) -> dict:
    """The conditions a stored record was computed under, ``code`` the identity of the code that computes it and
    ``model_config`` the model's effective configuration (default: its dataclass ``config``). Equal provenance is
    the only reason to reuse it."""
    if model_config is None:
        model_config = dataclasses.asdict(model.config)
    return {"code": dict(code), "device": str(torch.device(device)), **runtime_identity(device),
            "dtype": str(next(model.parameters()).dtype),
            "tf32": {"cuda_matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "batch_size": bs,
            "model": weights_digest(model), "model_config": config_digest(model_config),
            "inputs": {k: tokens_identity(t) for k, t in inputs.items()}, **config}


def _differs(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def stored(path: str | Path, prov: dict, compute: Callable[[], dict]) -> dict:
    """The record at ``path`` when its provenance equals ``prov``. With no file, ``compute()`` plus ``prov``, saved
    there. A record made under other conditions raises ``CacheMismatch`` and the file stays as it is."""
    path = Path(path)
    if path.exists():
        rec = torch.load(path, weights_only=True)
        if rec["provenance"] != prov:
            raise CacheMismatch(f"{path} was computed under other conditions "
                                f"({', '.join(_differs(prov, rec['provenance']))}). Remove it to recompute")
        return rec
    rec = compute() | {"provenance": prov}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, path)
    return rec
