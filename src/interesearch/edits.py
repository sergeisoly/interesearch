"""Temporary weight edits and frozen-parameter checks.

An edit writes ``W_m = W_0,m + Q_m`` (sum in the weight's dtype, then ``copy_``) into the live parameters and
restores ``W_0`` bit-exactly on exit, also when the body raises. Inside an edit, ``reference()`` runs a block with
``W_0`` and writes the edit back by the same formula. One model carries at most one active edit: a second
``applied_update`` on it, from any caller, raises instead of silently replacing the first edit's weights.
The active edit is recorded on the model object.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Mapping

import torch
import torch.nn as nn

Tensor = torch.Tensor


def _weight(model: nn.Module, module: str) -> nn.Parameter:
    return model.get_submodule(module).weight


def capture_base(model: nn.Module, modules: Iterable[str]) -> dict[str, Tensor]:
    """Exact clones of the live weights (same device and dtype) to restore from."""
    if getattr(model, "_active_edit", None) is not None:
        raise RuntimeError("an edit is active on this model: its weights are not W_0")
    return {m: _weight(model, m).detach().clone() for m in modules}


class ActiveEdit:
    """The edit currently written into a model. Yielded by ``applied_update``. ``update`` holds each ``Q`` already
    cast to its weight's device and dtype."""

    def __init__(self, model: nn.Module, base: Mapping[str, Tensor], update: Mapping[str, Tensor]) -> None:
        self.model, self.base, self.update = model, base, update
        self._in_reference = False

    def _write(self) -> None:
        """``W_0 + Q`` for the modules in the update, ``W_0`` for the other modules in base."""
        with torch.no_grad():
            for m, W0 in self.base.items():
                w = _weight(self.model, m)
                w.copy_(W0 + self.update[m] if m in self.update else W0)

    @contextmanager
    def reference(self) -> Iterator[nn.Module]:
        """Run the block with ``W_0`` (the reference forward), then write the edit back."""
        if getattr(self.model, "_active_edit", None) is not self or self._in_reference:
            raise RuntimeError("reference() needs this edit active on its model and not already in a reference block")
        self._in_reference = True
        try:
            with torch.no_grad():
                for m, W0 in self.base.items():
                    _weight(self.model, m).copy_(W0)
            yield self.model
        finally:
            try:
                self._write()
            finally:
                self._in_reference = False


@contextmanager
def applied_update(model: nn.Module, base: Mapping[str, Tensor],
                   update: Mapping[str, Tensor] | None) -> Iterator[ActiveEdit]:
    """Set ``W_m <- W_0,m + update[m]`` for every module in ``base``. Restore ``W_0`` bit-exactly on exit.

    ``update=None`` or an empty dict is the zero update: the model runs as the original target. Modules missing from
    ``update`` get the zero update. Before anything is written, these raise: an edit already active on ``model``
    (``RuntimeError``), an update key that is not in ``base`` (``KeyError``), a base or update tensor whose shape
    differs from the live weight's, or an update with a non-finite entry after the cast to the weight's dtype
    (``ValueError``).
    """
    if getattr(model, "_active_edit", None) is not None:
        raise RuntimeError("an edit is already active on this model. Use its reference() for W_0 inside it")
    update = update or {}
    unknown = sorted(m for m in update if m not in base)
    if unknown:
        raise KeyError(f"update names modules that are not in base: {unknown}")
    W = {m: _weight(model, m) for m in base}
    for m, W0 in base.items():
        if W0.shape != W[m].shape:
            raise ValueError(f"{m}: base shape {tuple(W0.shape)}, weight shape {tuple(W[m].shape)}")
    Q = {}
    for m, u in update.items():
        Q[m] = torch.as_tensor(u).to(device=W[m].device, dtype=W[m].dtype)
        if Q[m].shape != W[m].shape:
            raise ValueError(f"{m}: update shape {tuple(Q[m].shape)}, weight shape {tuple(W[m].shape)}")
        if not torch.isfinite(Q[m]).all():
            raise ValueError(f"{m}: update has non-finite entries")
    edit = ActiveEdit(model, base, Q)
    model._active_edit = edit
    try:
        edit._write()
        yield edit
    finally:
        try:
            with torch.no_grad():
                for m, W0 in base.items():
                    _weight(model, m).copy_(W0)
            for m, W0 in base.items():
                if not torch.equal(_weight(model, m), W0):
                    raise RuntimeError(f"{m}: weight not restored exactly after an edit")
        finally:
            model._active_edit = None


def logits_of(out) -> Tensor:
    """Logits of a forward output: the first element of the model's ``(logits, loss)`` tuple. No dtype cast."""
    return out[0] if isinstance(out, (tuple, list)) else out


# --------------------------------------------------------------------------- frozen parameters
def snapshot_params(model: nn.Module, exclude: Iterable[str] = ()) -> dict[str, Tensor]:
    """Clones of every parameter except the ``.weight`` of the excluded modules (the trainable matrices)."""
    excluded = {f"{m}.weight" for m in exclude}
    return {n: p.detach().clone() for n, p in model.named_parameters() if n not in excluded}


def assert_frozen(model: nn.Module, snapshot: Mapping[str, Tensor]) -> None:
    params = dict(model.named_parameters())
    changed = [n for n, p0 in snapshot.items() if not torch.equal(params[n], p0)]
    if changed:
        raise AssertionError(f"frozen parameters changed: {changed}")
