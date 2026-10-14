"""Command-line entry point for donor and target repair stages."""

from __future__ import annotations
import argparse
from copy import deepcopy
from ...recipes import load_recipe
from ..config import validate_recipe
from .support import TARGETS

from .block import cmd_block
from .donor import cmd_donor

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fit and compare ways to reuse an earlier repair on a newly fine-tuned model.",
        epilog="Usage and outputs: papers/guides/repair-transfer.md")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("block", "donor"):
        q = sub.add_parser(name, help={
            "block": "compare repair strategies on a new fine-tuned model",
            "donor": "learn the earlier repair that later targets can reuse",
        }[name])
        q.add_argument("--recipe", type=str, help="complete replacement for the published recipe")
        q.add_argument("--assets", required=True, help="assets directory with target/")
        q.add_argument("--finetune", required=True, help="fine-tune directory (finetune.py train)")
        q.add_argument("--fit", required=True, help="window directory of the fit windows")
        q.add_argument("--selection", required=True, help="window directory of the selection windows")
        q.add_argument("--readout", required=True, help="window directory of the readout windows")
        q.add_argument("--out", required=True, help="new directory for measurements and the run record")
        q.add_argument("--device", default="cuda")
        q.add_argument("--threads", type=int, default=None)
        q.add_argument("--eval-batch-size", type=int, default=None)
        q.add_argument("--steps", type=int, default=None)
        q.add_argument("--every", type=int, default=None)
        q.add_argument("--rows", type=int, default=None, help="reduced runs: first rows of the evaluation windows")
        if name == "block":
            q.add_argument("--target", required=True, choices=TARGETS)
            q.add_argument("--block", required=True, choices=("main", "nested"))
            q.add_argument("--donor-repair", required=True, help="donor_repair.pt")
            q.add_argument("--finetune-train", default=None, help="the fine-tune's training window directory "
                           "(primary_target, replication_target)")
            q.add_argument("--finetune-val", default=None, help="the fine-tune's validation window directory "
                           "(primary_target, replication_target)")
            q.add_argument("--step-caps", default=None)
            q.add_argument("--polish-iters", type=int, default=None)
    a = ap.parse_args(argv)
    published = load_recipe("interesearch.repair_transfer")
    a.resolved_recipe = load_recipe("interesearch.repair_transfer", a.recipe) if a.recipe else deepcopy(published)
    profile = a.resolved_recipe[a.cmd]
    for key in ("steps", "every"):
        value = getattr(a, key)
        if value is not None:
            profile[key] = value
        setattr(a, key, profile[key])
    if a.cmd == "block":
        try:
            a.step_caps = sorted({int(x) for x in a.step_caps.split(",") if x}) if a.step_caps is not None else profile["step_caps"]
        except ValueError:
            ap.error("--step-caps must be comma-separated integers")
        profile["step_caps"] = a.step_caps
        a.polish_iters = a.resolved_recipe["polish"]["max_iter"] if a.polish_iters is None else a.polish_iters
        a.resolved_recipe["polish"]["max_iter"] = a.polish_iters
    evaluation = a.resolved_recipe["evaluation"]
    for argument, key in (("eval_batch_size", "batch_size"), ("threads", "threads")):
        value = getattr(a, argument)
        if value is not None:
            evaluation[key] = value
        setattr(a, argument, evaluation[key])
    try:
        validate_recipe(a.resolved_recipe)
        if a.rows is not None and a.rows <= 0:
            raise ValueError("--rows must be positive")
    except ValueError as error:
        ap.error(str(error))
    # Device, threads, evaluation batching, and profile labels do not change the scientific protocol.
    def scientific(recipe):
        values = deepcopy(recipe)
        values.pop("profile")
        for key in ("batch_size", "threads"):
            values["evaluation"].pop(key)
        values["finetune"].pop("threads")
        return values

    frozen, resolved = scientific(published), scientific(a.resolved_recipe)
    a.deviations = {key: value for key, value in resolved.items() if value != frozen[key]}
    if a.rows is not None:
        a.deviations["rows"] = a.rows
    if a.cmd == "donor":
        a.reduced = bool(a.deviations)
        return cmd_donor(a)
    if a.deviations and a.target != "standin":
        ap.error(f"{', '.join(a.deviations)} away from the frozen values need --target standin")
    a.reduced = a.target == "standin"
    return cmd_block(a)


if __name__ == "__main__":
    raise SystemExit(main())
