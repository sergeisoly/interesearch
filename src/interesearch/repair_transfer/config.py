"""Domain checks for the concrete Repair Transfer experiment recipe."""


def validate_recipe(recipe: dict) -> None:
    """Reject values that cannot define the study's fitting and readout procedures."""
    def positive(value, name):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    for section, keys in {
        "block": ("steps", "every", "readout_min"),
        "donor": ("steps", "every", "readout_dates"),
        "evaluation": ("batch_size", "threads", "parity_tol"),
        "bootstrap": ("n_boot",),
        "selection": ("top_retain",),
        "costs": ("selection_tokens",),
        "polish": ("max_iter", "patience", "alpha_min", "lam_range", "lam0_frac", "norm_floor"),
        "fitting": ("eps", "grad_clip", "fit_path_tol", "cap_tol", "init_tol"),
        "finetune": ("val_batch_size", "baseline_batch_size", "threads"),
    }.items():
        for key in keys:
            positive(recipe[section][key], f"{section}.{key}")
    cfg = recipe["finetune"]["config"]
    for key in ("lr", "steps", "n_target", "n_retain_per_lang", "micro_batch", "grad_clip", "eps"):
        positive(cfg[key], f"finetune.config.{key}")
    for name, values in (("fitting", recipe["fitting"]), ("finetune.config", cfg)):
        if values["warmup"] < 0:
            raise ValueError(f"{name}.warmup must be nonnegative")
        if len(values["betas"]) != 2 or not all(0 <= x < 1 for x in values["betas"]):
            raise ValueError(f"{name}.betas must contain two values in [0, 1)")
    for name in ("fitting", "finetune"):
        if recipe[name]["weight_decay"] < 0:
            raise ValueError(f"{name}.weight_decay must be nonnegative")
    if not cfg["checkpoints"] or any(x <= 0 for x in cfg["checkpoints"]):
        raise ValueError("finetune.config.checkpoints must contain positive steps")
    if cfg["condition"] != "matrix":
        raise ValueError("Repair Transfer uses the matrix fine-tune condition")
    if recipe["selection"]["retain_languages"] != ["en", "fr", "es", "it"]:
        raise ValueError("the saved Repair Transfer measurement contract requires en/fr/es/it retention")
    if recipe["selection"]["rho_lo"] > recipe["selection"]["rho_hi"]:
        raise ValueError("selection.rho_lo must not exceed rho_hi")
    quantiles = recipe["bootstrap"]["quantiles"]
    if len(quantiles) != 2 or not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError("bootstrap.quantiles must be two ordered probabilities in (0, 1)")
    if not 0 < recipe["polish"]["backtrack"] < 1 or not 0 < recipe["polish"]["armijo"] < 1:
        raise ValueError("polish.backtrack and armijo must be in (0, 1)")
    for family, rates in recipe["fitting"]["lrs"].items():
        if not rates or any(lr <= 0 for lr in rates):
            raise ValueError(f"fitting.lrs.{family} must contain positive learning rates")
    if not recipe["donor"]["lrs"] or any(lr <= 0 for lr in recipe["donor"]["lrs"]):
        raise ValueError("donor.lrs must contain positive learning rates")
    for value in recipe["fitting"]["batch"].values():
        positive(value, "fitting.batch")
    for budget, groups in recipe["fitting"]["budgets"].items():
        for group, pair in groups.items():
            if len(pair) != 2 or any(v <= 0 for v in pair):
                raise ValueError(f"fitting.budgets.{budget}.{group} requires positive window/token counts")
    for profile in recipe["block"]["profiles"].values():
        if not profile["budgets"] or not profile["families"]:
            raise ValueError("block profiles need at least one budget and family")
        if set(profile["budgets"]) - recipe["fitting"]["budgets"].keys():
            raise ValueError("block profile names an unknown fitting budget")
        if set(profile["families"]) - recipe["fitting"]["lrs"].keys():
            raise ValueError("block profile names an unknown family")
    for name in ("block", "curve"):
        if any(x <= 0 for x in recipe[name]["step_caps"]):
            raise ValueError(f"{name}.step_caps must be positive")
