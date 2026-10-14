---
image: ../social-card.png
description: "Replay the repair-transfer analysis and fit repairs."
title: "Repair transfer: study guide"
format: html
---

The experiment deliberately weakens German prediction, then tries to repair the collateral damage to Dutch while
preserving the fine-tune's German effect and meeting constraints on other languages. Reusing a repair could reduce
the fitting needed for a new fine-tune. The earlier fine-tune is the *donor*, and its learned repair is the *donor repair*.
The new fine-tunes are the *targets*. Donor and targets are trained independently from the same base model with
the same recipe, using different seeds and Europarl samples from the same data distribution.

::: {.study-resources}
[Visual guide](../repair-transfer/explainer.qmd) · [Paper](../repair-transfer/index.qmd) · [PDF](../repair-transfer/index.pdf)
:::

[Setup, data and validation](reproduction.md)

::: {.guide-task-options}
| Task | Inputs and requirements | Output |
|---|---|---|
| [Replay the analysis](#replay-the-papers-analysis) | Saved measurements; CPU, no model download | Recovery comparisons, figures and cost accounting |
| [Run targets and repairs](#run-a-new-target-and-its-repairs) | Model assets, published donor repair and token windows; CUDA by default | New fine-tunes, fitted repairs and readout measurements |
| [Rebuild the donor](#rebuild-the-donor-repair) | Model assets and published donor windows; CUDA by default | A donor repair, validation record and fitting cost record |
:::

## Replay the paper's analysis

These commands turn saved measurements into repair comparisons and figures. They run on the CPU without a model download.

```bash
uv run python -m interesearch.repair_transfer.analysis.recovery data/repair_transfer \
    --out runs/replay/repair_transfer/analysis.json
uv run python -m interesearch.repair_transfer.figures runs/replay/repair_transfer/analysis.json \
    --out runs/replay/repair_transfer
```

Outputs in `runs/replay/repair_transfer/`:

| Output | What to look for |
|---|---|
| `tables/primary_table.md` | Recovery from fitting scalar coefficients over fixed donor directions versus donor initialization with trainable directions. |
| `figures/recovery.svg` | How much Dutch damage each repair removes. |
| `figures/warm_start.svg` | Whether donor initialization helps when fitting data or training steps are limited. |
| `tables/readout_table.md` | Whether recovering Dutch preserves the German effect and the other language constraints. |
| `tables/break_even_table.md` | When reuse can repay the cost of learning the donor. |
| `analysis.json` | Selections, confidence intervals, exclusions and cost accounting behind the summaries. |

Compare with `results/repair_transfer`. The results describe transfer from this donor to the two tested targets
under the same model and fine-tuning recipe. Transfer across a population of fine-tunes remains untested.

## Recipes and saved runs

Each study ships one `recipe.yaml`; Repair Transfer stores its published profile in
`src/interesearch/repair_transfer/recipe.yaml`. Pass a complete replacement with `--recipe PATH` to fine-tuning,
donor fitting or block fitting. The existing concrete command-line options override the corresponding recipe values.
Every model stage writes the complete resolved recipe into `run.json` before loading the model, and successful
block and donor exports carry that profile with their measurement records.

A changed scientific profile makes a donor run reduced. For a block it requires `--target standin` and stand-in
readout windows; protected study readout roles remain unavailable to reduced runs. Changing only the recipe's
filename or comments does not change its scientific profile.

Analysis replays the profile saved with each export. The published legacy exports use the packaged published
profile. On the analysis command, `--recipe PATH` checks that the requested replay settings match the saved
settings; it does not override them. Figures and tables use the resulting analysis and its saved profile.

## Reading the saved identifiers

| Identifier | Meaning in the paper |
|---|---|
| `strengths` | Scalar coefficients fitted over the donor's frozen input and output factors. |
| `strengths_converged`, `strengths_best` | The additional candidates from full-batch optimization, and the selection over these together with the regular `strengths` candidates. |
| `donor_init`, `donor_init_output` | Donor initialization with both factors trained, or only the output factor trained. |
| `fresh`, `random_init` | Fresh initialization, or initialization from a norm-matched randomized donor. The `_seed2` suffix marks the second seed. |
| `donor_fixed`, `donor_raw` | The donor repair scaled down if needed to satisfy the target norm cap, or the unchanged donor repair without that cap. Neither is fitted on the target. `donor_raw` is a descriptive reference. |

In a key such as `donor_init@2K:le50`, `2K` means 2,048 fit prediction positions and `:le50` restricts checkpoint
selection to steps up to 50. The `256` and `512` budgets use prefixes of the same fitting windows.

## Run a new target and its repairs

The commands use the published donor repair and token windows. Each target is trained and validated before
repair fitting and readout. Use empty output directories; the runner refuses non-empty directories.

To analyze the new measurements, complete both targets and both repair blocks below. The analysis also needs
the donor's cost record and checksums; the final step assembles this directory layout.

Fetch the model assets once:

```bash
uv run python -m interesearch.assets fetch assets
```

Model commands default to CUDA and use the model assets directly, without dictionary extraction. See the
[validated environments](reproduction.md#validated-environments) for reproduction limits.

Train the primary target:

```bash
uv run python -m interesearch.repair_transfer.method.finetune train --lr 3e-6 --seed 14 --assets assets \
    --train data/repair_transfer/windows/finetune_train/primary_target \
    --val data/repair_transfer/windows/finetune_val --out runs/finetune/primary_target
```

The target must pass the fine-tune validation checks before repair fitting can proceed. Outputs include the selected `update.pt`,
`val.json` and the run record `run.json`.

`main` fits repairs with 2,048 tokens. `nested` uses 256 and 512 tokens from prefixes of the same fitting windows.
These budgets count prediction positions, excluding the initial context token of each window.
Each block chooses checkpoints on selection windows before evaluating them on separate readout windows.

```bash
for block in main nested; do
  uv run python -m interesearch.repair_transfer.stages.run block \
      --target primary_target --block "$block" --assets assets \
      --finetune runs/finetune/primary_target \
      --finetune-train data/repair_transfer/windows/finetune_train/primary_target \
      --finetune-val data/repair_transfer/windows/finetune_val \
      --donor-repair data/repair_transfer/donor/donor_repair.pt \
      --fit data/repair_transfer/windows/fit \
      --selection data/repair_transfer/windows/selection \
      --readout data/repair_transfer/windows/readout \
      --out "runs/blocks/primary_target/$block"
done
```

For the replication target, repeat the fine-tune with `--seed 15`, replacing `primary_target` with
`replication_target` in the training-window path and output directory. Repeat both blocks with
`--target replication_target` and the corresponding fine-tune, training-window and output paths. Keep the fit,
selection and readout directories unchanged.

The analysis command expects both targets and both blocks, plus a `donor/` directory containing its fit-cost
record and checksums, in the same layout as `data/repair_transfer/`. When using the committed donor above, copy
`data/repair_transfer/donor/` into `runs/blocks/donor/`, then replace the analysis input `data/repair_transfer`
with `runs/blocks`. If you rebuild the donor below, use that donor's exported records and repair consistently.

Analysis requires a completed readout. Reduced runs require `--target standin` and stand-in windows and reject
the study's readout windows.

On the NVIDIA A10 I tested, the primary fine-tune took 57 seconds and wrote a 113 MB update.

## Rebuild the donor repair

The target commands above use the published donor. This optional procedure repeats its training and repair fitting.

Train the donor at both learning rates, then select a checkpoint across both runs:

```bash
for lr in 3e-6 1e-5; do
  uv run python -m interesearch.repair_transfer.method.finetune train --lr "$lr" --seed 11 --assets assets \
      --train data/repair_transfer/windows/finetune_train/donor \
      --val data/repair_transfer/windows/finetune_val --out "runs/finetune/donor_lr$lr"
done
uv run python -m interesearch.repair_transfer.method.finetune select \
    runs/finetune/donor_lr3e-6 runs/finetune/donor_lr1e-5
```

Use the selected run's directory as `--finetune` below, replacing `CHOSEN_DONOR_RUN`:

```bash
uv run python -m interesearch.repair_transfer.stages.run donor --assets assets \
    --finetune CHOSEN_DONOR_RUN \
    --fit data/repair_transfer/windows/donor_fit \
    --selection data/repair_transfer/windows/finetune_val \
    --readout data/repair_transfer/windows/donor_readout --out runs/donor
```

Check that `gate.pass` is true in `run.json`; a zero process exit code alone does not validate the donor. A valid result can
replace the committed donor through `--donor-repair runs/donor/donor_repair.pt`. Keep the adjacent `run.json`,
which validates the rebuilt donor. Its exported `fit_record.json` supplies the donor cost for analysis.
