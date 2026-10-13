---
image: ../social-card.png
description: "Replay the selective-editing analysis and run the available model procedures."
title: "Selective editing: study guide"
format: html
---

The study compares edits to VPD rank-one subcomponents with learned weight updates and neuron edits. Each edit
is calibrated on the target language, then evaluated for damage to a related language.

::: {.study-resources}
[Visual guide](../selective-editing/explainer.qmd) · [Paper](../selective-editing/index.qmd) · [PDF](../selective-editing/index.pdf)
:::

[Setup, data and validation](reproduction.md)

::: {.guide-task-options}
| Task | Inputs and requirements | Output |
|---|---|---|
| [Replay the analysis](#replay-the-papers-analysis) | Saved measurements; CPU, no model download | The paper's tables, figures and calibration checks |
| [Try one edit](#try-one-edit-on-the-model) | Model and decomposition assets; CPU | Prediction changes on four fixed windows per language |
| [Run selection and calibration](#run-selection-calibration-and-development-evaluation) | Assets, published development windows and a passing GPU curvature check | New selections, calibrated edits and development measurements |
:::

The confirmatory model evaluation requires unpublished token windows and texts, so it cannot be rerun from
this checkout. Its saved per-date measurements are included in the CPU replay.

## Replay the paper's analysis

These commands turn saved measurements into comparison tables and figures. They run on the CPU without a model download.

```bash
uv run python -m interesearch.selective_editing.confirm data/selective_editing \
    --out runs/replay/selective_editing
uv run python -m interesearch.selective_editing.posthoc data/selective_editing \
    --out runs/replay/selective_editing
uv run python -m interesearch.selective_editing.extended_comparisons data/selective_editing \
    --out runs/replay/selective_editing
uv run python -m interesearch.selective_editing.figures runs/replay/selective_editing \
    --out runs/replay/selective_editing
```

Outputs in `runs/replay/selective_editing/`:

| Output | What to look for |
|---|---|
| `claim_table.md` | Differences in damage to the related language, alongside the achieved target-effect differences. Negative related-language differences favor the subcomponent edit. |
| `figures/effects.svg` | The target and related-language effects side by side. |
| `confirm_result.json` | Verdicts, exclusions and the conditions attached to the headline comparison. |
| `calibration_trace.json` | Whether the recorded measurements reproduce each calibration decision. The committed data yield 128 agreements out of 128 decisions. |
| `extended_comparisons_table.md` | Exploratory comparisons with longer-trained updates and alternative rank-one directions. |

Compare with `results/selective_editing`. Replay tolerances are in the [shared guide](reproduction.md).

The overall `conditional` status means that the six planned randomized-dictionary contrasts have no estimate.
For an individual comparison, `conditional` means that matching and usability rules exclude some training draws.
The original learned baselines had a fixed training budget without established convergence, and earlier project
work was not blind to the related languages. The additional comparisons are exploratory and use development data already examined.

## Try one edit on the model

Weaken German prediction and measure Dutch and English on four fixed text windows per language. Higher
cross-entropy (CE) means worse next-token prediction. KL measures how far predictions move from the original model.

The download is about 3.18 GB. The example then runs on the CPU.

```bash
uv run python -m interesearch.assets fetch assets
uv run python -m interesearch.selective_editing.dictionary extract assets
uv run python -m interesearch.selective_editing.quickstart assets
```

Example output from my CPU test run with the default `--tau 1.0`. Final digits may vary across runtimes.
CE and KL are in nat/token.

```text
  de (target): CE 3.133188 -> 4.147641 (rise +1.014453, KL 0.905535)
  nl (protected): CE 3.051603 -> 3.055238 (rise +0.003635, KL 0.003228)
  en (English): CE 3.542583 -> 3.544675 (rise +0.002091, KL 0.000513)
Edit: subcomponent 0.attn.o:268 (h.0.attn.o_proj component 268), dose a = 0.945083, coefficient change -0.945083, resulting coefficient 0.054917
```

The edit raises German CE by about 1.014 on these four windows, while Dutch and English change by less than
0.004. The dose `a` sets the edit's size. Its signed coefficient change is `-a`, so the subcomponent's coefficient
falls from 1 to about 0.055. The saved JSON retains the keys `atom`, `a`, `delta_scale` and `resulting_mask` for
these quantities.

The four windows illustrate the intervention. The paper evaluates it on a separate, larger set of text windows.

Use `--tau 0.5` for the smaller calibrated effect or `--out runs/quickstart.json` to save the result. On the cloud
CPU I tested, evaluation took about 7 seconds with 3.3 GB peak resident memory after the download.

## Run selection, calibration and development evaluation

This procedure selects and calibrates new edits before evaluating them on the published development windows.
It requires the model assets and extracted dictionary used by the quickstart above.

Selection requires a passing curvature-check record from the same GPU model, SM count and torch/CUDA runtime.
It passed on the tested L40S and failed on A10 and L4. Sampling depends on the GPU model. See
[validated environments](reproduction.md#validated-environments).

After fetching the assets and extracting the dictionary as above, prepare the curvature record and Goodfire
reference environment. `ci_difference`, one of the comparison methods, uses Goodfire's original code at the
commit associated with the published decomposition.

```bash
uv run python -m interesearch.selective_editing.fisher assets \
    --retain data/selective_editing/windows/retain_en/en.npy \
    --out runs/selective_editing/fisher_check.json
git clone https://github.com/goodfire-ai/param-decomp vendor/param-decomp
git -C vendor/param-decomp checkout 74146b555f3eb3cd2bc639c3789fd3b7a099094a
(cd vendor/param-decomp && uv sync --frozen --no-install-project --no-dev)
```

On the Mac I tested, the Goodfire reference environment took about 2.5 minutes to install, and `ci_difference`
needed about 6 GB of RAM.

`prefreeze` selects and calibrates edits and trains comparators → `freeze` fixes the choices → `readout` evaluates
them on the published development windows.

```bash
uv run python -m interesearch.selective_editing.run prefreeze assets \
    --out runs/selective_editing --gate runs/selective_editing/fisher_check.json \
    --ci-python vendor/param-decomp/.venv/bin/python
uv run python -m interesearch.selective_editing.run freeze --out runs/selective_editing \
    --gate runs/selective_editing/fisher_check.json --note "Development run with published windows"
uv run python -m interesearch.selective_editing.run readout assets --out runs/selective_editing
```

The default input is `data/selective_editing/windows/`. The `readout` stage writes development measurements
to `runs/selective_editing/records/` and computes their analysis. Existing outputs are reused only when code,
settings and inputs match `RUN.json`. Progress is recorded in `status.jsonl`.

For a narrower run, `--cells de:0` selects the German target with its first training-data draw. `--reduced` starts
a new run at toy sizes for checking the code path. Its readout needs `--standin`, which uses selection data in
place of evaluation data. Use these runs to check the code path; reproducing the paper's results requires the full procedure.

The confirmatory model stage needs a separate `--confirm-windows` input. Those token windows and texts are not
published, including the reserve dates, so this checkout cannot rerun that stage. Its saved per-date readouts
are public and are included in the CPU analysis above. They evaluate six fixed edits on 58 session dates,
without validating the selection procedure on unseen related languages. The original freeze documents and frozen code list are
also not published. The paper marks which steps rest only on the author's records, which are not independently timestamped.

## Run the exploratory comparisons {#run-extended-comparisons}

These exploratory comparisons test longer training and alternative rank-one directions. They require the main
run's frozen choices and completed development readout, with a separate output directory.

```bash
uv run python -m interesearch.selective_editing.extended_comparisons_run prefreeze assets \
    --main runs/selective_editing --out runs/selective_editing_extended_comparisons \
    --gate runs/selective_editing/fisher_check.json
uv run python -m interesearch.selective_editing.extended_comparisons_run freeze \
    --main runs/selective_editing --out runs/selective_editing_extended_comparisons \
    --gate runs/selective_editing/fisher_check.json --note "Exploratory development comparison"
uv run python -m interesearch.selective_editing.extended_comparisons_run readout assets \
    --main runs/selective_editing --out runs/selective_editing_extended_comparisons
```

The saved records retain the historical study tag `selective_editing_addendum` for these comparisons
to preserve their recorded checksums.

This path retrains the exploratory updates, which were not retained, and produces new measurements.
The CPU replay uses saved records to reproduce the published analysis.

## Saved method identifiers

The exported method identifiers are `vpd_atom` (VPD subcomponent), `lora_rank1` (rank-one LoRA), `finetune_24`
(full-matrix fine-tuning), `neurons` (neuron edits) and `null_atom` (randomized dictionary). The exploratory
comparisons use `svd_atom` (SVD components), `whitened_rank1` (the whitened rank-one selector),
`lora_rank1_long` and `finetune_24_long` (the two learned comparators trained for 2,000 steps).
