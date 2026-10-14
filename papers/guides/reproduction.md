# Code and data

The study guides cover analysis replay and new model runs:
[selective editing](selective-editing.md) · [repair transfer](repair-transfer.md).
This page covers their shared setup, checks and builds.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run all Python commands from the repository root:

```bash
uv sync --locked
```

Model downloads also need `curl`; the Goodfire param-decomp environment needs `git`.

## Recipes

For a new model run, copy the study's recipe, change the required settings and pass the complete file with
`--recipe PATH`. Explicit command-line options override corresponding recipe settings, and runs save the
resolved settings. Use a new output directory when changing settings. The defaults are in
[selective editing's recipe](../../src/interesearch/selective_editing/recipe.yaml) and
[repair transfer's recipe](../../src/interesearch/repair_transfer/recipe.yaml).
Analysis replay uses the settings associated with the saved measurements.

## Data and reproduction limits

[data/](../../data/) contains saved measurements, calibration and training records, published token windows
and the donor repair. Loaders check exported files against `SHA256SUMS`; each `windows.json` records the text
source, tokenizer and selection rule.

- **Analysis replay** runs on CPU without downloading the model and regenerates [results/](../../results/).
  Tables and figures read those outputs; numbers in paper prose are maintained separately.
- **New model runs** require the inputs described in each study guide. Selective editing's confirmatory texts
  and token windows are not published. Its exploratory learned updates were not retained and must be retrained.

Model and decomposition downloads are checked against pinned sizes and hashes. To recheck downloaded assets:

```bash
uv run python -m interesearch.assets verify assets
```

## Validated environments

- **Apple M4 Mac, locked environment:** analysis replay matched all 23 published result files byte for byte;
  the tests and HTML/PDF builds passed. Replay tests allow relative error of 1e-12 and absolute error of 1e-15.
- **AMD EPYC 9454:** the CPU quickstart ran with about 3.3 GB of memory.
- **GPU checks on v0.1.0, before module reorganization:** the selective-editing curvature gate passed on L40S
  and failed its float64 batch-invariance criterion on A10 and L4, preventing selection there.

Selective-editing selection requires a gate from the same GPU model, SM count and torch/CUDA version.
Changing hardware may change selections or prevent a run. These checks do not cover complete experiments on
every environment.

## Check the code

```bash
uv run --locked pytest -q
```

The suite needs no downloaded model assets. It checks analysis replay, weight edits and model paths on a tiny
model, input separation, and experiment-runner file bindings.

## Build the site and papers

Install [Quarto](https://quarto.org/) for HTML. PDFs also need LuaLaTeX, `rsvg-convert`, and the STIX Two Text and
STIX Two Math fonts. From the repository root:

```bash
cd papers
quarto render
```

This builds the site and papers in `papers/_site/` from committed results. For HTML only, use
`quarto render --to html`. The checked PDF toolchain was Quarto 1.10.19, LuaHBTeX 1.24.0 and `rsvg-convert` 2.63.2.

## Source layout and licences

The [selective-editing](../../src/interesearch/selective_editing/) and
[repair-transfer](../../src/interesearch/repair_transfer/) source directories each contain `method/` for
numerical procedures, `stages/` for model runs, `analysis/` for saved-measurement replay, and `figures.py` for
tables and plots.

The model, decomposition and labels come from Goodfire's
[Interpreting Language Model Parameters](https://www.goodfire.com/research/interpreting-lm-parameters);
the model implementation is adapted from [param-decomp](https://github.com/goodfire-ai/param-decomp).
Text windows come from Europarl v7, WikiText-2, NTREX-128 and Pile-uncopyrighted.
See [NOTICE](../../NOTICE) for sources and terms, and [LICENSING.md](../../LICENSING.md) for licences.
