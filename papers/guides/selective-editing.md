# Selective editing

[Summary](https://sergeisoly.github.io/interesearch/selective-editing/explainer.html) · [Paper](https://sergeisoly.github.io/interesearch/selective-editing/) · [PDF](https://sergeisoly.github.io/interesearch/selective-editing/index.pdf) · [Setup](reproduction.md#setup)

Run commands from the repository root after `uv sync --locked`.

| Task | Requirements |
|---|---|
| [Recreate tables and figures](#replay-the-papers-analysis) | CPU; included measurements |
| [Try one weight edit](#try-one-edit-on-the-model) | CPU; about 3.2 GB download |
| [Run selection and calibration](#run-selection-calibration-and-development-evaluation) | Model assets; GPU with a passing curvature check |

Saved confirmatory measurements are included. Their input windows are unpublished, so the confirmatory model evaluation cannot be rerun from this checkout.

## Replay the paper's analysis

No model download is needed.

```bash
uv run python -m interesearch.selective_editing.analysis.confirm data/selective_editing \
    --out runs/replay/selective_editing
uv run python -m interesearch.selective_editing.analysis.posthoc data/selective_editing \
    --out runs/replay/selective_editing
uv run python -m interesearch.selective_editing.analysis.extended data/selective_editing \
    --out runs/replay/selective_editing
uv run python -m interesearch.selective_editing.figures runs/replay/selective_editing \
    --out runs/replay/selective_editing
```

In `runs/replay/selective_editing/`, start with:

| File | Contents |
|---|---|
| `tables/claim_table.md` | Main comparisons |
| `figures/effects.svg` | Target and related-language effects |
| `confirm_result.json` | Verdicts and exclusions |
| `tables/extended_comparisons_table.md` | Exploratory comparisons |

Compare with the committed [results](../../results/selective_editing/). Interpretation and study limitations are in the [paper](https://sergeisoly.github.io/interesearch/selective-editing/).

## Try one edit on the model

Apply a saved German-targeted edit, then measure German, Dutch and English on four windows per language.

```bash
uv run python -m interesearch.assets fetch assets
uv run python -m interesearch.selective_editing.method.dictionary extract assets
uv run python -m interesearch.selective_editing.quickstart assets
```

Example CPU results with the default `--tau 1.0`, in nats/token:

| Language | CE increase | KL from original predictions |
|---|---:|---:|
| German | 1.014453 | 0.905535 |
| Dutch | 0.003635 | 0.003228 |
| English | 0.002091 | 0.000513 |

Higher cross-entropy (CE) means worse next-token prediction; KL measures how much predictions change. Final digits may vary across runtimes. This small demonstration uses separate windows from the paper's evaluation.

Add `--tau 0.5` for a smaller edit or `--out runs/quickstart.json` to save the result.

## Run selection, calibration and development evaluation

First fetch the assets and extract the dictionary with the commands above. Use a GPU that passes the curvature check: it passed on the tested L40S and failed on A10 and L4. The check and selection must use the same GPU model, SM count and torch/CUDA runtime. See [validated environments](reproduction.md#validated-environments).

Prepare the check and the pinned Goodfire environment used by the `ci_difference` comparator:

```bash
uv run python -m interesearch.selective_editing.method.fisher assets \
    --retain data/selective_editing/windows/retain_en/en.npy \
    --out runs/selective_editing/fisher_check.json
git clone https://github.com/goodfire-ai/param-decomp vendor/param-decomp
git -C vendor/param-decomp checkout 74146b555f3eb3cd2bc639c3789fd3b7a099094a
(cd vendor/param-decomp && uv sync --frozen --no-install-project --no-dev)
```

Continue only if the check prints `passed`. The stages select and calibrate edits, freeze the choices, then evaluate them on the published development windows:

```bash
uv run python -m interesearch.selective_editing.stages.main prefreeze assets \
    --out runs/selective_editing --gate runs/selective_editing/fisher_check.json \
    --ci-python vendor/param-decomp/.venv/bin/python
uv run python -m interesearch.selective_editing.stages.main freeze --out runs/selective_editing \
    --gate runs/selective_editing/fisher_check.json --note "Development run with published windows"
uv run python -m interesearch.selective_editing.stages.main readout assets --out runs/selective_editing
```

Outputs go to `runs/selective_editing/records/` and `analysis/`. Repeating a command reuses valid outputs when its settings and inputs match. For custom settings, see [recipes](reproduction.md#recipes).

### Exploratory comparisons

<details>
<summary>Run longer training and alternative rank-one directions</summary>

Requires the completed development run above. These commands retrain updates that were not retained and write new measurements to a separate directory.

```bash
uv run python -m interesearch.selective_editing.stages.extended prefreeze assets \
    --main runs/selective_editing --out runs/selective_editing_extended_comparisons \
    --gate runs/selective_editing/fisher_check.json
uv run python -m interesearch.selective_editing.stages.extended freeze \
    --main runs/selective_editing --out runs/selective_editing_extended_comparisons \
    --gate runs/selective_editing/fisher_check.json --note "Exploratory development comparison"
uv run python -m interesearch.selective_editing.stages.extended readout assets \
    --main runs/selective_editing --out runs/selective_editing_extended_comparisons
```

</details>
