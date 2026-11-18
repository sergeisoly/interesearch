# interesearch

How far does a weight edit spread through a language model? These two studies use Goodfire’s 67M-parameter model to measure changes in next-token prediction. One tests whether a deliberately damaging edit can spare a related language. The other asks how much of the damage an earlier repair can undo on a new fine-tune.

[Research overview](https://sergeisoly.github.io/interesearch/)

## Selective language editing

I calibrated edits to weaken German, Spanish or French by a specified amount on validation text, then measured the effect on Dutch, Portuguese or Italian. On 58 further European Parliament session dates, six edits to individual parameter-decomposition subcomponents caused less damage to the related language than rank-one LoRA, full-matrix fine-tuning or neuron edits. The two learned methods preserved English predictions better.

[Short summary and interactive figures](https://sergeisoly.github.io/interesearch/selective-editing/explainer.html) · [Paper](https://sergeisoly.github.io/interesearch/selective-editing/) · [Study guide](papers/guides/selective-editing.md)

## Repair transfer

Fine-tuning to weaken German also hurt Dutch. I took a repair learned for one fine-tune and tried it on two others trained from the same base with the same recipe. Fitting coefficients on the donor’s fixed directions recovered 86.9% and 78.4% of the Dutch damage in the expanded search. Training the directions from that same starting point recovered 98.2% and 98.8%. Freshly fitted repairs nearly caught up at the full budget.

[Short summary and interactive figures](https://sergeisoly.github.io/interesearch/repair-transfer/explainer.html) · [Paper](https://sergeisoly.github.io/interesearch/repair-transfer/) · [Study guide](papers/guides/repair-transfer.md)

## Reproduce a result

From the repository root, replay the repair comparison on a CPU using the checked-in measurements:

```bash
uv sync --locked
uv run python -m interesearch.repair_transfer.analysis.recovery data/repair_transfer \
    --out runs/replay/repair_transfer/analysis.json
uv run python -m interesearch.repair_transfer.figures runs/replay/repair_transfer/analysis.json \
    --out runs/replay/repair_transfer
```

Open `runs/replay/repair_transfer/tables/primary_table.md` for the recovery estimates and paired intervals. The [selective-editing replay](papers/guides/selective-editing.md#replay-the-papers-analysis) rebuilds the comparisons from per-date measurements. Its confirmatory token windows are unpublished.

## Try a weight edit

Apply the saved German edit on a CPU and compare German, Dutch and English prediction loss on four example windows per language. This demonstration downloads about 3.2 GB.

```bash
uv run python -m interesearch.assets fetch assets
uv run python -m interesearch.selective_editing.method.dictionary extract assets
uv run python -m interesearch.selective_editing.quickstart assets
```

The [worked example](papers/guides/selective-editing.md#try-one-edit-on-the-model) gives expected values and shows how to change the edit’s strength.

## Code and data

| To inspect | Start here |
|---|---|
| How the fixed subcomponent is chosen | [Selection rule](src/interesearch/selective_editing/method/selector.py) |
| How repairs are learned and chosen | [Fitting](src/interesearch/repair_transfer/method/fitting.py) · [Candidate selection](src/interesearch/repair_transfer/method/selection.py) |
| What goes into the figures | [Saved measurements](data/) · [Analysis outputs and figures](results/) |
| Environments, tests and paper builds | [Reproduction guide](papers/guides/reproduction.md) |

Both studies use Goodfire’s model. The selective study also uses its released parameter decomposition and follows [Lucius Bushnaq’s fine-tuning experiments](https://www.lesswrong.com/posts/ieoWstubDQWLrMnhH/exploration-fine-tuning-with-parameter-decomposition). Manuscript review used [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).

## Citation and licensing

Each paper links its published version in a BibTeX entry at the end of the HTML article. For software citations, see [CITATION.cff](CITATION.cff). Code and the released donor repair use the MIT license. The papers and original research outputs use CC BY 4.0. [Licensing](LICENSING.md) and [NOTICE](NOTICE) give the terms for third-party materials.
