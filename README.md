# interesearch

Studies of language-model weight edits: selectively weakening one language while sparing a close relative,
and reusing repairs for collateral damage across fine-tunes trained on data from the same distribution.

## Studies

**Selective language editing.** I compare weight edits by their effects on a target language and its close relative.
Following Lucius Bushnaq's [Exploration: Fine-Tuning with Parameter Decomposition](https://www.lesswrong.com/posts/ieoWstubDQWLrMnhH/exploration-fine-tuning-with-parameter-decomposition),
I test German/Dutch, Spanish/Portuguese and French/Italian. Individual parameter-decomposition subcomponents,
learned weight updates (including full-matrix fine-tuning), and neuron edits are calibrated to the same target
effect on validation text. On 58 new session dates in the same model, the six subcomponent edits caused less
related-language damage than the comparators. The learned updates preserved English predictions better.

[Short summary](https://sergeisoly.github.io/interesearch/selective-editing/explainer.html) · [Selective Language Damage with Parameter-Decomposition Subcomponents](https://sergeisoly.github.io/interesearch/selective-editing/) · [Study guide](papers/guides/selective-editing.md)

**Repair transfer.** I reuse a learned repair for new fine-tunes trained independently from the same base model
on different samples from the same data distribution.
I compare rescaling its fixed directions, training from it as an initialization, and fitting a fresh repair,
varying the available data and training budget. On two new fine-tunes, fixed directions left more Dutch damage;
donor initialization helped most when fitting data and steps were limited, while fresh fitting nearly caught up
at the full budget.

[Short summary](https://sergeisoly.github.io/interesearch/repair-transfer/explainer.html) · [Transferring Learned Repairs Between Fine-Tunes](https://sergeisoly.github.io/interesearch/repair-transfer/) · [Study guide](papers/guides/repair-transfer.md)

Both studies use Goodfire's small language model. Selective editing also uses its released parameter decomposition.

## Install

```bash
uv sync --locked
```

Replay the saved analysis for [selective editing](papers/guides/selective-editing.md#replay-the-papers-analysis)
or [repair transfer](papers/guides/repair-transfer.md#replay-the-papers-analysis) on the CPU without downloading the model.

## Try a weight edit

Weaken the model's German predictions and see what happens to Dutch and English. This example applies one
previously selected edit to four fixed text windows per language. It downloads about 3.2 GB and runs on the CPU.

```bash
uv run python -m interesearch.assets fetch assets
uv run python -m interesearch.selective_editing.method.dictionary extract assets
uv run python -m interesearch.selective_editing.quickstart assets
```

The output compares prediction error (cross-entropy, or CE) before and after the edit.

The [selective-editing guide](papers/guides/selective-editing.md#try-one-edit-on-the-model) explains the output and how to
change the edit's strength.

## Work with the research

Each study guide covers its question, analysis outputs and model runs. The
[code and data guide](papers/guides/reproduction.md) covers shared data, validated environments, tests and HTML/PDF builds.

Code is MIT licensed. Papers and original research outputs are CC BY 4.0. See
[licensing and attribution](LICENSING.md) and [NOTICE](NOTICE).

## Cite

Cite the relevant paper for its findings. For use of the code, see [CITATION.cff](CITATION.cff).

```bibtex
@misc{solovev2026repair,
  author = {Solovev, Sergei},
  title  = {Transferring Learned Repairs Between Fine-Tunes},
  year   = {2026},
  month  = oct,
  note   = {Version 0.1.0},
  url    = {https://sergeisoly.github.io/interesearch/repair-transfer/}
}

@misc{solovev2026selective,
  author = {Solovev, Sergei},
  title  = {Selective Language Damage with Parameter-Decomposition Subcomponents},
  year   = {2026},
  month  = oct,
  note   = {Version 0.1.0},
  url    = {https://sergeisoly.github.io/interesearch/selective-editing/}
}
```

## Editorial tools

Manuscript review used [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
