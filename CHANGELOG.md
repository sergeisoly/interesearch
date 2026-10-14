# Changelog

## 0.2.0 — unreleased

Code is grouped by numerical method, analysis and model stage within each study. Experimental recipes live
in each study's `recipe.yaml`; entry points load them and save the resolved settings with a run. Markdown
tables moved to `results/<study>/tables/`. The papers and their version remain 0.1.0.

Selective Editing records the source version for each stage invocation, including resumed work. A source-file
change alone no longer blocks a stage. Checks of frozen artifacts, data roles, configuration, model assets and
curvature/cache validity remain. The all-matrix LoRA condition is named `lora_24`.

The command paths below replace those in 0.1.0. There are no compatibility wrappers. The assets, quickstart and
figures commands keep their names. Guides keep their URLs and document the current commands.

| Previous module after `python -m interesearch.` | New module |
|---|---|
| `selective_editing.confirm` | `selective_editing.analysis.confirm` |
| `selective_editing.posthoc` | `selective_editing.analysis.posthoc` |
| `selective_editing.extended_comparisons` | `selective_editing.analysis.extended` |
| `selective_editing.dictionary` | `selective_editing.method.dictionary` |
| `selective_editing.fisher` | `selective_editing.method.fisher` |
| `selective_editing.references` | `selective_editing.method.references` |
| `selective_editing.run` | `selective_editing.stages.main` |
| `selective_editing.extended_comparisons_run` | `selective_editing.stages.extended` |
| `repair_transfer.analysis` | `repair_transfer.analysis.recovery` |
| `repair_transfer.finetune` | `repair_transfer.method.finetune` |
| `repair_transfer.run` | `repair_transfer.stages.run` |

Repair Transfer's analysis command no longer accepts `--table`; the figures command writes all four tables.
The source data and the numerical results of the papers are unchanged. Model parity measurements documented
for 0.1.0 continue to refer to that release.
