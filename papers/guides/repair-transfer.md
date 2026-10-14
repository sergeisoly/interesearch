# Repair transfer

[Summary](https://sergeisoly.github.io/interesearch/repair-transfer/explainer.html) · [Paper](https://sergeisoly.github.io/interesearch/repair-transfer/index.html) · [PDF](https://sergeisoly.github.io/interesearch/repair-transfer/index.pdf)

Complete the [shared setup](reproduction.md#setup), then run these commands from the repository root.

- [Replay the analysis](#replay-the-papers-analysis): CPU, saved measurements, no model download.
- [Run targets and repairs](#run-a-new-target-and-its-repairs): model assets, published token windows and donor repair; CUDA by default.
- [Rebuild the donor](#rebuild-the-donor-repair): optional training and repair fitting.

## Replay the paper's analysis

```bash
uv run python -m interesearch.repair_transfer.analysis.recovery data/repair_transfer \
    --out runs/replay/repair_transfer/analysis.json
uv run python -m interesearch.repair_transfer.figures runs/replay/repair_transfer/analysis.json \
    --out runs/replay/repair_transfer
```

Compare the outputs with [`results/repair_transfer`](../../results/repair_transfer):

| Output under `runs/replay/repair_transfer/` | Contents |
|---|---|
| `analysis.json` | Selections, intervals, exclusions and costs |
| `tables/` | Recovery, readout constraints and break-even comparisons |
| `figures/` | Recovery, warm starts and quality–cost curves |

## Run a new target and its repairs

Use empty output directories. The model stages default to CUDA; see [validated environments](reproduction.md#validated-environments).
Fetch assets once:

```bash
uv run python -m interesearch.assets fetch assets
```

The loop trains both targets, then fits their repairs using the published donor. `main` uses 2,048 fit prediction positions;
`nested` uses 256 and 512. Each block selects checkpoints before evaluating separate readout windows.

```bash
for target in primary_target replication_target; do
  case "$target" in
    primary_target) seed=14 ;;
    replication_target) seed=15 ;;
  esac
  uv run python -m interesearch.repair_transfer.method.finetune train \
      --lr 3e-6 --seed "$seed" --assets assets \
      --train "data/repair_transfer/windows/finetune_train/$target" \
      --val data/repair_transfer/windows/finetune_val --out "runs/finetune/$target"

  for block in main nested; do
    uv run python -m interesearch.repair_transfer.stages.run block \
        --target "$target" --block "$block" --assets assets \
        --finetune "runs/finetune/$target" \
        --finetune-train "data/repair_transfer/windows/finetune_train/$target" \
        --finetune-val data/repair_transfer/windows/finetune_val \
        --donor-repair data/repair_transfer/donor/donor_repair.pt \
        --fit data/repair_transfer/windows/fit \
        --selection data/repair_transfer/windows/selection \
        --readout data/repair_transfer/windows/readout \
        --out "runs/blocks/$target/$block"
  done
done
```

Repair fitting requires the fine-tune to pass validation. Analysis requires both targets, both blocks and completed
readouts. Add the donor's cost record and checksums, then analyze the new measurements:

```bash
cp -R data/repair_transfer/donor runs/blocks/donor
uv run python -m interesearch.repair_transfer.analysis.recovery runs/blocks \
    --out runs/repair_transfer/analysis.json
uv run python -m interesearch.repair_transfer.figures runs/repair_transfer/analysis.json \
    --out runs/repair_transfer
```

## Change settings

See [recipe settings](reproduction.md#recipes) and the published [`recipe.yaml`](../../src/interesearch/repair_transfer/recipe.yaml).
Analysis uses saved settings; its `--recipe PATH` only checks that they match.

Changes to the scientific settings require `--target standin` for repair blocks and stand-in readout windows.
Reduced runs reject the study's readout windows. Donor runs with changed scientific settings are also marked reduced.
The paper-analysis command does not accept reduced runs.

## Rebuild the donor repair

<details>
<summary>Optional: train and validate a replacement donor</summary>

After fetching assets, train both donor learning rates and select a run:

```bash
for lr in 3e-6 1e-5; do
  uv run python -m interesearch.repair_transfer.method.finetune train --lr "$lr" --seed 11 --assets assets \
      --train data/repair_transfer/windows/finetune_train/donor \
      --val data/repair_transfer/windows/finetune_val --out "runs/finetune/donor_lr$lr"
done
uv run python -m interesearch.repair_transfer.method.finetune select \
    runs/finetune/donor_lr3e-6 runs/finetune/donor_lr1e-5
```

Replace `CHOSEN_DONOR_RUN` with the selected run's directory:

```bash
uv run python -m interesearch.repair_transfer.stages.run donor --assets assets \
    --finetune CHOSEN_DONOR_RUN \
    --fit data/repair_transfer/windows/donor_fit \
    --selection data/repair_transfer/windows/finetune_val \
    --readout data/repair_transfer/windows/donor_readout --out runs/donor
```

Require `gate.pass: true` in `runs/donor/run.json`; a zero exit code alone does not validate the donor.
Use `--donor-repair runs/donor/donor_repair.pt` in both targets' repair blocks and keep its adjacent `run.json`.
For analysis, copy `runs/donor` to `runs/blocks/donor` instead of the published donor; its `fit_record.json` and
`SHA256SUMS` supply the matching cost record.

</details>
