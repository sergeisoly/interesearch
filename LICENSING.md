# Licensing

This repository combines the author's own work with third-party material. Each part keeps the terms below. The
licences of the author's work grant only the rights the author holds.

| Part | Licence |
|---|---|
| Code: `src/`, `tests/`, `papers/**/*.js`, build and package configuration (`pyproject.toml`, `uv.lock`, `.python-version`, `papers/_quarto.yml`, `papers/styles.css`), and the commands and code blocks in the README, guides and papers | MIT, [`LICENSE`](LICENSE) |
| `data/repair_transfer/donor/donor_repair.pt`: rank-one repair factors trained by the author (not Goodfire's model weights) | MIT, [`LICENSE`](LICENSE) |
| The papers (`papers/` sources other than the code and configuration above, their PDF and HTML builds), `README.md`, this file, the author's text in [`NOTICE`](NOTICE), and the author's figures and tables | CC BY 4.0, [`LICENSE-CONTENT`](LICENSE-CONTENT) |
| The author's exported measurements, analysis results and run records: `data/` except `donor_repair.pt` and the token arrays, including every `windows.json` and `SHA256SUMS`, plus `results/` | CC BY 4.0, [`LICENSE-CONTENT`](LICENSE-CONTENT) |
| Token arrays (`.npy` files under `data/*/windows/` and `data/quickstart/`) | Terms of their text sources, per file and per row, as listed in [`NOTICE`](NOTICE) |
| `src/interesearch/model.py`, adapted from Goodfire's param-decomp | MIT, with Goodfire's copyright notice reproduced in [`NOTICE`](NOTICE) |

The licence texts and third-party notices reproduced in `LICENSE`, `LICENSE-CONTENT` and `NOTICE` keep their own terms.

Not included: the target model, the VPD decomposition and the subcomponent labels are Goodfire's and are downloaded
from their source, not redistributed. The code's dependencies keep their own licences. Using the code to build an
edit from the downloaded decomposition does not relicense Goodfire's material.
