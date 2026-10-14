| Method | τ | Target rise | Protected rise | English rise | English KL, C (10⁻³) | English KL, D (10⁻³) | Edit norm |
|:--|:--|:--|:--|:--|:--|:--|:--|
| VPD subcomponent | 0.5 | 0.48–0.52 | 0.0003–0.0030 | 0.0002–0.0012 | 0.28–0.41 | 0.17–0.56 | 0.29–0.34 |
| VPD subcomponent | 1 | 0.98–1.02 | 0.0017–0.0053 | 0.0003–0.0019 | 0.49–0.77 | 0.31–1.01 | 0.40–0.47 |
| Rank-one LoRA | 0.5 | 0.45–0.51 | 0.027–0.114 | −0.0002 to 0.0001 | 0.05–0.08 | 0.04–0.16 | 0.12–0.17 |
| Rank-one LoRA | 1 | 0.92–1.00 | 0.053–0.236 | −0.0004 to 0.0001 | 0.08–0.15 | 0.08–0.25 | 0.15–0.24 |
| Full-matrix fine-tuning | 0.5 | 0.45–0.51 | 0.058–0.201 | 0.0007–0.0014 | 0.08–0.21 | 0.08–0.24 | 0.07–0.08 |
| Full-matrix fine-tuning | 1 | 0.92–1.03 | 0.105–0.407 | 0.0010–0.0020 | 0.16–0.42 | 0.15–0.49 | 0.10–0.11 |
| Neuron edits | 0.5 | 0.48–0.52 | 0.423–0.523 | −0.0008 to 0.0010 | 3.29–5.63 | 0.61–5.45 | 1.12–3.16 |
| Neuron edits | 1 | 0.98–1.00 | 0.756–0.876 | 0.013–0.038 | 13.38–38.67 | 11.85–39.86 | 5.85–9.80 |

: Confirmation effects on 58 dates and development English KL. Each method–τ row covers 3 targets × 3 draws; VPD repeats one fixed edit per target. Confirmation ranges span these 9 run entries. Development English KL spans all four English sets per run: the validation set and three evaluation sets. CE rises and KL are in nat/token. C: confirmation; D: development. KL is from the base model; edit norm is the Frobenius norm of the weight change. {#tbl-effects tbl-colwidths="[13,4,10,14,17,14,14,10]"}
