| Pair | τ | Reference | Target rise | Protected, confirmation | Protected, development | Draws used |
|:--|:--|:--|:--|:--|:--|:--|
| de → nl | 0.5 | OMP from full-matrix fine-tuning | 0.49 / 0.51 | −0.072 [−0.080, −0.065] | −0.071 [−0.080, −0.063] | 0, 1, 2 |
| de → nl | 0.5 | OMP from rank-one LoRA | 0.48 / 0.50 | −0.024 [−0.027, −0.022] | −0.024 [−0.027, −0.021] | 0, 1, 2 |
| de → nl | 0.5 | CI-difference selection | 0.52 / 0.52 | −0.776 [−0.815, −0.737] | −0.789 [−0.823, −0.756] | 0, 1, 2 |
| de → nl | 0.5 | Previously chosen subcomponent | 0.48 / 0.49 | 0 (identical edit) | 0 (identical edit) | 0, 1, 2 |
| de → nl | 1 | OMP from full-matrix fine-tuning | 0.98 / 1.01 | −0.134 [−0.145, −0.123] | −0.133 [−0.145, −0.120] | 0, 1, 2 |
| de → nl | 1 | OMP from rank-one LoRA | 0.98 / 1.00 | −0.045 [−0.048, −0.041] | −0.044 [−0.048, −0.040] | 0, 1, 2 |
| de → nl | 1 | CI-difference selection | 0.98 / 0.99 | −1.369 [−1.428, −1.311] | −1.414 [−1.464, −1.365] | 0, 1, 2 |
| de → nl | 1 | Previously chosen subcomponent | 0.98 / 1.00 | 0 (identical edit) | 0 (identical edit) | 0, 1, 2 |
| es → pt | 0.5 | OMP from full-matrix fine-tuning | 0.51 / 0.51 | −0.185 [−0.202, −0.169] | −0.182 [−0.197, −0.167] | 0, 1, 2 |
| es → pt | 0.5 | OMP from rank-one LoRA | 0.51 / 0.51 | −0.185 [−0.202, −0.169] | −0.182 [−0.197, −0.167] | 2 |
| es → pt | 0.5 | CI-difference selection | 0.45 / 0.46 | −0.484 [−0.508, −0.460] | −0.494 [−0.520, −0.469] | 0, 1, 2 |
| es → pt | 1 | OMP from full-matrix fine-tuning | no edit | no estimate | no estimate | none |
| es → pt | 1 | OMP from rank-one LoRA | no edit | no estimate | no estimate | none |
| es → pt | 1 | CI-difference selection | 0.96 / 0.98 | −1.046 [−1.085, −1.007] | −1.083 [−1.126, −1.039] | 0, 1, 2 |
| fr → it | 0.5 | OMP from full-matrix fine-tuning | 0.50 / 0.52 | −0.425 [−0.448, −0.403] | −0.436 [−0.457, −0.415] | 0, 1, 2 |
| fr → it | 0.5 | OMP from rank-one LoRA | 0.47 / 0.49 | −0.618 [−0.653, −0.582] | −0.629 [−0.662, −0.596] | 0 |
| fr → it | 0.5 | CI-difference selection | 0.50 / 0.51 | −0.483 [−0.511, −0.455] | −0.489 [−0.513, −0.464] | 0, 1, 2 |
| fr → it | 1 | OMP from full-matrix fine-tuning | 1.01 / 1.05 | −0.470 [−0.493, −0.447] | −0.479 [−0.498, −0.460] | 0, 2 |
| fr → it | 1 | OMP from rank-one LoRA | 0.99 / 1.04 | −1.236 [−1.307, −1.165] | −1.271 [−1.329, −1.213] | 0 |
| fr → it | 1 | CI-difference selection | 0.95 / 1.00 | −1.339 [−1.403, −1.276] | −1.367 [−1.422, −1.312] | 0, 2 |

: Alternative subcomponent-selection comparisons. Target rise: the alternative edit's achieved target CE rise, averaged over the draws its contrast used, on confirmation / development data. The selected subcomponent's rise is in @tbl-interventions. Protected: VPD subcomponent minus reference, difference in protected CE rise, with simultaneous 95% intervals adjusted within this comparison family. These comparisons are descriptive. All values in nat/token. Draws used: the draws in which the reference has a matched, usable edit. With fewer than three the contrast is conditional, with none it has no estimate. {#tbl-references tbl-colwidths="[8,4,16,14,24,24,10]"}
