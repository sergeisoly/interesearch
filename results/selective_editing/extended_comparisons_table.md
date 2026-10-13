| Pair | τ | Comparator | Runs (stable) | Protected, all runs | Protected, stable runs | Target rise, subcomponent / comparator |
|:--|:--|:--|:--|:--|:--|:--|
| de → nl | 0.5 | Whitened rank-one selection | 3 | −0.076 [−0.084, −0.068] |  | 0.49 / 0.50 |
| de → nl | 0.5 | Rank-one LoRA, longer training | 9 (8) | −0.044 [−0.050, −0.038] | −0.044 [−0.050, −0.038] | 0.49 / 0.49 |
| de → nl | 0.5 | Full-matrix fine-tuning, longer training | 9 (9) | −0.098 [−0.111, −0.086] | −0.098 [−0.111, −0.086] | 0.49 / 0.51 |
| de → nl | 0.5 | SVD component | 0 of 3 | no estimate: 3 failed the guard |  |  |
| de → nl | 1 | Whitened rank-one selection | 3 | −0.170 [−0.183, −0.157] |  | 1.00 / 0.98 |
| de → nl | 1 | Rank-one LoRA, longer training | 9 (8) | −0.073 [−0.081, −0.064] | −0.072 [−0.080, −0.064] | 1.00 / 0.96 |
| de → nl | 1 | Full-matrix fine-tuning, longer training | 9 (9) | −0.181 [−0.200, −0.163] | −0.181 [−0.199, −0.164] | 1.00 / 0.99 |
| de → nl | 1 | SVD component | 0 of 3 | no estimate: 3 failed the guard |  |  |
| es → pt | 0.5 | Whitened rank-one selection | 3 | −0.201 [−0.216, −0.186] |  | 0.53 / 0.49 |
| es → pt | 0.5 | Rank-one LoRA, longer training | 9 (6) | −0.105 [−0.117, −0.092] | −0.110 [−0.122, −0.097] | 0.53 / 0.50 |
| es → pt | 0.5 | Full-matrix fine-tuning, longer training | 9 (9) | −0.170 [−0.186, −0.155] | −0.170 [−0.185, −0.156] | 0.53 / 0.52 |
| es → pt | 0.5 | SVD component | 0 of 3 | no estimate: 3 failed the guard |  |  |
| es → pt | 1 | Whitened rank-one selection | 3 | −0.442 [−0.471, −0.414] |  | 1.05 / 0.99 |
| es → pt | 1 | Rank-one LoRA, longer training | 9 (6) | −0.183 [−0.202, −0.164] | −0.195 [−0.214, −0.176] | 1.05 / 1.02 |
| es → pt | 1 | Full-matrix fine-tuning, longer training | 9 (9) | −0.337 [−0.363, −0.310] | −0.337 [−0.363, −0.311] | 1.05 / 1.04 |
| es → pt | 1 | SVD component | 0 of 3 | no estimate: 2 not reached, 1 failed the guard |  |  |
| fr → it | 0.5 | Whitened rank-one selection | 3 | −0.027 [−0.031, −0.023] |  | 0.53 / 0.49 |
| fr → it | 0.5 | Rank-one LoRA, longer training | 9 (6) | −0.017 [−0.021, −0.013] | −0.017 [−0.021, −0.014] | 0.53 / 0.54 |
| fr → it | 0.5 | Full-matrix fine-tuning, longer training | 9 (9) | −0.039 [−0.043, −0.035] | −0.039 [−0.043, −0.035] | 0.53 / 0.49 |
| fr → it | 0.5 | SVD component | 0 of 3 | no estimate: 3 failed the guard |  |  |
| fr → it | 1 | Whitened rank-one selection | 3 | −0.050 [−0.056, −0.045] |  | 1.06 / 0.99 |
| fr → it | 1 | Rank-one LoRA, longer training | 9 (6) | −0.027 [−0.032, −0.021] | −0.027 [−0.032, −0.022] | 1.06 / 1.08 (1.07) |
| fr → it | 1 | Full-matrix fine-tuning, longer training | 9 (9) | −0.070 [−0.076, −0.065] | −0.070 [−0.075, −0.065] | 1.06 / 1.00 |
| fr → it | 1 | SVD component | 0 of 3 | no estimate: 2 failed the guard, 1 not reached |  |  |

: Exploratory comparisons on the 64 development dates: VPD subcomponent minus comparator, difference in protected CE rise in nat/token, with simultaneous 95% intervals. All runs and stable runs (learned updates without unstable runs) are separate families. Every interval lies below zero and none lies within ±0.02, in both families. Runs: the comparator's matched and usable runs (draws, times three seeds for the learned updates), and in parentheses those that are stable. Each contrast includes all three VPD subcomponent draws. Target rise: the achieved target-test CE rise, mean over all runs, and in parentheses over stable runs where it differs. {#tbl-extended-comparisons tbl-colwidths="[8,4,18,7,23,23,17]"}
