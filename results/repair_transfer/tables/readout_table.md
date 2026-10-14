| Target | Repair | Recovery (%) [95% CI] | KL recovery | German effect kept | German drift | Worst retain change |
|:--|:--|:--|--:|--:|--:|--:|
| Primary | Scalar coefficients only | 86.9 [85.5, 88.3] | 86.7 | 1.005 | 0.0003 | +0.0003 |
| Primary | Scalar coefficients only + expanded search | 86.9 [85.5, 88.3] | 86.7 | 1.005 | 0.0003 | +0.0003 |
| Primary | Donor initialization | 98.2 [97.6, 98.9] | 98.3 | 0.996 | 0.0005 | −0.0007 |
| Primary | Donor initialization, output only | 97.1 [96.4, 97.7] | 98.1 | 1.003 | 0.0003 | −0.0006 |
| Primary | Random initialization | 97.6 [97.0, 98.3] | 98.2 | 0.988 | 0.0005 | −0.0009 |
| Primary | Random initialization, second seed | 97.2 [96.5, 98.0] | 97.5 | 0.990 | 0.0005 | −0.0012 |
| Primary | Fresh | 97.2 [96.5, 97.8] | 97.7 | 0.990 | 0.0005 | −0.0014 |
| Primary | Fresh, second seed | 98.2 [97.5, 98.9] | 97.8 | 0.985 | 0.0006 | −0.0015 |
| Primary | Fixed donor repair | 86.3 [84.8, 87.9] | 84.4 | 0.999 | 0.0002 | −0.0001 |
| Replication | Scalar coefficients only | 77.3 [75.7, 79.0] | 76.5 | 1.007 | 0.0002 | +0.0003 |
| Replication | Scalar coefficients only + expanded search | 78.4 [76.7, 80.0] | 77.8 | 1.007 | 0.0003 | +0.0004 |
| Replication | Donor initialization | 98.8 [98.1, 99.4] | 98.6 | 1.001 | 0.0003 | −0.0005 |
| Replication | Donor initialization, output only | 97.4 [96.8, 98.1] | 98.2 | 1.007 | 0.0002 | −0.0006 |
| Replication | Random initialization | 98.7 [98.0, 99.4] | 97.9 | 0.990 | 0.0004 | −0.0010 |
| Replication | Random initialization, second seed | 97.5 [96.8, 98.3] | 97.8 | 0.996 | 0.0004 | −0.0009 |
| Replication | Fresh | 97.4 [96.7, 98.0] | 98.1 | 0.994 | 0.0004 | −0.0012 |
| Replication | Fresh, second seed | 98.0 [97.4, 98.6] | 98.4 | 0.990 | 0.0004 | −0.0012 |
| Replication | Fixed donor repair | 58.6 [55.6, 61.4] | 48.0 | 1.009 | 0.0003 | −0.0001 |

: Readout of the 2,048-token cells (600-step cap for ordinary runs) and the fixed donor repair on the 92 readout dates. Rows marked '+ expanded search' report the repair selected from ordinary checkpoints and two final candidates from additional runs that met the declared stopping criteria. KL recovery (%) is the share of the fine-tune's Dutch KL from the base model that the repair removes (required to be positive). German effect kept is the ratio of German CE damage with and without the repair (required 0.9–1.1). German drift is the repair's German KL from the fine-tuned model relative to the base model's KL from the fine-tuned model (required at most 0.05). Worst retain change is the largest CE change in English, French, Spanish or Italian against the fine-tuned model in nat/token (required at most +0.01). {#tbl-readout tbl-colwidths="[13,25,20,10,10,10,12]"}
