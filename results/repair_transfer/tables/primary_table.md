| Repair | Recovery of Dutch damage (%) | 95% CI |
|:--|--:|:--|
| Donor initialization (both factors trained) | 98.2 | [97.6, 98.9] |
| Scalar coefficients only + expanded search | 86.9 | [85.5, 88.3] |
| Difference (percentage points) | +11.3 | [+10.0, +12.7] |

: Primary target, 2,048 fit tokens, 92 readout dates. 95% intervals from a joint bootstrap over dates (5,000 draws). Donor-initialized fitting exceeds the tested scalar-coefficient search by more than 3 percentage points. The scalar search includes 48 checkpoints from 4 learning-rate runs with a 600-step cap and two final candidates from additional full-batch runs that met the declared stopping criteria. Selection considers all 50 candidates and can retain an ordinary checkpoint. {#tbl-primary}
