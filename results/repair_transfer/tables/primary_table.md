| Repair | Recovery of Dutch damage (%) | 95% CI |
|:--|--:|:--|
| Donor initialization (both factors trained) | 98.2 | [97.6, 98.9] |
| Scalar coefficients only + full-batch optimization | 86.9 | [85.5, 88.3] |
| Difference (percentage points) | +11.3 | [+10.0, +12.7] |

: Primary target, 2,048 fit tokens, 92 readout dates. 95% intervals from a joint bootstrap over dates (5,000 draws). Donor-initialized fitting exceeds the tested scalar-coefficient search by more than 3 percentage points. The scalar search includes 4 learning rates × 600 steps plus two full-batch optimizations, with 50 candidates in total. Selection considers both ordinary training checkpoints and converged candidates. {#tbl-primary}
