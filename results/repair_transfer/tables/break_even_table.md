| Currency | Target | Fresh | Fresh cell | Donor init. | Donor init. cell | Break-even K\* |
|:--|:--|--:|:--|--:|:--|--:|
| Unique tokens | Primary | 57,600 | 256 / 600 | 57,600 | 256 / 600 | no per-target saving |
|  | Replication | 57,600 | 256 / 600 | 57,600 | 256 / 600 | no per-target saving |
| Fit-token presentations | Primary | 409,600 | 2,048 / ≤50 | 51,200 | 256 / ≤50 | 10.3 |
|  | Replication | 409,600 | 2,048 / ≤50 | 51,200 | 256 / ≤50 | 10.3 |
| GPU seconds (estimate) | Primary | 16.8 | 2,048 / ≤50 | 14.5 | 256 / ≤50 | 151.4 |
|  | Replication | 17.1 | 2,048 / ≤50 | 13.9 | 256 / ≤50 | 105.9 |

: Target-side cost of the cheapest cell of fresh fitting and of donor initialization that reaches the reference (fresh at 2,048 tokens and up to 600 steps), found separately in each currency. A cell is written as unique fit tokens / step cap. K\* = donor cost / (fresh cost − donor-initialization cost) is a modelled amortization: reuse pays off for more than K\* targets. The donor's one-time cost and what each currency counts and leaves out are in the [cost-accounting appendix](#sec-app-cost). GPU seconds are an estimate of allocated shared work, and the donor's come from a different run. {#tbl-break-even tbl-colwidths="[17,13,10,14,10,13,23]"}
