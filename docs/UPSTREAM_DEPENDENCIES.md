# Upstream dependencies

The release does not vendor third-party repositories.

| Component | Source / revision | Use |
| --- | --- | --- |
| SFTGNN | commit `4b5ac55c4f4e9ec88f1854e93fc4db4b2147178d` | primary graph backbone |
| CartNet | commit `1c794049db48badfe71f7030b58ea50665ba9fd5` | strong geometric backbone |
| ALIGNN | public reference implementation used by the experiment preflight | transfer backbone |
| MatBERT | `matbert-base-cased` checkpoint | frozen text representation |

Review the licenses and citation requirements of each dependency before
redistributing its source, weights, or derived dataset artifacts.
