# CartNet + GPSR strong-backbone validation

## Decision

Phase 0 passed all four official-reproduction checks. The fixed-split Phase 1 comparison completed all 10 matched CartNet versus CartNet+GPSR pairs at seed 42. The preregistered verdict is **NO_STRONG_BACKBONE_GAIN**: macro GN-MAE is **1.0050**, mean relative gain is **-0.50%**, and GPSR improves **6/10** tasks. Because macro GN-MAE is at least 1.0, Phase 2 was not run; seeds 7 and 1234 were not started, and no GPSR hyperparameter was changed.

## Frozen protocol

- Official CartNet revision: `1c794049db48badfe71f7030b58ea50665ba9fd5`
- Runtime: Python 3.10.12, PyTorch 2.2.2+cu121, PyG 2.5.2, CUDA 12.1, RTX 4090
- CartNet: cutoff 5.0, all neighbors within the radius, hidden width 256, four interaction layers
- Training: Adam, learning rate 1e-3, OneCycleLR with `pct_start=0.01`, batch size 64, 500 epochs
- Selection: minimum validation MAE; the selected checkpoint was evaluated on the test set once
- GPSR: frozen 768-dimensional MatBERT input, injection after `CartNet.layers[0]`, Linear-GELU-Linear projector, NodeGate, channel-wise alpha initialized to 1e-3, and sample-wise Frobenius projection with tau=0.10
- All 10 external split manifests are disjoint and have exact expected-versus-loaded sample counts.

## Phase 0: official implementation sanity check

| Anchor | Retrained test MAE | Official reference | Relative error | Result |
|---|---:|---:|---:|---|
| JARVIS Bandgap MBJ | 0.251049 | 0.25303 | 0.78% | PASS |
| JARVIS Bandgap OPT | 0.117355 | 0.11531 | 1.77% | PASS |
| MP Bandgap | 0.189438 | 0.1908 | 0.71% | PASS |
| MP Bulk Modulus | 0.033881 | 0.0330 | 2.67% | PASS |

All four errors are below the 10% reproduction threshold, so the CartNet implementation is on the expected performance scale.

## Phase 1: matched 10-task result

| Task | Category | CartNet | CartNet+GPSR | Relative gain | Improved |
|---|---|---:|---:|---:|:---:|
| Jarvis-Bandgap_MBJ | Electronic | 0.247743 | 0.232509 | +6.15% | Yes |
| Jarvis-Bandgap_OPT | Electronic | 0.112505 | 0.111859 | +0.57% | Yes |
| Jarvis-FormationEnergy | Energy | 0.027078 | 0.027384 | -1.13% | No |
| Jarvis-TotalEnergy | Energy | 0.026709 | 0.026427 | +1.06% | Yes |
| Jarvis-BulkModulusKv | Mechanical | 8.788755 | 8.603874 | +2.10% | Yes |
| Jarvis-ShearModulusGv | Mechanical | 8.735687 | 8.883594 | -1.69% | No |
| MP-Bandgap | Electronic | 0.173546 | 0.176348 | -1.61% | No |
| MP-FormationEnergy | Energy | 0.021771 | 0.021739 | +0.15% | Yes |
| MP-BulkModuli | Mechanical | 0.030244 | 0.033991 | -12.39% | No |
| MP-ShearModuli | Mechanical | 0.071540 | 0.070281 | +1.76% | Yes |

The largest positive effect is on JARVIS MBJ bandgap (+6.15%). The dominant negative result is MP Bulk Moduli (-12.39%), which moves the macro average above parity despite six task-level improvements.

## Category dependence

| Category | Macro GN-MAE | Mean relative gain | Improved tasks |
|---|---:|---:|---:|
| Electronic | 0.9830 | +1.70% | 2/3 |
| Energy | 0.9998 | +0.02% | 2/3 |
| Mechanical | 1.0255 | -2.55% | 2/4 |

The earlier Electronic > Energy > Mechanical ordering remains visible, but the CartNet result is not uniformly positive. Electronic tasks retain useful headroom; energy tasks are effectively neutral; mechanical tasks are negative on average.

## Constraint and representation checks

- Maximum observed `rho_final`: **0.10000017**, below the required 0.1001 threshold.
- Mean `rho_final` across tasks: approximately **0.09988**.
- Mean clipped fraction: approximately **99.43%**; the hard projection therefore remains active for nearly all samples.
- Mean projection scale: approximately **0.1260**.
- Mean cosine similarity `cos(H,H')`: approximately **0.99508**, consistent with a bounded perturbation.
- All graph-path equivalence checks pass; maximum output error is at floating-point scale and below 1e-7.
- Tau=0.10 required no retuning and was not changed.

## Cross-backbone interpretation

| Backbone | Macro GN-MAE | Mean gain | Improved evaluations |
|---|---:|---:|---:|
| SFTGNN | 0.9664 | +3.36% | 29/30 |
| ALIGNN | 0.9703 | +2.97% | 26/30 |
| CartNet | 1.0050 | -0.50% | 6/10 |

The strong CartNet backbone removes the average gain seen with SFTGNN and ALIGNN. This is evidence of diminishing semantic headroom: a stronger geometric encoder can already recover part of the abstraction that GPSR supplied to weaker backbones. The positive results on six tasks, especially electronic properties, show task-dependent transfer, but they do not support a claim of consistent improvement on CartNet.

## Answers to the preregistered questions

1. **CartNet reproduction:** normal; all four anchors pass with 0.71–2.67% relative error.
2. **CartNet performance on the fixed split:** the complete per-task baselines are reported above and in `cartnet_seed1_10task.csv`.
3. **Does GPSR improve CartNet overall?** No. Macro GN-MAE is 1.0050.
4. **How many tasks improve?** 6 of 10.
5. **Macro gain:** -0.50%.
6. **Does the stronger backbone shrink semantic gain?** Yes; the average gain falls from roughly 3% on SFTGNN/ALIGNN to -0.50% on CartNet.
7. **Was tau retuned?** No; tau remained 0.10 throughout.
8. **Is hard projection frequently active?** Yes; the mean clipped fraction is about 99.43%.
9. **Which category benefits most?** Electronic (+1.70%), followed by Energy (+0.02%); Mechanical is negative (-2.55%).
10. **Is there a consistent trend across all three backbones?** No. SFTGNN and ALIGNN are positive, while CartNet is slightly negative overall.
11. **Are absolute MAEs near published CartNet performance?** Phase 0 establishes the expected CartNet scale. Phase 1 uses the paper's fixed data and splits, so direct SOTA claims against literature values are not valid without identical protocols.
12. **Can the paper claim effectiveness on a recent high-performing backbone?** Not as a consistent overall gain. A defensible statement is that GPSR transfers selectively to CartNet, with gains concentrated in electronic properties, while aggregate improvement disappears under the preregistered metric.

## Final disposition

The gate requires Phase 2 to stop, so no additional seeds or tuning were run. The result should be reported as a negative aggregate strong-backbone validation with clear property dependence, rather than omitted or reframed as a positive result.
