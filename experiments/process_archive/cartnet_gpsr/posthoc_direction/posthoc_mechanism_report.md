# CartNet GPSR Post-hoc Residual Direction Analysis

## Scope and safeguards

This is an exploratory post-hoc analysis of the already selected seed-42 minimum-validation-MAE CartNet-GPSR checkpoints. No training, fine-tuning, checkpoint reselection, task substitution, or change to tau, injection depth, NodeGate, alpha, or the projector was performed. Test labels are used only for frozen-checkpoint mechanism analysis.

Two comparisons are kept separate throughout:

1. **Formal performance comparison:** independently trained CartNet versus independently trained CartNet+GPSR.
2. **Mechanistic counterfactual:** the same trained CartNet+GPSR checkpoint evaluated with its semantic residual OFF and ON.

Residual-OFF is therefore not a CartNet baseline. It is a within-model counterfactual that probes how the trained GPSR model uses its residual after graph and semantic parameters have co-adapted.

## Formal performance result retained unchanged

| Task | Independent CartNet MAE | Independent CartNet+GPSR MAE | Formal relative gain |
|---|---:|---:|---:|
| J-MBJ | 0.247743 | 0.232509 | +6.15% |
| J-Bulk | 8.788755 | 8.603874 | +2.10% |
| MP-Bulk | 0.030244 | 0.033991 | -12.39% |

The aggregate CartNet conclusion remains `NO_STRONG_BACKBONE_GAIN` and is not revised by this analysis.

## Checkpoint replay and within-model counterfactual

| Task | Test N | Selected epoch | Residual-OFF MAE | Residual-ON MAE | Samples improved by ON |
|---|---:|---:|---:|---:|---:|
| J-MBJ | 1,818 | 343 | 0.295281 | 0.232509 | 69.14% [66.94%, 71.18%] |
| J-Bulk | 1,968 | 358 | 11.630696 | 8.603874 | 65.40% [63.26%, 67.48%] |
| MP-Bulk | 583 | 404 | 0.055983 | 0.033991 | 75.13% [71.70%, 78.56%] |

Residual-ON replay reproduced each stored test MAE within numerical tolerance. The MP-Bulk residual is beneficial within its own trained GPSR checkpoint: switching it off increases MAE from 0.033991 to 0.055983. Consequently, the formal -12.39% result cannot be attributed to an immediately harmful residual direction inside the final GPSR checkpoint. The independently trained CartNet solution is simply better (0.030244), whereas the GPSR graph pathway has co-adapted to rely on the semantic correction.

## Magnitude and projection

| Task | Median rho_raw | Mean rho_raw | Median projection scale | Mean projection scale | Median rho_final | Clipped fraction |
|---|---:|---:|---:|---:|---:|---:|
| J-MBJ | 0.7582 | 0.8046 | 0.1319 | 0.1576 | 0.100000 | 100.00% |
| J-Bulk | 1.5360 | 1.6145 | 0.0651 | 0.0816 | 0.100000 | 100.00% |
| MP-Bulk | 0.3799 | 0.4053 | 0.2632 | 0.3273 | 0.100000 | 97.60% |

MP-Bulk has the smallest raw relative residual and the largest projection scale. Its raw residual is therefore not abnormally large, and its projection is not stronger than in either successful task. H1 (raw-magnitude conflict) is not supported.

All three median `rho_final` values equal 0.10 to floating-point precision. Because the hard bound makes `rho_final` nearly constant, it has little sample-wise discriminative value. The relevant comparisons are raw magnitude, projection scale, direction, and output sensitivity.

## Direction and sample-level outcome

| Task | Median helpful alignment | Bootstrap 95% CI | Median target-directed shift | Median normalized error gain | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| J-MBJ | 0.1470 | [0.1298, 0.1650] | 0.000024 | 0.000027 | [0.000016, 0.000040] |
| J-Bulk | 0.1186 | [0.1079, 0.1297] | 1.742451 | 0.089338 | [0.078825, 0.099900] |
| MP-Bulk | 0.3916 | [0.3709, 0.4157] | 0.033861 | 0.415909 | [0.363435, 0.454393] |

MP-Bulk has the most positive helpful alignment, not the least. Its median target-directed shift and normalized error gain are also positive. H2 (MP-Bulk direction mismatch) is not supported for the final trained checkpoint.

Helpful alignment is associated with sample-level improvement within every task:

| Task | Spearman(helpful alignment, normalized error gain) |
|---|---:|
| J-MBJ | 0.592 |
| J-Bulk | 0.773 |
| MP-Bulk | 0.668 |

This supports the usefulness of the directional diagnostic at the sample level, while showing that MP-Bulk's formal task-level degradation is not explained by a globally adverse residual direction.

Raw residual magnitude has little monotonic association with outcome: the Spearman correlations between `rho_raw` and normalized error gain are 0.063 (J-MBJ), -0.125 (J-Bulk), and 0.067 (MP-Bulk). Projection-scale correlations have the same small magnitudes with opposite signs because almost all samples are clipped.

## First-order approximation validity

| Task | Pearson(first-order shift, actual Delta y) | Spearman |
|---|---:|---:|
| J-MBJ | 0.664 | 0.905 |
| J-Bulk | 0.707 | 0.974 |
| MP-Bulk | 0.969 | 0.985 |

The first-order directional approximation tracks the actual prediction shift well in rank for all three tasks and especially well for MP-Bulk. The gradient-alignment interpretation is therefore informative for these checkpoints. The lower Pearson coefficients for J-MBJ and J-Bulk indicate some nonlinear scale distortion, so the approximation should not be treated as exact there.

## Downstream sensitivity

| Task | Median downstream sensitivity | Bootstrap 95% CI | Mean downstream sensitivity |
|---|---:|---:|---:|
| J-MBJ | 0.3470 | [0.1054, 0.7346] | 5.9755 |
| J-Bulk | 2.3318 | [2.1928, 2.5057] | 5.4280 |
| MP-Bulk | 6.8888 | [6.4549, 7.3770] | 8.6381 |

MP-Bulk has substantially greater median downstream sensitivity than both controls; the bootstrap intervals are well separated. H3 (downstream amplification) is supported. Similar injection-state perturbation ratios near 0.10 can therefore produce very different task-normalized output movements. A latent Frobenius norm bound constrains representation displacement but does not bound the Lipschitz sensitivity of the remaining CartNet blocks and regression head.

The sensitivity distribution is heavy-tailed, particularly for J-MBJ, which explains the difference between its median and mean. The median and its bootstrap interval are the primary comparison.

## Gate diagnostics

| Task | Mean gate | Mean gate std | Fraction gate values <0.1 | Fraction >0.9 | Mean binary entropy |
|---|---:|---:|---:|---:|---:|
| J-MBJ | 0.6901 | 0.4617 | 30.95% | 68.97% | 0.00057 |
| J-Bulk | 0.6084 | 0.4880 | 39.16% | 60.83% | 0.00005 |
| MP-Bulk | 0.5447 | 0.4964 | 45.31% | 54.26% | 0.00320 |

All three models have strongly saturated channel gates. MP-Bulk is not uniquely more saturated; its entropy is slightly larger, although still very small. Gate saturation does not isolate the MP-Bulk formal failure.

## Answers to the required questions

### Q1. Is MP-Bulk rho_raw abnormally large?

No. MP-Bulk has the smallest median `rho_raw` (0.3799), compared with 0.7582 for J-MBJ and 1.5360 for J-Bulk.

### Q2. Is the MP-Bulk projection scale abnormally small?

No. MP-Bulk has the largest median projection scale (0.2632), compared with 0.1319 and 0.0651. Its residual is less strongly contracted.

### Q3. Are all three rho_final values close to tau=0.10?

Yes. All medians are 0.100000 to floating-point precision; 97.60-100% of samples are clipped.

### Q4. Is MP-Bulk helpful alignment worse?

No. MP-Bulk has the highest median helpful alignment (0.3916), with a 95% bootstrap interval [0.3709, 0.4157].

### Q5. Is helpful alignment related to sample error improvement?

Yes. Within-task Spearman correlations are 0.592, 0.773, and 0.668 for J-MBJ, J-Bulk, and MP-Bulk, respectively. This is an exploratory association, not a causal estimate.

### Q6. Does the first-order gradient shift explain actual Delta y?

It explains the rank ordering strongly: Spearman correlations are 0.905-0.985. Pearson correlation is strongest for MP-Bulk (0.969) and moderate-to-strong for the two JARVIS tasks (0.664 and 0.707), where nonlinear scaling remains.

### Q7. Does MP-Bulk have higher downstream sensitivity?

Yes. Its median sensitivity is 6.8888, roughly 3.0 times J-Bulk and 19.9 times J-MBJ. The median bootstrap intervals do not overlap.

### Q8. Why do J-Bulk and MP-Bulk have opposite formal outcomes?

The final MP-Bulk GPSR checkpoint does not exhibit a harmful residual: its residual improves 75.13% of samples and substantially lowers its own OFF-path MAE. The opposite formal result arises when comparing two independently optimized models. This points to optimization and representation co-adaptation, combined with greater downstream sensitivity, rather than a larger or less target-aligned final residual. The present post-hoc design cannot identify the training trajectory that produced the inferior GPSR optimum.

### Q9. Which mechanism is best supported?

`DOWNSTREAM_AMPLIFICATION_SUPPORTED`, with an important qualification: amplification is present, but the amplified residual is mostly beneficial inside the final MP-Bulk GPSR checkpoint. Neither `RAW_MAGNITUDE_CONFLICT_SUPPORTED` nor `DIRECTION_MISMATCH_SUPPORTED` is supported. No residual-level mechanism in the final checkpoint fully explains why independently trained CartNet is better.

### Q10. Do the results support “bounded magnitude alone does not guarantee beneficial semantic direction”?

They support the separation between magnitude and direction at the sample level: 24.87-34.60% of samples are harmed despite the same hard bound, and helpful alignment correlates with error gain. They do not support using direction mismatch as the explanation for the MP-Bulk task-level failure, because MP-Bulk has the most positive direction statistics. The more directly supported statement is: **a latent norm bound does not imply bounded or uniformly beneficial prediction-level effects.**

### Q11. Which conclusions are only exploratory associations?

All relationships involving helpful alignment, sample error gain, sensitivity, gate saturation, and cross-task differences are exploratory associations from one frozen seed-42 checkpoint per task. They use test labels, do not randomize residual direction, and do not observe the training trajectory. They cannot establish that sensitivity or co-adaptation caused the formal MP-Bulk degradation.

## Mechanism verdict

The strongest supported mechanism is **downstream amplification**, not raw residual magnitude or adverse residual direction. However, the central negative formal result is a difference between independently trained optima, while residual OFF/ON is a within-checkpoint intervention after co-adaptation. The post-hoc analysis therefore yields the following bounded conclusion:

> Similar injection-state perturbation magnitudes can induce substantially different prediction-level effects across properties, demonstrating that the latent norm bound does not constrain downstream output sensitivity. In the final MP-Bulk GPSR checkpoint, the semantic residual is target-aligned and beneficial relative to its co-adapted residual-OFF pathway; therefore, the formal degradation relative to independently trained CartNet is not explained by larger or less helpful final residuals.

The formal CartNet aggregate verdict remains unchanged.
