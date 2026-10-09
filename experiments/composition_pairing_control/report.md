# Composition-conditioned semantic pairing control

**Complete and audited.**

This exploratory analysis reuses fourteen frozen, validation-selected SFTGNN-GPSR checkpoints. Exact reduced composition is the primary donor restriction; element set is the secondary restriction. There is no training, checkpoint reselection, or outcome-based change of tasks, seeds, grouping, or the twenty draws.

Audit status: PASS; audited 28/28 fixed checkpoint–group results. Missing: 0; audit errors: 0.

## Results on the same eligible subset

The correct, OFF, conditional, and global-context MAEs in each row use exactly the same eligible crystals. Global context is sliced from the earlier whole-test global donor assignments; its donor pool differs from the conditional pool.

| Group | Task | Seed | Eligible/full N | Coverage | Distinct maps/20 | Correct MAE | OFF MAE | Conditional MAE | Global-context MAE | Correct advantage vs conditional, % [descriptive 95% CI] | Coverage flag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| reduced_composition | Jarvis-Bandgap_MBJ | 42 | 167/1818 | 9.19% | 20/20 | 0.3134 | 0.333821 | 0.322084 | 0.374055 | 2.69592 [-7.7923, 12.5856] | adequate N; scope remains subset-specific |
| element_set | Jarvis-Bandgap_MBJ | 42 | 347/1818 | 19.09% | 20/20 | 0.244754 | 0.265917 | 0.253085 | 0.294841 | 3.29184 [-3.50012, 9.92643] | adequate N; scope remains subset-specific |
| reduced_composition | Jarvis-Bandgap_OPT | 42 | 679/5573 | 12.18% | 20/20 | 0.175521 | 0.197306 | 0.185729 | 0.226129 | 5.49599 [-0.107864, 10.428] | adequate N; scope remains subset-specific |
| element_set | Jarvis-Bandgap_OPT | 42 | 1713/5573 | 30.74% | 20/20 | 0.144532 | 0.157973 | 0.155731 | 0.175586 | 7.19084 [3.11846, 11.0944] | adequate N; scope remains subset-specific |
| reduced_composition | Jarvis-FormationEnergy | 42 | 679/5573 | 12.18% | 20/20 | 0.0356043 | 0.0428595 | 0.0383959 | 0.0463252 | 7.27059 [1.28528, 13.0808] | adequate N; scope remains subset-specific |
| element_set | Jarvis-FormationEnergy | 42 | 1713/5573 | 30.74% | 20/20 | 0.0340895 | 0.0394828 | 0.0368108 | 0.0420844 | 7.39275 [4.50386, 10.2498] | adequate N; scope remains subset-specific |
| reduced_composition | Jarvis-BulkModulusKv | 42 | 169/1968 | 8.59% | 20/20 | 6.73821 | 7.73955 | 7.50723 | 8.35809 | 10.2438 [2.26603, 18.5198] | adequate N; scope remains subset-specific |
| element_set | Jarvis-BulkModulusKv | 42 | 370/1968 | 18.80% | 20/20 | 8.16386 | 8.8434 | 8.83231 | 9.49156 | 7.56828 [3.30781, 12.1669] | adequate N; scope remains subset-specific |
| reduced_composition | MP-Bandgap | 42 | 12/516 | 2.33% | 1/20 | 0.095236 | 0.114821 | 0.0962794 | 0.145356 | 1.08373 [-38.4853, 10.9433] | eligible N<30; only one distinct donor map |
| element_set | MP-Bandgap | 42 | 90/516 | 17.44% | 20/20 | 0.242366 | 0.263491 | 0.261108 | 0.289892 | 7.17812 [-0.449098, 15.5984] | adequate N; scope remains subset-specific |
| reduced_composition | MP-BulkModuli | 42 | 45/583 | 7.72% | 15/20 | 0.0475145 | 0.0681092 | 0.0669327 | 0.0851919 | 29.0115 [4.83984, 45.1357] | adequate N; scope remains subset-specific |
| element_set | MP-BulkModuli | 42 | 117/583 | 20.07% | 20/20 | 0.0414964 | 0.0600215 | 0.0631307 | 0.0740691 | 34.269 [16.6382, 45.8404] | adequate N; scope remains subset-specific |
| reduced_composition | Jarvis-Bandgap_MBJ | 7 | 167/1818 | 9.19% | 20/20 | 0.315588 | 0.349952 | 0.332897 | 0.364817 | 5.19967 [0.039733, 11.0752] | adequate N; scope remains subset-specific |
| element_set | Jarvis-Bandgap_MBJ | 7 | 347/1818 | 19.09% | 20/20 | 0.269724 | 0.2909 | 0.276445 | 0.303965 | 2.43136 [-2.91821, 7.8261] | adequate N; scope remains subset-specific |
| reduced_composition | Jarvis-BulkModulusKv | 7 | 169/1968 | 8.59% | 20/20 | 7.13345 | 7.44868 | 7.59448 | 8.18842 | 6.07055 [-9.22887, 18.6558] | adequate N; scope remains subset-specific |
| element_set | Jarvis-BulkModulusKv | 7 | 370/1968 | 18.80% | 20/20 | 8.24787 | 8.82391 | 8.87985 | 9.38299 | 7.11701 [0.153308, 12.9255] | adequate N; scope remains subset-specific |
| reduced_composition | Jarvis-FormationEnergy | 7 | 679/5573 | 12.18% | 20/20 | 0.0363541 | 0.0438144 | 0.0391485 | 0.0447104 | 7.1379 [2.90104, 11.6156] | adequate N; scope remains subset-specific |
| element_set | Jarvis-FormationEnergy | 7 | 1713/5573 | 30.74% | 20/20 | 0.0337826 | 0.0390995 | 0.0358494 | 0.0398935 | 5.76527 [3.41563, 8.3333] | adequate N; scope remains subset-specific |
| reduced_composition | MP-Bandgap | 7 | 12/516 | 2.33% | 1/20 | 0.127226 | 0.473118 | 0.115649 | 0.409092 | -10.0107 [-43.6644, 3.46078] | eligible N<30; only one distinct donor map |
| element_set | MP-Bandgap | 7 | 90/516 | 17.44% | 20/20 | 0.262631 | 0.32229 | 0.272943 | 0.344292 | 3.77798 [-7.18067, 14.8548] | adequate N; scope remains subset-specific |
| reduced_composition | Jarvis-Bandgap_MBJ | 1234 | 167/1818 | 9.19% | 20/20 | 0.31039 | 0.3225 | 0.309887 | 0.358876 | -0.162273 [-9.8646, 8.33442] | adequate N; scope remains subset-specific |
| element_set | Jarvis-Bandgap_MBJ | 1234 | 347/1818 | 19.09% | 20/20 | 0.266082 | 0.278966 | 0.264025 | 0.309303 | -0.779031 [-6.8758, 5.01233] | adequate N; scope remains subset-specific |
| reduced_composition | Jarvis-BulkModulusKv | 1234 | 169/1968 | 8.59% | 20/20 | 7.01895 | 7.31363 | 7.47252 | 8.06007 | 6.06982 [-2.17296, 14.8855] | adequate N; scope remains subset-specific |
| element_set | Jarvis-BulkModulusKv | 1234 | 370/1968 | 18.80% | 20/20 | 8.18813 | 8.71468 | 8.85447 | 9.36348 | 7.5255 [3.05226, 12.2755] | adequate N; scope remains subset-specific |
| reduced_composition | Jarvis-FormationEnergy | 1234 | 679/5573 | 12.18% | 20/20 | 0.0369214 | 0.043383 | 0.0397297 | 0.0462565 | 7.06868 [2.55282, 11.678] | adequate N; scope remains subset-specific |
| element_set | Jarvis-FormationEnergy | 1234 | 1713/5573 | 30.74% | 20/20 | 0.0343758 | 0.0394832 | 0.0368108 | 0.0415782 | 6.61487 [4.18183, 8.8951] | adequate N; scope remains subset-specific |
| reduced_composition | MP-Bandgap | 1234 | 12/516 | 2.33% | 1/20 | 0.119072 | 0.129241 | 0.121553 | 0.217365 | 2.04059 [-22.6126, 18.3921] | eligible N<30; only one distinct donor map |
| element_set | MP-Bandgap | 1234 | 90/516 | 17.44% | 20/20 | 0.25508 | 0.297967 | 0.282757 | 0.323772 | 9.78828 [1.03187, 19.3747] | adequate N; scope remains subset-specific |

## Normalized summaries by grouping and training seed

Physical-unit MAEs are not averaged across properties. These summaries average task-specific relative effects and do not treat repeated seeds as independent tasks.

| Group | Seed | Audited/required tasks | Eligible tasks | Correct better | Mean task advantage (%) | Weak-coverage eligible tasks |
|---|---:|---:|---:|---:|---:|---:|
| reduced_composition | 42 | 6/6 | 6 | 6 | 9.30026 | 1 |
| reduced_composition | 7 | 4/4 | 4 | 3 | 2.09936 | 1 |
| reduced_composition | 1234 | 4/4 | 4 | 3 | 3.75421 | 1 |
| element_set | 42 | 6/6 | 6 | 6 | 11.1485 | 0 |
| element_set | 7 | 4/4 | 4 | 4 | 4.7729 | 0 |
| element_set | 1234 | 4/4 | 4 | 3 | 5.7874 | 0 |

## Scientific interpretation and limits

A correct-pairing advantage under exact reduced-composition restriction is evidence that the frozen checkpoint responds to structural correspondence among the covered same-composition crystals. Element-set restriction is chemically weaker because stoichiometry may change. A null or adverse conditional effect is retained and does not invalidate the previously observed global effect.

The contrast between global and composition-conditioned effects is **not a causal percentage decomposition of composition and structural information**. Donor similarity, subgroup coverage, description duplication, and distribution shift differ. These results also do not establish gains over an independently trained graph-only model or explain CartNet's independently trained MP-bulk regression.

Intervals use 2,000 paired bootstrap resamples of crystals after averaging conditional absolute error within each crystal over the twenty assignment draws. They describe this fixed checkpoint and subset. Crystals share composition groups and donor assignments, so they are statistically dependent; these crystal-bootstrap intervals may understate composition-cluster uncertainty and are not confirmatory population-level significance tests.

Twenty draws can repeat the same donor map. Distinct-map counts and the preflight mapping-space sizes must accompany interpretation. Eligible N<30 or one distinct map is explicitly marked weak coverage and retained. Groups whose descriptions or embeddings are identical can change identities without changing semantic inputs; equality rates are in task_summary.csv.

All available groups are reported. Full coverage, null/negative outcomes, and ineligible groups are preserved without selection by effect size. The fixed split, limited seeds, checkpoint co-adaptation, and post-hoc use of test labels restrict generalization.

## Integrity

The auditor checks the fixed 14×2 schedule, sample identity/order, full-test donor indices, twenty eligible bijections, no fixed points, exact group membership, description/embedding equality rates, finite predictions, CSV/NPZ/JSON metric agreement, unchanged tau bound (maximum <=0.1001), and formal/prior replay differences <=1e-5 in native standardized output units. Input hashes and per-draw donor hashes are recorded in validation.json.
