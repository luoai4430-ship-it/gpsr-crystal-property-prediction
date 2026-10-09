# Composition-conditioned semantic pairing control

This exploratory frozen-checkpoint analysis asks whether crystal–text
correspondence remains useful when donor descriptions have the same chemical
composition. The existing global permutation control changes both composition
and structural description; this conditional control isolates a narrower
question. It introduces no training, checkpoint selection, or hyperparameter
search. Manuscript files are not changed by this experiment.

## Fixed scope

- The same fourteen validation-selected bounded SFTGNN-GPSR checkpoints used
  for `semantic_pairing_permutation` are used unchanged.
- Seed 42: JARVIS MBJ band gap, OPT band gap, formation energy, bulk modulus,
  MP band gap, and MP bulk modulus.
- Seeds 7 and 1234: JARVIS MBJ band gap, formation energy, bulk modulus, and
  MP band gap. The original test sample IDs and their order must be identical
  across training seeds of a task.
- Primary grouping: exact **reduced composition**, with exact rational
  stoichiometric counts. Proportional formula units belong to the same group.
- Secondary grouping: the exact **element set**. This preserves element
  identity but permits different stoichiometric ratios.
- Each grouping receives twenty fixed, seeded within-group permutation draws.
  Only groups containing at least two test samples can be deranged. Every
  eligible sample must receive a donor from its own group, and every donor is
  used once per draw. There are no fixed points among eligible samples.
- Singleton samples remain in the preflight tables but are excluded from
  conditional performance summaries. No sample is silently discarded for
  missing or malformed metadata.

## Label-free preflight

Run with the standard-library Python interpreter:

```sh
python experiments/composition_pairing_control/preflight.py --self-test
python experiments/composition_pairing_control/preflight.py \
  --data-root /root/autodl-tmp/dataset1/dataset \
  --pairing-root experiments/semantic_pairing_permutation \
  --out experiments/composition_pairing_control
```

The preflight hashes the complete source description files, but only requires
valid metadata for the fixed test IDs used by this experiment. Unused training
and validation metadata do not define its donor groups. It uses task
`description.csv` files and the fourteen archived
`sample_metrics.csv` files only. It reads material identity, chemical formula,
and description for grouping, never a property target or an outcome metric.
Descriptions are matched by exact `material_id`/`id`, with explicitly recorded
`source_cif_id` aliases if present. An existing `__record_<integer>` suffix may
be resolved to its exact structure alias and is recorded. Any missing,
conflicting, duplicate test identity, invalid formula, unexpected task/seed,
or changed cross-seed test order stops the preflight.

Some source spreadsheet metadata contain malformed date-like formulas. When
strict parsing fails, the same structure ID's original CIF must supply exactly
one `_chemical_formula_sum` tag. This label-free recovery is explicitly recorded
with the original metadata, recovered formula, CIF path, and SHA256; a missing
or ambiguous CIF fails. Original CSV files and all model inputs stay unchanged.

The formula parser supports nested parentheses/brackets and positive integer,
decimal, or explicit rational counts, using `Decimal` and `Fraction` without
floating-point rounding. Unknown elements, charges, hydrates, isotope notation,
and unsupported syntax fail explicitly. Reduced composition keys are sorted
element/count pairs, e.g. `Fe0.5Mg0.5O` becomes `Fe:1|Mg:1|O:2`.

Outputs:

- `sample_groups.csv`: one table per task in original test order; includes
  `task,sample_index,sample_id,reduced_composition,element_set,description_sha256`
  plus formula, match evidence, and group sizes. It contains each unique task
  sample once; seed-specific equality is recorded separately in the JSON.
- `group_coverage.csv`: per-task primary/secondary coverage, eligible group
  sizes, same-description donor-pair rate, samples with no differently described
  eligible peer, and the capped combinatorial mapping-space size.
- `preflight.json`: status, all source-file hashes, fixed run list,
  schema, original sample-order hashes, and output hashes. Failure is explicit.

Identical description hashes are retained and quantified. They can make a
nominal identity permutation leave its semantic embedding unchanged. They are
not dropped or regrouped in response to experimental outcomes.

## Frozen inference and comparisons

The inference implementation must first replay the correct pairing against
archived formal predictions, preserving the previous documented numerical
tolerance and recording discrepancies in both standardized and physical units.
Keep the original projected residual implementation and `tau=0.10` unchanged.
For every draw, save complete donor indices and verify group membership,
one-to-one use, no eligible fixed points, text embedding hashes, and unchanged
targets. Record the count of distinct donor maps rather than claiming that
twenty draws necessarily yield twenty unique permutations.

For each grouping, report results on its **same eligible sample subset**:

1. Correct crystal–text pairing.
2. The same checkpoint with semantic residual OFF.
3. Twenty conditional donor maps.
4. The twenty previously computed global permutations, sliced to this exact
   eligible subset. This contextual comparison uses the original whole-test
   donor pool; it does not imply a subset-only permutation null and needs no
   new global inference.

Report MAE, paired sample error changes with bootstrap intervals, coverage,
unique-map count, and description/embedding equality rates. Do not use the
conditional results to choose grouping, tasks, seeds, checkpoint, or model.
The primary and secondary outcomes must both be reported, including null or
negative outcomes. A tiny exact-composition subset, or a mapping space of one,
is descriptive evidence with limited support: twenty repeated draws do not
constitute twenty independent controls or training replicates.

## Scientific interpretation

A correct-pairing advantage under exact reduced-composition matching supports
correspondence-sensitive information beyond stoichiometry in the evaluated
checkpoint and covered subset. Its disappearance would suggest that part of
the global-permutation effect depends on composition mismatch. The element-set
control offers a broader but chemically weaker secondary comparison.

These interventions do not establish an independent-model performance gain,
out-of-distribution robustness, causal training mechanisms, or the source of
CartNet's negative MP-bulk result. Group coverage differs by task, and eligible
crystals may differ systematically from singleton crystals. The already
observed fixed-test outcomes make this a post-hoc exploratory analysis.

The metadata and donor maps require little disk space. Reuse existing graph
caches and checkpoints; do not create redundant datasets or remove archived
predictions, metrics, best checkpoints, or failure evidence.
