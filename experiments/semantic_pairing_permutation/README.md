# Semantic pairing permutation control

This exploratory, post-hoc experiment tests whether a frozen GPSR checkpoint
depends on the correct crystal-to-text pairing. It compares:

1. correctly paired frozen MatBERT embeddings;
2. semantic residual OFF in the same checkpoint; and
3. repeated, fixed-point-free permutations of test embeddings.

The permutations preserve the empirical embedding distribution and change only
the sample correspondence. They are never used for training, checkpoint
selection, task selection, or hyperparameter tuning.
