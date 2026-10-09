# Method summary

GPSR adds one semantic residual update to an intermediate graph representation.
For node state `H` and frozen MatBERT embedding `t`, the SFTGNN implementation
projects `t` into the graph hidden dimension, applies a node-conditioned gate,
and scales channels with a learnable vector initialized to `1e-3`.

The bounded model computes a per-crystal Frobenius ratio

```text
rho = ||Delta H||_F / (||H||_F + epsilon)
```

and rescales the residual when necessary so that the final ratio is at most
`tau = 0.10`. The bound applies locally at the injection state; it is not a
bound on the final prediction or on downstream hidden states.

The paper configuration uses a frozen 768-dimensional MatBERT embedding,
injection after the first valid graph block, a node gate, channel-wise alpha,
and minimum-validation-MAE checkpoint selection.

Implementation entry points:

- `model/semantic_residual.py`: SFTGNN GPSR injector and model wrapper.
- `experiments/backbone_transfer_alignn/alignn_gpsr_model.py`: ALIGNN adapter.
- `experiments/cartnet_gpsr/cartnet_gpsr_model.py`: CartNet adapter.
- `experiments/final_confirmatory/semantic_intervention/smm_model.py`:
  semantic-message-modulation comparison.
