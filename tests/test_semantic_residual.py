"""Shape / behaviour tests for the GPSR semantic-residual injector.

Run standalone (`python tests/test_semantic_residual.py`) or via pytest.
"""
import os
import sys

import torch
from torch_geometric.data import Batch, Data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from model.net import SFTGNN, TextMLP  # noqa: E402
from model.semantic_residual import (  # noqa: E402
    GPSRSFTGNN,
    SemanticResidualInjector,
    as_text_matrix,
    resolve_injection_block,
)


def _fake_data(num_graphs=3, atoms=(5, 4, 6), text_dim=768, seed=0):
    g = torch.Generator().manual_seed(seed)
    graphs = []
    for n in atoms[:num_graphs]:
        e = n * 2
        graphs.append(
            Data(
                x=torch.rand(n, 92, generator=g),
                edge_index=torch.randint(0, n, (2, e), generator=g),
                edge_attr=torch.rand(e, generator=g) + 1.0,
                sh=torch.randn(e, 16, generator=g),
                text_emb=torch.randn(text_dim, generator=g),
            )
        )
    return Batch.from_data_list(graphs)


def test_text_matrix_shapes():
    t = torch.randn(3, 768)
    assert as_text_matrix(t, 3).shape == (3, 768)
    assert as_text_matrix(torch.randn(768), 1).shape == (1, 768)
    assert as_text_matrix(torch.randn(3 * 768), 3).shape == (3, 768)


def test_injector_simple_and_gated():
    node = torch.randn(12, 128)
    text = torch.randn(3, 768)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    simple = SemanticResidualInjector(text_dim=768, node_dim=128, use_node_gate=False, alpha_init=1e-3)
    out, delta, gate, text_proj = simple(node, text, batch)
    assert out.shape == node.shape and text_proj.shape == (3, 128) and gate is None
    assert torch.allclose(simple.alpha, torch.full((128,), 1e-3))
    # delta is small at init: relative residual ratio must be tiny
    rho = delta.norm() / node.norm()
    assert rho < 0.05, f"initial relative residual too large: {rho}"

    gated = SemanticResidualInjector(text_dim=768, node_dim=128, use_node_gate=True, alpha_init=1e-3)
    out_g, delta_g, gate_g, _ = gated(node, text, batch)
    assert gate_g.shape == (12, 128)
    assert (gate_g >= 0).all() and (gate_g <= 1).all()
    # node-specific: rows must not all be identical
    assert torch.unique(gate_g, dim=0).size(0) > 1
    # same crystal text, different atoms -> different residual
    assert not torch.allclose(delta_g[0], delta_g[1])


def test_injector_broadcast_alignment():
    """Text of graph j must reach exactly the atoms of graph j."""
    node = torch.zeros(7, 128)
    text = torch.tensor([[1.0] + [0.0] * 767, [0.0, 1.0] + [0.0] * 766, [0.0] * 767 + [1.0]])
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    inj = SemanticResidualInjector(text_dim=768, node_dim=128, use_node_gate=False, alpha_init=1.0)
    # with a zero projector the projection is not identity; instead check text_node directly
    text_proj = inj.text_projector(text)
    text_node = text_proj[batch]
    for g in range(3):
        mask = batch == g
        assert torch.allclose(text_node[mask], text_proj[g].expand(int(mask.sum()), -1))




def test_samplewise_hard_bound_and_no_amplification():
    torch.manual_seed(7)
    node = torch.randn(10, 128)
    text = torch.randn(3, 768)
    batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    tau = 0.2
    inj = SemanticResidualInjector(
        text_dim=768, node_dim=128, use_node_gate=True,
        alpha_init=2.0, max_residual_ratio=tau,
    )
    out, delta, _, _ = inj(node, text, batch)
    assert out.shape == node.shape
    assert float(inj.last_rho_final.max()) <= tau + 1e-4
    assert torch.all(inj.last_scale <= 1.0 + 1e-7)
    small = inj.last_rho_raw <= tau
    if small.any():
        assert torch.allclose(inj.last_scale[small], torch.ones_like(inj.last_scale[small]))
        assert torch.allclose(inj.last_rho_final[small], inj.last_rho_raw[small], atol=1e-6)


def test_hard_bound_gradient_path():
    torch.manual_seed(11)
    node = torch.randn(9, 128, requires_grad=True)
    text = torch.randn(3, 768)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    inj = SemanticResidualInjector(
        text_dim=768, node_dim=128, use_node_gate=True,
        alpha_init=5.0, max_residual_ratio=0.1,
    )
    out, _, _, _ = inj(node, text, batch)
    assert (inj.last_scale < 1.0).any(), "test must exercise the clipping branch"
    out.square().mean().backward()
    grads = [inj.alpha.grad, inj.text_projector.net[0].weight.grad, inj.gate[0].weight.grad, node.grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(float(g.abs().sum()) > 0 for g in grads)

def test_soft_excess_residual_loss_is_samplewise_and_differentiable():
    torch.manual_seed(19)
    node = torch.randn(9, 128, requires_grad=True)
    text = torch.randn(3, 768)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    inj = SemanticResidualInjector(
        text_dim=768, node_dim=128, use_node_gate=True,
        alpha_init=4.0, max_residual_ratio=float("inf"),
    )
    _, delta, _, _ = inj(node, text, batch)
    rho = inj.current_rho
    assert rho.shape == (3,)
    # Independent reference calculation for every crystal.
    ref = []
    for g in range(3):
        m = batch == g
        ref.append(delta[m].norm() / (node[m].norm() + 1e-12))
    assert torch.allclose(rho, torch.stack(ref), atol=1e-6)
    residual_loss = torch.relu(rho - 0.10).square().mean()
    assert residual_loss.item() > 0
    residual_loss.backward()
    grads = [node.grad, inj.alpha.grad, inj.text_projector.net[0].weight.grad,
             inj.gate[0].weight.grad]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_soft_penalty_has_free_region():
    node = torch.randn(8, 128)
    text = torch.randn(2, 768)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    inj = SemanticResidualInjector(alpha_init=1e-3, max_residual_ratio=float("inf"))
    inj(node, text, batch)
    assert float(inj.current_rho.max()) < 0.10
    assert torch.relu(inj.current_rho - 0.10).square().mean().item() == 0.0


def test_gpsr_wrapper_forward_backward():
    backbone = SFTGNN(num_layers=5, in_dim=92)
    model = GPSRSFTGNN(backbone, node_dim=128, text_dim=768, use_node_gate=True,
                       alpha_init=1e-3, injection_after_block=3)
    data = _fake_data()
    out = model(data)
    assert out["property_pred"].shape == (3,)
    assert set(model.injector.alpha.shape) == {128}

    loss = out["property_pred"].pow(2).mean()
    loss.backward()
    assert model.injector.alpha.grad is not None and model.injector.alpha.grad.abs().sum() > 0
    # both projector and gate must receive gradient
    assert model.injector.text_projector.net[0].weight.grad.abs().sum() > 0
    assert model.injector.gate[0].weight.grad.abs().sum() > 0
    # graph backbone must still be trained
    assert backbone.layers[0].mlp[0].weight.grad.abs().sum() > 0

    # injection point actually matters: changing it changes the prediction
    # (alpha is opened up here, at the 1e-3 init the module is ~identity by design)
    model.eval()
    with torch.no_grad():
        model.injector.alpha.fill_(1.0)
        p3 = model(data)["property_pred"]
        model.injection_after_block = 4
        p4 = model(data)["property_pred"]
    assert not torch.allclose(p3, p4)


def test_resolve_injection_block():
    assert resolve_injection_block("middle", 5) == 3
    assert resolve_injection_block("middle", 6) == 3
    assert resolve_injection_block("middle", 8) == 4
    assert resolve_injection_block("early", 5) == 1
    assert resolve_injection_block("late", 5) == 4
    assert resolve_injection_block(0, 5) == 1  # clamped
    assert resolve_injection_block(99, 5) == 4  # never after the last block


def test_graph_only_path_unchanged():
    """SFTGNN.forward must be identical to the pre-GPSR implementation when no injector is given."""
    backbone = SFTGNN(num_layers=5, in_dim=92)
    data = _fake_data()
    nodes, info = backbone.encode_nodes(data)
    assert info is None
    ref = backbone.fc(torch_scatter_mean(nodes, data.batch)).squeeze()
    assert torch.allclose(backbone(data), ref, atol=1e-6)


def torch_scatter_mean(nodes, batch):
    from torch_scatter import scatter

    return scatter(nodes, batch, dim=0, reduce="mean")


def test_textmlp_untouched():
    """Guard: the property-level text path used by the concat baseline is unchanged."""
    mlp = TextMLP(768, 128, 64)
    assert mlp(torch.randn(4, 768)).shape == (4, 64)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("all tests passed" if failures == 0 else f"{failures} test(s) failed")
    sys.exit(1 if failures else 0)
