"""Graph-Preserving Semantic Residual Injection (GPSR).

The property-level residual model (`model/text_residual_shfmat.py`) corrects the
prediction after pooling:

    y_final = y_graph + alpha * delta_text

GPSR instead injects the crystal text once into the *atom* representation in the
middle of the message-passing stack:

    H0 -> GraphBlock_1..k -> Hmid -> GPSR(Hmid, t) -> Hmid' -> GraphBlock_k+1..L
       -> mean-pool -> regression head -> y

so the remaining graph blocks can still re-mix the semantic residual with the
local geometry.  The injected term is gated (optionally node-wise) and scaled by
a learnable channel-wise vector `alpha` that starts at 1e-3, i.e. the module is
close to an identity map at initialisation -- the "graph-preserving" part.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


def as_text_matrix(text_feat, num_graphs: int):
    """Normalise a collated graph-level text attribute to [B, text_dim].

    PyG concatenates a per-graph [768] attribute along dim 0, so a batch arrives
    either as [B, 768] (B > 1), as [768] (B == 1) or, for some collate paths, as
    a flat [B * 768].  All three are folded back to [B, text_dim] here.
    """
    if text_feat is None:
        return None
    if text_feat.dim() == 1:
        return text_feat.view(num_graphs, -1)
    if text_feat.dim() == 2 and text_feat.size(0) != num_graphs:
        if text_feat.numel() == num_graphs * text_feat.size(-1):
            return text_feat.reshape(num_graphs, -1)
    return text_feat


class TextProjector(nn.Module):
    """768 -> D -> D, as used by GPSR (no attention, embedding stays frozen)."""

    def __init__(self, text_dim: int = 768, node_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(text_dim, node_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(node_dim, node_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.net(text_feat)


class SampleReliabilityGate(nn.Module):
    """Crystal-level semantic gate from pre-injection graph and text features."""

    def __init__(self, node_dim: int, bias_init: float = 3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.net[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.net[2].bias, float(bias_init))

    def forward(self, graph_repr: torch.Tensor, text_repr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([graph_repr, text_repr], dim=-1))


class SemanticResidualInjector(nn.Module):
    """Gated atom-level semantic residual.

    node_out = node_feat + alpha * gate * text_proj[batch]

    - `alpha`  : channel-wise residual scale, shared by all samples, init 1e-3.
    - `gate`   : sigmoid(MLP([h_i, z_t])) per atom and per channel; with
                 use_node_gate=False the gate is a constant 1.0 (GPSR-Simple).
    """

    def __init__(
        self,
        text_dim: int = 768,
        node_dim: int = 128,
        use_node_gate: bool = True,
        alpha_init: float = 1e-3,
        gate_hidden_dim: int = None,
        dropout: float = 0.0,
        record_stats: bool = True,
        max_residual_ratio: float = float("inf"),
        use_sample_gate: bool = False,
        sample_gate_bias: float = 3.0,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.node_dim = node_dim
        self.use_node_gate = use_node_gate
        self.alpha_init = float(alpha_init)
        self.record_stats = record_stats
        self.max_residual_ratio = float(max_residual_ratio)
        self.use_sample_gate = bool(use_sample_gate)
        self.sample_gate_bias = float(sample_gate_bias)
        if self.max_residual_ratio <= 0:
            raise ValueError(f"max_residual_ratio must be positive, got {self.max_residual_ratio}")
        self._bound_checked = False

        self.text_projector = TextProjector(text_dim, node_dim, dropout)
        self.sample_gate = (
            SampleReliabilityGate(node_dim=node_dim, bias_init=self.sample_gate_bias)
            if self.use_sample_gate else None
        )
        if use_node_gate:
            hidden = gate_hidden_dim or node_dim
            self.gate = nn.Sequential(
                nn.Linear(2 * node_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, node_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None
        self.alpha = nn.Parameter(torch.full((node_dim,), self.alpha_init, dtype=torch.float32))
        self.last_stats = None
        self.last_gate = None
        self.last_delta = None
        self.last_node_feat = None
        self.last_batch_index = None
        self.last_sample_stats = None
        self.last_scale = None
        self.last_rho_raw = None
        self.last_rho_final = None
        # Differentiable tensors from the current forward pass. These are used
        # by the soft excess-residual loss; no tensor in this path is detached.
        self.current_rho = None
        self.current_h_norm = None
        self.current_delta_norm = None
        self.last_node_stats = None
        self.last_sample_gate = None
        # Evaluation-only hook used by the conditional shuffled-gate control.
        # Training leaves this as None.
        self.sample_r_override = None

    def forward(self, node_feat: torch.Tensor, text_feat: torch.Tensor, batch: torch.Tensor):
        """Returns (node_out, delta, gate, text_proj)."""
        if text_feat is None:
            raise ValueError("SemanticResidualInjector requires text_emb on the batch")
        if batch is None:
            batch = torch.zeros(node_feat.size(0), dtype=torch.long, device=node_feat.device)

        text_proj = self.text_projector(text_feat.to(node_feat.dtype))       # [B, D]
        text_node = text_proj[batch]                                          # [N, D]
        if text_node.shape != node_feat.shape:
            raise AssertionError(
                f"text/graph alignment failed: text_node={tuple(text_node.shape)} "
                f"node_feat={tuple(node_feat.shape)}"
            )
        num_graphs = int(text_proj.size(0))
        graph_repr = scatter(node_feat, batch, dim=0, dim_size=num_graphs, reduce="mean")
        if self.sample_gate is not None:
            sample_r = self.sample_gate(graph_repr, text_proj)
        else:
            sample_r = torch.ones((num_graphs, 1), dtype=node_feat.dtype, device=node_feat.device)
        if self.sample_r_override is not None:
            override = self.sample_r_override.to(device=node_feat.device, dtype=node_feat.dtype).reshape(-1, 1)
            if override.size(0) != num_graphs:
                raise ValueError(f"sample_r_override has {override.size(0)} values for {num_graphs} graphs")
            sample_r = override
        r_node = sample_r[batch]
        if self.use_node_gate:
            gate = self.gate(torch.cat([node_feat, text_node], dim=-1))       # [N, D]
            delta_raw = r_node * text_node * gate * self.alpha
        else:
            gate = None
            delta_raw = r_node * text_node * self.alpha

        # Sample-wise hard bound. Each crystal gets its own Frobenius ratio and
        # scale; no tensor in this path is detached, so gradients flow through
        # rho and scale when clipping is active.
        eps = 1e-12
        h_sq = scatter(node_feat.square().sum(dim=-1), batch, dim=0,
                       dim_size=num_graphs, reduce="sum")
        d_raw_sq = scatter(delta_raw.square().sum(dim=-1), batch, dim=0,
                           dim_size=num_graphs, reduce="sum")
        h_norm = h_sq.sqrt()
        d_raw_norm = d_raw_sq.sqrt()
        rho_raw = d_raw_norm / (h_norm + eps)
        self.current_rho = rho_raw
        self.current_h_norm = h_norm
        self.current_delta_norm = d_raw_norm
        if math.isfinite(self.max_residual_ratio):
            scale = torch.clamp(self.max_residual_ratio / (rho_raw + eps), max=1.0)
        else:
            scale = torch.ones_like(rho_raw)
        delta = delta_raw * scale[batch].unsqueeze(-1)
        d_final_sq = scatter(delta.square().sum(dim=-1), batch, dim=0,
                             dim_size=num_graphs, reduce="sum")
        d_final_norm = d_final_sq.sqrt()
        rho_final = d_final_norm / (h_norm + eps)

        if not self._bound_checked and math.isfinite(self.max_residual_ratio):
            max_final = float(rho_final.detach().max().item())
            if max_final > self.max_residual_ratio + 1e-4:
                raise AssertionError(
                    f"sample-wise GPSR bound failed: max rho_final={max_final:.8f} "
                    f"> tau={self.max_residual_ratio:.8f}"
                )
            self._bound_checked = True

        node_out = node_feat + delta
        self.last_scale = scale.detach()
        self.last_rho_raw = rho_raw.detach()
        self.last_rho_final = rho_final.detach()
        gate_flat = gate if gate is not None else torch.ones_like(delta_raw)
        gate_atom = gate_flat.mean(dim=-1)
        gate_atom_mean = scatter(gate_atom, batch, dim=0, dim_size=num_graphs, reduce="mean")
        gate_atom_var = scatter((gate_atom - gate_atom_mean[batch]).square(), batch,
                                dim=0, dim_size=num_graphs, reduce="mean")
        atom_gate_dispersion = gate_atom_var.sqrt()
        cos_delta_h = F.cosine_similarity(delta_raw, node_feat, dim=-1, eps=eps)
        cos_h_hprime = F.cosine_similarity(node_feat, node_out, dim=-1, eps=eps)
        text_proj_norm = text_proj.norm(dim=-1)
        graph_text_cosine = F.cosine_similarity(graph_repr, text_proj, dim=-1, eps=eps)
        self.last_sample_gate = sample_r.detach().squeeze(-1)
        self.last_sample_stats = {
            "rho": rho_raw.detach(),
            "rho_raw": rho_raw.detach(),
            "rho_final": rho_final.detach(),
            "scale": scale.detach(),
            "clipped": (rho_raw.detach() > self.max_residual_ratio).to(node_feat.dtype),
            "delta_norm": d_final_norm.detach(),
            "delta_raw_norm": d_raw_norm.detach(),
            "delta_final_norm": d_final_norm.detach(),
            "H_norm": h_norm.detach(),
            "text_proj_norm": text_proj_norm.detach(),
            "node_text_norm": text_proj_norm.detach(),
            "atom_gate_dispersion": atom_gate_dispersion.detach(),
            "sample_r": sample_r.detach().squeeze(-1),
            "graph_text_cosine": graph_text_cosine.detach(),
            "cos_h_off_on": (scatter((node_feat * node_out).sum(-1), batch, dim=0, dim_size=num_graphs, reduce="sum") /
                             (h_norm * scatter(node_out.square().sum(-1), batch, dim=0, dim_size=num_graphs, reduce="sum").sqrt() + eps)).detach(),
            "cos_delta_h": (scatter((delta * node_feat).sum(-1), batch, dim=0, dim_size=num_graphs, reduce="sum") /
                            (d_final_norm * h_norm + eps)).detach(),
        }
        self.last_node_stats = {
            "cos_delta_h": cos_delta_h.detach(),
            "cos_h_hprime": cos_h_hprime.detach(),
        }

        if self.record_stats and not self.training:
            with torch.no_grad():
                self.last_stats = {
                    "mean_abs_delta_raw": delta_raw.abs().mean().item(),
                    "mean_abs_delta_final": delta.abs().mean().item(),
                    "mean_abs_delta": delta.abs().mean().item(),
                    "mean_abs_H": node_feat.abs().mean().item(),
                    "mean_abs_node_feat": node_feat.abs().mean().item(),
                    "rho": rho_final.mean().item(),
                    "mean_rho_raw": rho_raw.mean().item(),
                    "median_rho_raw": rho_raw.median().item(),
                    "max_rho_raw": rho_raw.max().item(),
                    "mean_rho_final": rho_final.mean().item(),
                    "median_rho_final": rho_final.median().item(),
                    "max_rho_final": rho_final.max().item(),
                    "clipped_sample_ratio": (rho_raw > self.max_residual_ratio).float().mean().item(),
                    "mean_scale": scale.mean().item(),
                    "mean_gate": gate_flat.mean().item(),
                    "std_gate": gate_flat.std().item(),
                    "gate_p10": torch.quantile(gate_flat, 0.10).item(),
                    "gate_p50": torch.quantile(gate_flat, 0.50).item(),
                    "gate_p90": torch.quantile(gate_flat, 0.90).item(),
                    "atom_gate_dispersion": atom_gate_dispersion.mean().item(),
                    "alpha_mean": self.alpha.mean().item(),
                    "alpha_abs_mean": self.alpha.abs().mean().item(),
                    "alpha_max_abs": self.alpha.abs().max().item(),
                    "alpha_l2_norm": self.alpha.norm().item(),
                    "active_alpha_ratio": (self.alpha.abs() > 0.01).float().mean().item(),
                    "strong_alpha_ratio": (self.alpha.abs() > 0.05).float().mean().item(),
                    "mean_text_proj_norm": text_proj_norm.mean().item(),
                    "median_text_proj_norm": text_proj_norm.median().item(),
                    "mean_node_text_norm": text_proj_norm[batch].mean().item(),
                    "mean_delta_norm": d_raw_norm.mean().item(),
                    "mean_H_norm": h_norm.mean().item(),
                    "mean_cos_delta_h": cos_delta_h.mean().item(),
                    "median_cos_delta_h": cos_delta_h.median().item(),
                    "std_cos_delta_h": cos_delta_h.std().item(),
                    "ratio_cos_gt_05": (cos_delta_h > 0.5).float().mean().item(),
                    "ratio_cos_abs_lt_02": (cos_delta_h.abs() < 0.2).float().mean().item(),
                    "ratio_cos_lt_minus02": (cos_delta_h < -0.2).float().mean().item(),
                    "mean_cos_h_hprime": cos_h_hprime.mean().item(),
                    "p10_cos_h_hprime": torch.quantile(cos_h_hprime, 0.10).item(),
                    "mean_r": sample_r.mean().item(),
                    "std_r": sample_r.std(unbiased=False).item(),
                    "min_r": sample_r.min().item(),
                    "max_r": sample_r.max().item(),
                    "mean_graph_text_cosine": graph_text_cosine.mean().item(),
                    "num_nodes": int(node_feat.size(0)),
                    "num_graphs": num_graphs,
                }
                # kept for node-specificity and bounded-residual diagnostics.
                self.last_gate = None if gate is None else gate.detach()
                self.last_delta = delta.detach()
                self.last_node_feat = node_feat.detach()
                self.last_batch_index = batch.detach()
        else:
            self.last_stats = None
        return node_out, delta, gate, text_proj

    def alpha_repr(self) -> str:
        with torch.no_grad():
            a = self.alpha.detach()
            return (
                f"alpha_mean={a.mean().item():.6e}; "
                f"alpha_absmean={a.abs().mean().item():.6e}; "
                f"alpha_maxabs={a.abs().max().item():.6e}; "
                f"alpha_min={a.min().item():.6e}"
            )


def resolve_injection_block(position, num_layers: int) -> int:
    """Map a coarse depth request to a block index in 1..num_layers-1.

    The injector is called after `k` SFTConv blocks, so k must leave at least one
    block before pooling (never a post-pooling fusion).
    """
    if num_layers < 2:
        raise ValueError(f"GPSR needs at least 2 graph blocks, got {num_layers}")
    if isinstance(position, str):
        frac = {"early": 0.25, "middle": 0.5, "late": 0.75}[position.lower()]
        k = int(frac * num_layers + 0.5)
    else:
        k = int(position)
    return max(1, min(num_layers - 1, k))


class GPSRSFTGNN(nn.Module):
    """SFTGNN with one SemanticResidualInjector inserted after block `k`."""

    def __init__(
        self,
        sftgnn: nn.Module,
        node_dim: int = 128,
        text_dim: int = 768,
        use_node_gate: bool = True,
        alpha_init: float = 1e-3,
        injection_after_block: int = 3,
        dropout: float = 0.0,
        record_stats: bool = True,
        max_residual_ratio: float = float("inf"),
        use_sample_gate: bool = False,
        sample_gate_bias: float = 3.0,
        name: str = "GPSR",
    ):
        super().__init__()
        self.sftgnn = sftgnn
        self.name = name
        self.node_dim = node_dim
        self.injection_after_block = int(injection_after_block)
        self.injector = SemanticResidualInjector(
            text_dim=text_dim,
            node_dim=node_dim,
            use_node_gate=use_node_gate,
            alpha_init=alpha_init,
            dropout=dropout,
            record_stats=record_stats,
            max_residual_ratio=max_residual_ratio,
            use_sample_gate=use_sample_gate,
            sample_gate_bias=sample_gate_bias,
        )

    def forward(self, data) -> dict:
        text_feat = as_text_matrix(getattr(data, "text_emb", None), data.num_graphs)
        node_features, inject_info = self.sftgnn.encode_nodes(
            data,
            injector=self.injector,
            injection_after_block=self.injection_after_block,
            text_feat=text_feat,
        )
        # pooling happens after the full (semantically corrected) block stack
        h_crystal = scatter(node_features, data.batch, dim=0, reduce="mean")
        pred = self.sftgnn.fc(h_crystal).squeeze(-1)
        out = {
            "property_pred": pred,
            "pred": pred,
            "embeddings": {
                "crystal": h_crystal,
                "text": inject_info["text_proj"] if inject_info else None,
            },
        }
        # current_rho remains attached to the graph for the soft penalty.
        out["gpsr_rho"] = self.injector.current_rho
        if self.injector.last_stats is not None:
            out["gpsr_stats"] = self.injector.last_stats
        return out
