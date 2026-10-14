"""Frozen GPSR method adapted to the official ALIGNN backbone."""
from __future__ import annotations

from typing import Dict, Tuple

import dgl
import torch
from torch import nn

from alignn.models.alignn import ALIGNN, ALIGNNConfig


class ALIGNNGPSR(ALIGNN):
    """ALIGNN with semantic residual injection after ALIGNN block 1."""

    def __init__(self, mode: str, tau: float = 0.10):
        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=4,
            gcn_layers=4,
            atom_input_features=92,
            edge_input_features=80,
            triplet_input_features=40,
            embedding_features=64,
            hidden_features=256,
            output_features=1,
        )
        super().__init__(config)
        if mode not in {"early", "hard"}:
            raise ValueError(f"unsupported semantic mode: {mode}")
        self.mode = mode
        self.tau = float(tau)
        self.semantic_enabled = True
        self.text_projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.node_gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.full((256,), 1e-3))

    @staticmethod
    def _batch_index(g: dgl.DGLGraph, device: torch.device) -> torch.Tensor:
        counts = g.batch_num_nodes().to(device=device, dtype=torch.long)
        return torch.repeat_interleave(
            torch.arange(counts.numel(), device=device), counts
        )

    @staticmethod
    def _sample_sum(values: torch.Tensor, batch_index: torch.Tensor, n: int) -> torch.Tensor:
        out = torch.zeros(n, device=values.device, dtype=values.dtype)
        out.index_add_(0, batch_index, values)
        return out

    def apply_semantic(
        self,
        h: torch.Tensor,
        text: torch.Tensor,
        batch_index: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n_graphs = int(text.shape[0])
        if not self.semantic_enabled:
            delta = torch.zeros_like(h)
            return h, {"delta_final": delta}

        z_t = self.text_projector(text)
        text_node = z_t[batch_index]
        gate = self.node_gate(torch.cat([h, text_node], dim=-1))
        delta_raw = self.alpha * gate * text_node

        h_sq = self._sample_sum(h.square().sum(dim=-1), batch_index, n_graphs)
        raw_sq = self._sample_sum(
            delta_raw.square().sum(dim=-1), batch_index, n_graphs
        )
        h_norm = torch.sqrt(h_sq + 1e-24)
        raw_norm = torch.sqrt(raw_sq + 1e-24)
        rho_raw = raw_norm / (h_norm + 1e-12)

        if self.mode == "hard":
            scale = torch.minimum(
                torch.ones_like(rho_raw),
                torch.full_like(rho_raw, self.tau) / (rho_raw + 1e-12),
            )
        else:
            scale = torch.ones_like(rho_raw)
        delta_final = delta_raw * scale[batch_index, None]
        h_prime = h + delta_final

        if not collect_diagnostics:
            return h_prime, {}

        final_sq = self._sample_sum(
            delta_final.square().sum(dim=-1), batch_index, n_graphs
        )
        final_norm = torch.sqrt(final_sq + 1e-24)
        rho_final = final_norm / (h_norm + 1e-12)
        dot_delta_h = self._sample_sum(
            (delta_final * h).sum(dim=-1), batch_index, n_graphs
        )
        hp_sq = self._sample_sum(
            h_prime.square().sum(dim=-1), batch_index, n_graphs
        )
        hp_norm = torch.sqrt(hp_sq + 1e-24)
        dot_h_hp = self._sample_sum(
            (h * h_prime).sum(dim=-1), batch_index, n_graphs
        )

        gate_atom = gate.mean(dim=-1)
        dispersion = []
        for sample_idx in range(n_graphs):
            vals = gate_atom[batch_index == sample_idx]
            dispersion.append(vals.std(unbiased=False) if vals.numel() else vals.new_zeros(()))

        diagnostics = {
            "rho_raw": rho_raw,
            "rho_final": rho_final,
            "scale": scale,
            "delta_raw_norm": raw_norm,
            "delta_final_norm": final_norm,
            "h_norm": h_norm,
            "cos_delta_h": dot_delta_h / (final_norm * h_norm + 1e-12),
            "cos_h_hprime": dot_h_hp / (h_norm * hp_norm + 1e-12),
            "gate_mean": gate.mean().reshape(1),
            "gate_std": gate.std(unbiased=False).reshape(1),
            "atom_gate_dispersion": torch.stack(dispersion),
            "text_proj_norm": torch.linalg.vector_norm(z_t, dim=-1),
            "delta_final": delta_final,
        }
        return h_prime, diagnostics

    def forward(self, inputs, text: torch.Tensor, collect_diagnostics: bool = False):
        g, lg, _lattice = inputs
        lg = lg.local_var()
        g = g.local_var()

        z = self.angle_embedding(lg.edata.pop("h"))
        x = self.atom_embedding(g.ndata.pop("atom_features"))
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        diagnostics: Dict[str, torch.Tensor] = {}
        batch_index = self._batch_index(g, x.device)
        for idx, alignn_layer in enumerate(self.alignn_layers):
            x, y, z = alignn_layer(g, lg, x, y, z)
            if idx == 0:
                x, diagnostics = self.apply_semantic(
                    x, text, batch_index, collect_diagnostics
                )

        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)
        pooled = self.readout(g, x)
        out = torch.squeeze(self.link(self.fc(pooled)))
        return out, diagnostics


def graph_recovery_check(model: ALIGNNGPSR, device: torch.device) -> float:
    """The semantic module must be the exact identity when disabled."""
    generator = torch.Generator(device=device).manual_seed(314159)
    h = torch.randn((17, 256), generator=generator, device=device)
    text = torch.randn((3, 768), generator=generator, device=device)
    batch_index = torch.tensor(
        [0] * 4 + [1] * 6 + [2] * 7, device=device, dtype=torch.long
    )
    prior = model.semantic_enabled
    model.semantic_enabled = False
    recovered, _ = model.apply_semantic(h, text, batch_index)
    model.semantic_enabled = prior
    return float((recovered - h).abs().max().item())
