"""Frozen GPSR adapter for the official CartNet implementation."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from models.cartnet import CartNet


class CartNetGPSR(CartNet):
    """Official CartNet with GPSR injected after ``layers[0]`` only."""

    def __init__(self, tau: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        d = int(self.dim_in)
        self.tau = float(tau)
        self.semantic_enabled = True
        self.text_projector = nn.Sequential(
            nn.Linear(768, d), nn.GELU(), nn.Linear(d, d)
        )
        self.node_gate = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d), nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.full((d,), 1e-3))

    @staticmethod
    def _sample_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
        out = torch.zeros(n, device=values.device, dtype=values.dtype)
        out.index_add_(0, index, values)
        return out

    def apply_semantic(
        self,
        h: torch.Tensor,
        text: torch.Tensor,
        batch_index: torch.Tensor,
        collect_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.semantic_enabled:
            return h, {"delta_final": torch.zeros_like(h)}

        n = int(text.shape[0])
        z = self.text_projector(text)
        z_node = z[batch_index]
        gate = self.node_gate(torch.cat((h, z_node), dim=-1))
        delta_raw = self.alpha * gate * z_node

        h_norm = torch.sqrt(self._sample_sum(h.square().sum(-1), batch_index, n) + 1e-24)
        raw_norm = torch.sqrt(
            self._sample_sum(delta_raw.square().sum(-1), batch_index, n) + 1e-24
        )
        rho_raw = raw_norm / (h_norm + 1e-12)
        scale = torch.minimum(
            torch.ones_like(rho_raw),
            torch.full_like(rho_raw, self.tau) / (rho_raw + 1e-12),
        )
        delta = delta_raw * scale[batch_index, None]
        h_prime = h + delta
        if not collect_diagnostics:
            return h_prime, {}

        final_norm = torch.sqrt(
            self._sample_sum(delta.square().sum(-1), batch_index, n) + 1e-24
        )
        rho_final = final_norm / (h_norm + 1e-12)
        hp_norm = torch.sqrt(
            self._sample_sum(h_prime.square().sum(-1), batch_index, n) + 1e-24
        )
        dot_dh = self._sample_sum((delta * h).sum(-1), batch_index, n)
        dot_hhp = self._sample_sum((h * h_prime).sum(-1), batch_index, n)
        gate_atom = gate.mean(-1)
        dispersion = []
        for i in range(n):
            vals = gate_atom[batch_index == i]
            dispersion.append(vals.std(unbiased=False) if vals.numel() else vals.new_zeros(()))
        info = {
            "rho_raw": rho_raw,
            "rho_final": rho_final,
            "scale": scale,
            "clipped": (scale < 1.0).to(h.dtype),
            "delta_raw_norm": raw_norm,
            "delta_final_norm": final_norm,
            "h_norm": h_norm,
            "cos_delta_h": dot_dh / (final_norm * h_norm + 1e-12),
            "cos_h_hprime": dot_hhp / (h_norm * hp_norm + 1e-12),
            "gate_mean": gate.mean().reshape(1),
            "gate_std": gate.std(unbiased=False).reshape(1),
            "atom_gate_dispersion": torch.stack(dispersion),
            "text_proj_norm": torch.linalg.vector_norm(z, dim=-1),
            "alpha_mean_abs": self.alpha.abs().mean().reshape(1),
            "projector_weight_norm": torch.linalg.vector_norm(
                torch.cat([p.reshape(-1) for p in self.text_projector.parameters()])
            ).reshape(1),
            "delta_final": delta,
        }
        return h_prime, info

    def forward(self, batch, collect_diagnostics: bool = False):
        batch = self.encoder(batch)
        info: Dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.layers):
            batch = layer(batch)
            if idx == 0:
                batch.x, info = self.apply_semantic(
                    batch.x, batch.text_emb.float(), batch.batch,
                    collect_diagnostics=collect_diagnostics,
                )
        pred, true = self.head(batch)
        return pred, true, info

    def first_block_state(self, batch, semantic: bool):
        prior = self.semantic_enabled
        self.semantic_enabled = semantic
        batch = self.encoder(batch)
        batch = self.layers[0](batch)
        before = batch.x.clone()
        batch.x, _ = self.apply_semantic(batch.x, batch.text_emb.float(), batch.batch)
        after = batch.x.clone()
        self.semantic_enabled = prior
        return before, after

