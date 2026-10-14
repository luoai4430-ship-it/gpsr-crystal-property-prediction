"""Matched Semantic Message Modulation baseline for the confirmatory study.

This is a paper/code-derived SFTGNN adaptation of the official SAGE-CartNet
``TextConditionedMiddleModule``.  It is deliberately called SMM rather than
SAGE-Net because the released implementation is CartNet-specific.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from model.semantic_residual import TextProjector, as_text_matrix


class SemanticMessageModulator(nn.Module):
    def __init__(self, text_dim: int = 768, node_dim: int = 128,
                 hidden_dim: int = 128, correction_scale: float = 0.1,
                 gate_bias: float = -3.0):
        super().__init__()
        self.text_projector = TextProjector(text_dim, node_dim, dropout=0.0)
        self.condition_norm = nn.LayerNorm(2 * node_dim)
        self.conditioner = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.modulation = nn.Linear(hidden_dim, 3 * node_dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.text_scale = nn.Parameter(torch.tensor(1.0))
        self.correction_scale = float(correction_scale)
        self.gate_bias = float(gate_bias)
        self.semantic_enabled = True
        self.last_sample_stats = None
        self.last_node_stats = None
        self.last_stats = None
        self.last_gate = None
        self.last_node_feat = None
        self.last_delta = None
        self.last_batch_index = None
        self.current_rho = None

    def forward(self, x_before, x_after, text_feat, batch):
        if text_feat is None:
            raise ValueError("SMM requires text_emb")
        num_graphs = int(text_feat.size(0))
        text_proj = self.text_projector(text_feat.to(x_after.dtype))
        graph_context = scatter(x_before, batch, dim=0, dim_size=num_graphs, reduce="mean")
        hidden = self.conditioner(self.condition_norm(torch.cat([graph_context, text_proj], dim=-1)))
        gamma, beta, gate_logits = self.modulation(hidden).chunk(3, dim=-1)
        gamma = self.correction_scale * torch.tanh(gamma)
        beta = self.correction_scale * torch.tanh(beta)
        gate_graph = torch.sigmoid(gate_logits + self.gate_bias)
        block_delta = x_after - x_before
        correction = gate_graph[batch] * (gamma[batch] * block_delta + beta[batch])
        correction = self.text_scale * correction
        if not self.semantic_enabled:
            correction = torch.zeros_like(correction)
        node_out = x_after + correction

        eps = 1e-12
        h_sq = scatter(x_after.square().sum(-1), batch, dim=0, dim_size=num_graphs, reduce="sum")
        d_sq = scatter(correction.square().sum(-1), batch, dim=0, dim_size=num_graphs, reduce="sum")
        h_norm, d_norm = h_sq.sqrt(), d_sq.sqrt()
        rho = d_norm / (h_norm + eps)
        self.current_rho = rho
        dot_h_on = scatter((x_after * node_out).sum(-1), batch, dim=0,
                           dim_size=num_graphs, reduce="sum")
        on_norm = scatter(node_out.square().sum(-1), batch, dim=0,
                          dim_size=num_graphs, reduce="sum").sqrt()
        dot_d_h = scatter((correction * x_after).sum(-1), batch, dim=0,
                          dim_size=num_graphs, reduce="sum")
        cos_h_on = dot_h_on / (h_norm * on_norm + eps)
        cos_d_h = dot_d_h / (d_norm * h_norm + eps)
        node_cos_d_h = F.cosine_similarity(correction, x_after, dim=-1, eps=eps)
        node_cos_h_on = F.cosine_similarity(x_after, node_out, dim=-1, eps=eps)
        gate_node = gate_graph[batch]
        text_norm = text_proj.norm(dim=-1)

        self.last_sample_stats = {
            "rho": rho.detach(), "rho_raw": rho.detach(), "rho_final": rho.detach(),
            "scale": torch.ones_like(rho).detach(), "clipped": torch.zeros_like(rho).detach(),
            "delta_norm": d_norm.detach(), "delta_raw_norm": d_norm.detach(),
            "delta_final_norm": d_norm.detach(), "H_norm": h_norm.detach(),
            "text_proj_norm": text_norm.detach(), "node_text_norm": text_norm.detach(),
            "atom_gate_dispersion": torch.zeros_like(rho).detach(),
            "sample_r": torch.ones_like(rho).detach(),
            "graph_text_cosine": F.cosine_similarity(graph_context, text_proj, dim=-1, eps=eps).detach(),
            "cos_h_off_on": cos_h_on.detach(), "cos_delta_h": cos_d_h.detach(),
        }
        self.last_node_stats = {
            "cos_delta_h": node_cos_d_h.detach(),
            "cos_h_hprime": node_cos_h_on.detach(),
        }
        self.last_node_feat = x_after.detach()
        self.last_delta = correction.detach()
        self.last_batch_index = batch.detach()
        self.last_gate = gate_node.detach()
        self.last_stats = {
            "mean_abs_delta_raw": correction.abs().mean().item(),
            "mean_abs_delta_final": correction.abs().mean().item(),
            "mean_abs_delta": correction.abs().mean().item(),
            "mean_abs_H": x_after.abs().mean().item(),
            "mean_abs_node_feat": x_after.abs().mean().item(),
            "rho": rho.mean().item(),
            "mean_rho_raw": rho.mean().item(), "median_rho_raw": rho.median().item(),
            "max_rho_raw": rho.max().item(), "mean_rho_final": rho.mean().item(),
            "median_rho_final": rho.median().item(), "max_rho_final": rho.max().item(),
            "clipped_sample_ratio": 0.0, "mean_scale": 1.0,
            "mean_gate": gate_node.mean().item(), "std_gate": gate_node.std().item(),
            "gate_p10": torch.quantile(gate_node, .1).item(),
            "gate_p50": torch.quantile(gate_node, .5).item(),
            "gate_p90": torch.quantile(gate_node, .9).item(),
            "atom_gate_dispersion": 0.0,
            "alpha_mean": self.text_scale.item(),
            "alpha_abs_mean": self.text_scale.abs().item(),
            "alpha_max_abs": self.text_scale.abs().item(),
            "alpha_l2_norm": self.text_scale.abs().item(),
            "active_alpha_ratio": 1.0, "strong_alpha_ratio": 1.0,
            "mean_text_proj_norm": text_norm.mean().item(),
            "median_text_proj_norm": text_norm.median().item(),
            "mean_node_text_norm": text_norm[batch].mean().item(),
            "mean_delta_norm": d_norm.mean().item(), "mean_H_norm": h_norm.mean().item(),
            "mean_cos_delta_h": node_cos_d_h.mean().item(),
            "median_cos_delta_h": node_cos_d_h.median().item(),
            "std_cos_delta_h": node_cos_d_h.std().item(),
            "ratio_cos_gt_05": (node_cos_d_h > .5).float().mean().item(),
            "ratio_cos_abs_lt_02": (node_cos_d_h.abs() < .2).float().mean().item(),
            "ratio_cos_lt_minus02": (node_cos_d_h < -.2).float().mean().item(),
            "mean_cos_h_hprime": node_cos_h_on.mean().item(),
            "p10_cos_h_hprime": torch.quantile(node_cos_h_on, .1).item(),
            "mean_r": 1.0, "std_r": 0.0, "min_r": 1.0, "max_r": 1.0,
            "mean_graph_text_cosine": self.last_sample_stats["graph_text_cosine"].mean().item(),
            "num_nodes": int(x_after.size(0)), "num_graphs": num_graphs,
        }
        return node_out, correction, gate_node, text_proj

    def alpha_repr(self):
        return f"text_scale={self.text_scale.detach().item():.6e}"


class SMMSFTGNN(nn.Module):
    def __init__(self, sftgnn, node_dim=128, text_dim=768, injection_after_block=1,
                 name="SemanticMessageModulation"):
        super().__init__()
        self.sftgnn = sftgnn
        self.name = name
        self.injection_after_block = int(injection_after_block)
        self.injector = SemanticMessageModulator(text_dim=text_dim, node_dim=node_dim)

    def forward(self, data):
        text_feat = as_text_matrix(getattr(data, "text_emb", None), data.num_graphs)
        h = self.sftgnn.atom_embedding(data.x)
        index = data.edge_index
        sh = self.sftgnn.sh_mlp(data.sh)
        with torch.no_grad():
            dist = 1 / data.edge_attr.unsqueeze(-1)
        dist = self.sftgnn.rbf(dist)
        edge_attr = torch.cat([dist, sh], dim=-1)
        info = None
        for i, layer in enumerate(self.sftgnn.layers):
            before = h
            after = layer(h, index, edge_attr)
            if i + 1 == self.injection_after_block:
                h, delta, gate, text_proj = self.injector(before, after, text_feat, data.batch)
                info = {"delta": delta, "gate": gate, "text_proj": text_proj}
            else:
                h = after
        crystal = scatter(h, data.batch, dim=0, reduce="mean")
        pred = self.sftgnn.fc(crystal).squeeze(-1)
        out = {"property_pred": pred, "pred": pred,
               "embeddings": {"crystal": crystal, "text": info["text_proj"]}}
        out["gpsr_rho"] = self.injector.current_rho
        out["gpsr_stats"] = self.injector.last_stats
        return out
