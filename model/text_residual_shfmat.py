import math

import torch
import torch.nn as nn

from .net import SFTGNN


class TextProjector(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 128,
        graph_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, graph_dim),
            nn.LayerNorm(graph_dim),
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.projector(text_emb)


class TextResidualHead(nn.Module):
    def __init__(self, graph_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(graph_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        final = self.head[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def forward(self, h_graph: torch.Tensor, h_text: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                h_graph,
                h_text,
                torch.abs(h_graph - h_text),
                h_graph * h_text,
            ],
            dim=-1,
        )
        return self.head(features).squeeze(-1)


class TextResidualSHFMat(nn.Module):
    def __init__(
        self,
        sftgnn: SFTGNN,
        graph_dim: int = 128,
        text_in_dim: int = 768,
        text_hidden_dim: int = 128,
        residual_hidden_dim: int = 128,
        alpha_init: float = 0.03,
        alpha_max: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be in (0, alpha_max)")
        self.sftgnn = sftgnn
        self.graph_dim = graph_dim
        self.alpha_max = float(alpha_max)
        self.text_projector = TextProjector(
            in_dim=text_in_dim,
            hidden_dim=text_hidden_dim,
            graph_dim=graph_dim,
            dropout=dropout,
        )
        self.residual_head = TextResidualHead(
            graph_dim=graph_dim,
            hidden_dim=residual_hidden_dim,
            dropout=dropout,
        )
        alpha_ratio = alpha_init / alpha_max
        alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def _get_text_emb(self, data, batch_size: int, device, dtype) -> torch.Tensor:
        if hasattr(data, "text_emb") and data.text_emb is not None:
            text_emb = data.text_emb
            if text_emb.dim() == 1:
                text_emb = text_emb.view(batch_size, -1)
            return text_emb.to(device=device, dtype=dtype)
        return torch.zeros(batch_size, 768, device=device, dtype=dtype)

    def forward(self, data) -> dict:
        h_graph = self.sftgnn.forward_embedding(data)
        y_graph = self.sftgnn.fc(h_graph).squeeze(-1)
        batch_size = h_graph.size(0)

        text_emb = self._get_text_emb(data, batch_size, h_graph.device, h_graph.dtype)
        h_text = self.text_projector(text_emb)
        delta_text = self.residual_head(h_graph, h_text)
        alpha = self.alpha
        y_final = y_graph + alpha * delta_text

        embeddings = {"crystal": h_graph, "text": h_text}
        return {
            "pred": y_final,
            "property_pred": y_final,
            "pred_graph": y_graph,
            "delta_text": delta_text,
            "alpha": alpha,
            "h_graph": h_graph,
            "h_text": h_text,
            "embeddings": embeddings,
        }
