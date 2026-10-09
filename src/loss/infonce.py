import torch
import torch.nn.functional as F


def infonce_loss(text: torch.Tensor, graph: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if text.shape != graph.shape:
        raise ValueError(f"InfoNCE inputs must have equal shape, got {text.shape} and {graph.shape}")
    text = F.normalize(text, dim=-1)
    graph = F.normalize(graph, dim=-1)
    logits = text @ graph.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
