import torch
import torch.nn.functional as F


def modality_router_balance_loss(logits):
    if logits.ndim != 3 or logits.shape[1:] != (2, 2) or logits.shape[0] == 0:
        raise ValueError(
            "Independent router logits must have shape [B, 2 modalities, 2 experts], "
            f"got {tuple(logits.shape)}."
        )
    probabilities = F.softmax(logits, dim=-1)
    hard_usage = F.one_hot(
        probabilities.detach().argmax(dim=-1), num_classes=2
    ).to(probabilities.dtype).mean(dim=0)
    mean_probability = probabilities.mean(dim=0)
    return (2.0 * (hard_usage * mean_probability).sum(dim=-1)).mean()
