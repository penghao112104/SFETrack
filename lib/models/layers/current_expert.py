from __future__ import annotations

import torch
import torch.nn as nn


class OnlineTemplateRecalibrator(nn.Module):
    """Use the initial template to reweight the online template token by token."""

    def __init__(self, dim: int = 768):
        super().__init__()
        hidden = max(1, dim // 4)
        self.init_norm = nn.LayerNorm(dim)
        self.online_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_weight = None

    def forward(self, init_template: torch.Tensor, online_template: torch.Tensor) -> torch.Tensor:
        if init_template.shape != online_template.shape:
            raise ValueError(
                f"Template shapes must match for recalibration, got "
                f"{tuple(init_template.shape)} and {tuple(online_template.shape)}."
            )
        z0 = self.init_norm(init_template)
        zu = self.online_norm(online_template)
        relation = torch.cat([zu, z0, zu - z0, zu * z0], dim=-1)
        weight = torch.sigmoid(self.mlp(relation))
        self.last_weight = weight.detach()
        return online_template + self.gamma * online_template * (2.0 * weight - 1.0)


class CurrentSelfAttentionExpert(nn.Module):
    """Current expert: recalibrate the online template and model current search tokens."""

    def __init__(self, dim: int = 768, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.mode = "current"
        self.template_recalibrator = OnlineTemplateRecalibrator(dim=dim)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.delta_proj = nn.Linear(dim, dim)
        self.last_history_update = None

    def forward(
        self,
        x: torch.Tensor,
        search_len: int,
        history_token: torch.Tensor,
        return_hist_ctx: bool = False,
    ):
        self.last_history_update = None
        search_len = int(search_len)
        if search_len <= 0 or search_len >= x.shape[1]:
            raise ValueError(f"Invalid search_len={search_len} for token length {x.shape[1]}.")

        template_len = x.shape[1] - search_len
        current_search = x[:, template_len:, :]
        if (
            history_token.dim() != 3
            or history_token.shape[0] != current_search.shape[0]
            or history_token.shape[2] != current_search.shape[2]
        ):
            raise ValueError(
                f"history_token shape {tuple(history_token.shape)} must have the same batch and "
                f"channel dimensions as current search {tuple(current_search.shape)}."
            )

        template_tokens = x[:, :template_len, :]
        if template_len % 2 != 0:
            raise ValueError(f"Current expert expects two equal templates, got template_len={template_len}.")
        one_template_len = template_len // 2
        init_template = template_tokens[:, :one_template_len, :]
        online_template = template_tokens[:, one_template_len:, :]
        enhanced_online_template = self.template_recalibrator(init_template, online_template)
        enhanced_templates = torch.cat([init_template, enhanced_online_template], dim=1)

        history_len = history_token.shape[1]
        joint = torch.cat([history_token, enhanced_templates, current_search], dim=1)
        q = self.norm(joint)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        joint = joint + attn_out
        history_ctx = joint[:, :history_len, :]
        search_start = history_len + template_len
        search_ctx = joint[:, search_start:search_start + search_len, :]
        search_delta = self.delta_proj(search_ctx)

        feature_delta = torch.zeros_like(x)
        feature_delta[:, template_len:, :] = search_delta
        self.last_history_update = history_ctx

        if return_hist_ctx:
            return feature_delta, {
                "ctx": search_ctx,
                "decoded_search": None,
            }
        return feature_delta
