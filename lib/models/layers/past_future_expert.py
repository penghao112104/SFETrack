from __future__ import annotations

import torch
import torch.nn as nn


def _import_mamba():
    try:
        from mamba_ssm import Mamba

        return Mamba
    except ImportError as e:
        raise ImportError(
            "The future expert uses Mamba. Install with: pip install mamba-ssm  "
            "(CUDA toolkit required to build; see https://github.com/state-spaces/mamba)"
        ) from e


class BiMamba(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        Mamba = _import_mamba()
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fuse = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yf = self.fwd(x)
        x_rev = torch.flip(x, dims=[1])
        yb = self.bwd(x_rev)
        yb = torch.flip(yb, dims=[1])
        return self.fuse(torch.cat([yf, yb], dim=-1))


class HistoryCrossAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, query_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        if query_tokens.dim() != 3 or kv_tokens.dim() != 3:
            raise ValueError(
                "HistoryCrossAttention expects query_tokens and kv_tokens as (B,N,D)."
            )
        if query_tokens.shape[0] != kv_tokens.shape[0] or query_tokens.shape[2] != kv_tokens.shape[2]:
            raise ValueError(
                f"query_tokens shape {tuple(query_tokens.shape)} is incompatible "
                f"with kv_tokens shape {tuple(kv_tokens.shape)}."
            )
        q = self.q_norm(query_tokens)
        kv = self.kv_norm(kv_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        return query_tokens + attn_out


class HistoryUpdateSelfAttention(nn.Module):
    """Mix history tokens and decoded search features with one-head self-attention."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, history_token: torch.Tensor, decoded_search: torch.Tensor):
        if history_token.dim() != 3 or decoded_search.dim() != 3:
            raise ValueError("HistoryUpdateSelfAttention expects inputs as (B,N,D).")
        if (
            history_token.shape[0] != decoded_search.shape[0]
            or history_token.shape[2] != decoded_search.shape[2]
        ):
            raise ValueError(
                f"history_token shape {tuple(history_token.shape)} is incompatible "
                f"with decoded_search shape {tuple(decoded_search.shape)}."
            )

        history_len = history_token.shape[1]
        joint = torch.cat([history_token, decoded_search], dim=1)
        q = self.norm(joint)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        joint = joint + attn_out
        return joint[:, :history_len, :], joint[:, history_len:, :]


class SearchFeatureDecoder(nn.Module):

    def __init__(self, dim: int, hidden_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = max(dim, int(round(dim * float(hidden_ratio))))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, history_context: torch.Tensor) -> torch.Tensor:
        return history_context + self.mlp(self.norm(history_context))


class PastFutureMambaExpert(nn.Module):

    def __init__(
        self,
        mode: str,
        dim: int = 768,
        num_heads: int = 1,
        dropout: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if mode not in ("past", "future"):
            raise ValueError(f"mode must be 'past' or 'future', got {mode}")
        self.mode = mode
        self.cross_attn = HistoryCrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.decoder = SearchFeatureDecoder(dim=dim, dropout=dropout)
        self.history_update_attn = (
            HistoryUpdateSelfAttention(dim=dim, dropout=dropout)
            if self.mode == "past"
            else None
        )
        self.mixer = (
            BiMamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            if self.mode == "future"
            else None
        )
        self.mixer_norm = nn.LayerNorm(dim) if self.mode == "future" else None
        self.delta_proj = nn.Linear(dim, dim)
        self.last_history_update = None

    def forward(
        self,
        x: torch.Tensor,
        search_len: int,
        history_token: torch.Tensor,
        past_query_tokens: torch.Tensor = None,
        return_hist_ctx: bool = False,
    ):
        self.last_history_update = None
        h = x
        search_len = int(search_len)
        if search_len <= 0 or search_len >= h.shape[1]:
            raise ValueError(f"Invalid search_len={search_len} for token length {h.shape[1]}.")

        template_len = h.shape[1] - search_len
        current_search = h[:, template_len:, :]
        if (
            history_token.dim() != 3
            or history_token.shape[0] != current_search.shape[0]
            or history_token.shape[2] != current_search.shape[2]
        ):
            raise ValueError(
                f"history_token shape {tuple(history_token.shape)} must have the same batch and "
                f"channel dimensions as current search {tuple(current_search.shape)}."
            )

        updated_decoded_search = None
        if self.mode == "past":
            query_tokens = past_query_tokens if torch.is_tensor(past_query_tokens) else current_search
            if query_tokens.shape != current_search.shape:
                raise ValueError(
                    f"past_query_tokens shape {tuple(query_tokens.shape)} must match current search "
                    f"shape {tuple(current_search.shape)}."
                )
            history_context = self.cross_attn(query_tokens, history_token)
            decoded_search = self.decoder(history_context)
            history_update, updated_decoded_search = self.history_update_attn(history_token, decoded_search)
            temporal_search = decoded_search
        else:
            history_context = self.cross_attn(current_search, history_token)
            decoded_search = self.decoder(history_context)
            history_len = history_token.shape[1]
            mamba_in = torch.cat([history_token, current_search, decoded_search], dim=1)
            mixed = mamba_in + self.mixer(self.mixer_norm(mamba_in))
            history_update = mixed[:, :history_len, :]
            temporal_search = mixed[:, history_len:history_len + search_len, :]

        if self.mode == "past":
            search_delta = self.delta_proj(updated_decoded_search)
        else:
            search_delta = self.delta_proj(temporal_search)
        feature_delta = torch.zeros_like(h)
        feature_delta[:, template_len:, :] = search_delta
        self.last_history_update = history_update

        if return_hist_ctx:
            aux_search = decoded_search if self.mode == "future" else temporal_search
            aux = {
                "ctx": aux_search,
                "decoded_search": aux_search,
                "history_update": self.last_history_update,
            }
            if self.mode == "future":
                aux["mamba_current_search"] = temporal_search
            if updated_decoded_search is not None:
                aux["self_attn_decoded_search"] = updated_decoded_search
            return feature_delta, aux
        return feature_delta
