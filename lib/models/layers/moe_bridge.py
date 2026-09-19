import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.layers.current_expert import CurrentSelfAttentionExpert
from lib.models.layers.past_future_expert import PastFutureMambaExpert


class TemporalQualityRouter(nn.Module):
    """Choose a temporal expert from the current target-aware search tokens."""

    def __init__(self, dim, hidden_ratio=0.25, dropout=0.0):
        super().__init__()
        dim = int(dim)
        self.hidden_dim = max(64, int(round(dim * float(hidden_ratio))))

        self.feature_norm = nn.LayerNorm(dim)
        self.quality_head = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 2),
        )
        nn.init.zeros_(self.quality_head[-1].bias)

    def _summarize_modality(self, search_tokens):
        if search_tokens.ndim != 3:
            raise ValueError(
                "Temporal router expects current search tokens with shape [B, N, C]."
            )
        return self.feature_norm(search_tokens).mean(dim=1)

    def forward(self, search_tokens):
        return self.quality_head(self._summarize_modality(search_tokens))


class MoEBridge(nn.Module):
    """Route each modality independently through the shared temporal experts."""

    def __init__(
        self,
        dim=768,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        history_alpha=0.8,
        router_hidden_ratio=0.25,
        past_cache_gap=1,
    ):
        super().__init__()
        dropout = 0.0
        self.inner_dim = int(dim)
        self.history_alpha = min(max(float(history_alpha), 0.0), 1.0)
        self.past_cache_gap = max(1, int(past_cache_gap))
        self.num_experts = 3
        self.num_temporal_experts = 2

        self.router = nn.ModuleDict({
            modality: TemporalQualityRouter(
                dim=self.inner_dim,
                hidden_ratio=router_hidden_ratio,
                dropout=dropout,
            )
            for modality in ("rgb", "tir")
        })
        self.router_hidden_dim = self.router["rgb"].hidden_dim
        self.expert_past = PastFutureMambaExpert(
            mode="past",
            dim=self.inner_dim,
            num_heads=1,
            dropout=dropout,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.expert_future = PastFutureMambaExpert(
            mode="future",
            dim=self.inner_dim,
            num_heads=1,
            dropout=dropout,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.expert_current = CurrentSelfAttentionExpert(
            dim=self.inner_dim,
            num_heads=1,
            dropout=dropout,
        )


        self._history_tokens = {}
        self._past_search_tokens = {}
        self.dynamic_history = False
    def route_modalities(self, search_rgb: torch.Tensor, search_tir: torch.Tensor):
        raise RuntimeError(
            "route_modalities is obsolete: build both expert candidates first, "
            "then call route_candidate_pair."
        )

    def zero_init_delta_projections(self):
        """Start every expert residual branch as an exact zero update."""
        for expert in (self.expert_past, self.expert_current, self.expert_future):
            nn.init.zeros_(expert.delta_proj.weight)
            if expert.delta_proj.bias is not None:
                nn.init.zeros_(expert.delta_proj.bias)

    def reset_router_heads(self):
        """Keep the Past/Future output biases neutral at initialization."""
        for router in self.router.values():
            nn.init.zeros_(router.quality_head[-1].bias)

    @staticmethod
    def _fit_token_len(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        target_len = int(target_len)
        if tokens.shape[1] == target_len:
            return tokens
        if tokens.shape[1] <= 0:
            raise ValueError("Cannot build history tokens from an empty template sequence.")
        repeat = (target_len + tokens.shape[1] - 1) // tokens.shape[1]
        return tokens.repeat(1, repeat, 1)[:, :target_len, :]

    def reset_history(self, stream_id=None):
        if stream_id is None:
            self._history_tokens.clear()
            self._past_search_tokens.clear()
        else:
            self._history_tokens.pop(str(stream_id), None)
            self._past_search_tokens.pop(str(stream_id), None)

    def set_dynamic_history(self, enabled: bool = True):
        self.dynamic_history = bool(enabled)

    def _build_template_history(self, h: torch.Tensor, template_len: int) -> torch.Tensor:


        history_len = template_len // 2 if template_len % 2 == 0 else template_len
        return h[:, :history_len, :]

    def _get_history_token(self, h: torch.Tensor, template_len: int, search_len: int, stream_id=None):
        template_history = self._build_template_history(h, template_len)
        if self.training or not self.dynamic_history:
            return template_history, template_history

        key = str(stream_id if stream_id is not None else "default")
        prev = self._history_tokens.get(key, None)
        if (
            not torch.is_tensor(prev)
            or prev.shape != template_history.shape
            or prev.device != template_history.device
        ):
            prev = template_history
        else:
            prev = prev.to(device=template_history.device, dtype=template_history.dtype)
        return prev, template_history

    def _get_past_search_token(self, current_search: torch.Tensor, stream_id=None):
        if self.training or not self.dynamic_history:
            return None
        key = str(stream_id if stream_id is not None else "default")
        history = self._past_search_tokens.get(key, None)
        if torch.is_tensor(history):
            history = [history]
        if not isinstance(history, list) or len(history) < self.past_cache_gap:
            return None
        prev = history[-self.past_cache_gap]
        if (
            not torch.is_tensor(prev)
            or prev.shape != current_search.shape
            or prev.device != current_search.device
        ):
            return None
        return prev.to(device=current_search.device, dtype=current_search.dtype)

    def _set_past_search_token(self, current_search: torch.Tensor, stream_id=None):
        if self.training or not self.dynamic_history:
            return
        key = str(stream_id if stream_id is not None else "default")
        history = self._past_search_tokens.get(key, None)
        if torch.is_tensor(history):
            history = [history]
        if not isinstance(history, list):
            history = []
        history.append(current_search.detach())
        while len(history) > self.past_cache_gap:
            history.pop(0)
        self._past_search_tokens[key] = history

    @staticmethod
    def _resize_token_len(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        target_len = int(target_len)
        if tokens.shape[1] == target_len:
            return tokens
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}.")
        return F.interpolate(
            tokens.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _update_history(self, history_base: torch.Tensor, gate: torch.Tensor, updates, stream_id=None):
        if self.training or not self.dynamic_history or len(updates) == 0:
            return
        valid_updates = []
        for update in updates:
            if not torch.is_tensor(update):
                return
            if update.shape[0] != history_base.shape[0] or update.shape[2] != history_base.shape[2]:
                return
            valid_updates.append(self._resize_token_len(update, history_base.shape[1]))
        stacked = torch.stack(valid_updates, dim=1)
        candidate = (gate.view(gate.shape[0], gate.shape[1], 1, 1) * stacked).sum(dim=1) / 2.0
        new_history = self.history_alpha * history_base + (1.0 - self.history_alpha) * candidate
        key = str(stream_id if stream_id is not None else "default")
        self._history_tokens[key] = new_history.detach()

    def build_stream_candidates(
        self,
        x,
        template_len: int,
        base_out=None,
        return_expert_hist_ctx: bool = False,
        past_search_tokens: torch.Tensor = None,
        stream_id: str = "rgb",
    ):
        """Run all experts once and return the two complete temporal candidates."""
        template_len = int(template_len)
        if base_out is None:
            base_out = x
        if base_out.shape != x.shape:
            raise ValueError("base_out and x must have the same shape.")
        if template_len <= 0 or template_len >= x.shape[1]:
            raise ValueError(f"Invalid template_len={template_len} for x_len={x.shape[1]}")

        search_len = x.shape[1] - template_len
        base_current_search = base_out[:, template_len:, :]
        if torch.is_tensor(past_search_tokens) and past_search_tokens.shape == base_current_search.shape:
            past_search_token = past_search_tokens.to(
                device=base_current_search.device,
                dtype=base_current_search.dtype,
            )
        else:
            past_search_token = self._get_past_search_token(
                base_current_search, stream_id=stream_id
            )
        past_available = torch.is_tensor(past_search_token)
        history_base, _ = self._get_history_token(
            base_out, template_len, search_len, stream_id=stream_id
        )

        pred_ctx_past = pred_ctx_future = None
        if return_expert_hist_ctx:
            out_past, pred_ctx_past = self.expert_past(
                base_out,
                search_len=search_len,
                history_token=history_base,
                past_query_tokens=past_search_token,
                return_hist_ctx=True,
            )
            past_history_update = getattr(self.expert_past, "last_history_update", None)
            out_future, pred_ctx_future = self.expert_future(
                base_out,
                search_len=search_len,
                history_token=history_base,
                return_hist_ctx=True,
            )
            future_history_update = getattr(self.expert_future, "last_history_update", None)
            out_current = self.expert_current(
                base_out,
                search_len=search_len,
                history_token=history_base,
            )
            current_history_update = getattr(self.expert_current, "last_history_update", None)
        else:
            out_past = self.expert_past(
                base_out,
                search_len=search_len,
                history_token=history_base,
                past_query_tokens=past_search_token,
            )
            past_history_update = getattr(self.expert_past, "last_history_update", None)
            out_future = self.expert_future(
                base_out,
                search_len=search_len,
                history_token=history_base,
            )
            future_history_update = getattr(self.expert_future, "last_history_update", None)
            out_current = self.expert_current(
                base_out,
                search_len=search_len,
                history_token=history_base,
            )
            current_history_update = getattr(self.expert_current, "last_history_update", None)

        def _search_delta(expert_out):
            if expert_out.shape != base_out.shape:
                raise ValueError(
                    f"expert output shape {tuple(expert_out.shape)} must match "
                    f"base_out shape {tuple(base_out.shape)}"
                )
            delta = torch.zeros_like(base_out)
            delta[:, template_len:, :] = expert_out[:, template_len:, :]
            return delta

        out_past = _search_delta(out_past)
        out_future = _search_delta(out_future)
        out_current = _search_delta(out_current)
        shared_current = base_out + out_current
        candidate_past = shared_current + out_past
        candidate_future = shared_current + out_future

        if isinstance(pred_ctx_past, dict) and torch.is_tensor(past_search_token):
            pred_ctx_past["query_tokens"] = past_search_token.detach()
        return {
            "base_out": base_out,
            "base_current_search": base_current_search,
            "template_len": template_len,
            "history_base": history_base,
            "history_updates": [
                current_history_update,
                past_history_update,
                future_history_update,
            ],
            "past_available": bool(past_available),
            "candidate_past": candidate_past,
            "candidate_future": candidate_future,
            "pred_ctx_past": pred_ctx_past,
            "pred_ctx_future": pred_ctx_future,
            "stream_id": stream_id,
        }

    def route_candidate_pair(self, rgb_bundle, tir_bundle):
        if rgb_bundle["stream_id"] != "rgb" or tir_bundle["stream_id"] != "tir":
            raise ValueError("Expected RGB and TIR candidate bundles in that order.")
        return {
            "rgb": self._route_stream(rgb_bundle),
            "tir": self._route_stream(tir_bundle),
        }

    def _route_stream(self, bundle):
        router = self.router[bundle["stream_id"]]
        current_search = bundle["base_current_search"]
        router_trainable = self.training and any(
            parameter.requires_grad for parameter in router.parameters()
        )
        if router_trainable:
            logits = router(current_search)
        else:
            with torch.no_grad():
                logits = router(current_search)
        probability = F.softmax(logits, dim=-1)

        hard = F.one_hot(probability.argmax(dim=-1), num_classes=2).to(probability.dtype)

        past_available = bool(bundle["past_available"])
        if not past_available:
            probability = torch.zeros_like(probability)
            probability[:, 1] = 1.0
            hard = probability
            gate = hard
        elif router_trainable:


            gate = hard + probability - probability.detach()
        else:

            gate = hard
        return {
            "logits": logits,
            "hard": hard,
            "gate": gate,
        }

    def apply_candidate_route(self, bundle, route, return_expert_hist_ctx: bool = False):
        """Apply this modality's hard route and update its temporal caches."""
        hard = route["hard"]
        gate = route.get("gate", hard)
        candidate_stack = torch.stack(
            [bundle["candidate_past"], bundle["candidate_future"]], dim=1
        )
        out = (gate.view(gate.shape[0], 2, 1, 1) * candidate_stack).sum(dim=1)

        current_vec = torch.ones_like(hard[:, 0])
        history_gate = torch.stack([current_vec, hard[:, 0], hard[:, 1]], dim=1)
        self._update_history(
            bundle["history_base"],
            history_gate,
            bundle["history_updates"],
            stream_id=bundle["stream_id"],
        )
        self._set_past_search_token(
            bundle["base_current_search"], stream_id=bundle["stream_id"]
        )

        aux = None
        if return_expert_hist_ctx:
            aux = {
                "past": bundle["pred_ctx_past"],
                "future": bundle["pred_ctx_future"],
                "target_search": bundle["base_current_search"].detach(),
            }
        return out, aux

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        old_to_new_prefixes = {
            prefix + "expert_a.": prefix + "expert_past.",
            prefix + "expert_b.": prefix + "expert_current.",
            prefix + "expert_c.": prefix + "expert_future.",
        }
        for old_prefix, new_prefix in old_to_new_prefixes.items():
            for key in list(state_dict.keys()):
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    state_dict.setdefault(new_key, state_dict[key])
                    state_dict.pop(key, None)


        old_norm_prefix = prefix + "norm."
        shared_router_norm_prefix = prefix + "router_norm."
        for key in list(state_dict.keys()):
            if key.startswith(old_norm_prefix):
                state_dict.pop(key, None)
            elif key.startswith(shared_router_norm_prefix):
                state_dict.pop(key, None)
            elif key.startswith(prefix + "router_norm_rgb.") or key.startswith(
                prefix + "router_norm_tir."
            ):
                state_dict.pop(key, None)

        old_future_norm_prefix = prefix + "expert_future.search_out_norm."
        mixer_norm_prefix = prefix + "expert_future.mixer_norm."
        for key in list(state_dict.keys()):
            if key.startswith(old_future_norm_prefix):
                new_key = mixer_norm_prefix + key[len(old_future_norm_prefix):]
                state_dict.setdefault(new_key, state_dict[key])
                state_dict.pop(key, None)

        removed_norm_prefixes = (
            prefix + "expert_current.out_norm.",
            prefix + "expert_past.history_update_attn.out_norm.",
            prefix + "expert_past.search_out_norm.",
        )
        for key in list(state_dict.keys()):
            if key.startswith(removed_norm_prefixes):
                state_dict.pop(key, None)


        state_dict.pop(prefix + "current_residual_scale", None)
        state_dict.pop(prefix + "temporal_residual_scale", None)


        router_prefix = prefix + "router."
        router_state = self.router.state_dict()
        legacy_router_found = False
        for key in list(state_dict.keys()):
            if not key.startswith(router_prefix):
                continue
            sub_key = key[len(router_prefix):]
            if sub_key.startswith(("feature_norm.", "quality_head.")):
                legacy_router_found = True
            cur_value = router_state.get(sub_key, None)
            old_value = state_dict[key]
            if cur_value is None or not torch.is_tensor(old_value) or cur_value.shape != old_value.shape:
                state_dict.pop(key, None)

        if legacy_router_found:
            warnings.warn(
                "This checkpoint uses a shared temporal router. Its router weights "
                "cannot initialize the independent RGB/TIR routers and were skipped. "
                "Train the new routers before evaluation; this is not an exact resume.",
                UserWarning,
                stacklevel=2,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x,
        template_len=None,
        base_out=None,
        return_expert_hist_ctx: bool = False,
        past_search_tokens: torch.Tensor = None,
        stream_id=None,
        external_temporal_probs: torch.Tensor = None,
        external_temporal_gate: torch.Tensor = None,
        router_stream: str = "rgb",
    ):
        raise RuntimeError(
            "MoEBridge now requires paired RGB/TIR candidate routing. "
            "Use build_stream_candidates, route_candidate_pair and "
            "apply_candidate_route through VisionTransformerCE."
        )
