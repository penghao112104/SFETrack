import torch
from torch import nn
import torch.nn.functional as F


class TemplateResponseFusion(nn.Module):

    def __init__(self, dim: int = 768, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.template_norm = nn.LayerNorm(self.dim)
        self.search_norm = nn.LayerNorm(self.dim)

        self.template_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.search_proj = nn.Linear(self.dim, self.dim, bias=False)


        self.common_fuse = nn.Linear(self.dim * 2, self.dim, bias=False)

    @torch.no_grad()
    def reset_learnable_identity(self):
        """Start from same-space matching and an equal RGB/TIR common template."""
        nn.init.eye_(self.template_proj.weight)
        nn.init.eye_(self.search_proj.weight)
        nn.init.zeros_(self.common_fuse.weight)
        identity = torch.eye(
            self.dim,
            device=self.common_fuse.weight.device,
            dtype=self.common_fuse.weight.dtype,
        )
        self.common_fuse.weight[:, : self.dim].copy_(0.5 * identity)
        self.common_fuse.weight[:, self.dim :].copy_(0.5 * identity)

    @staticmethod
    def _prepare_mask(mask, search):
        if mask is None:
            return torch.ones(search.shape[:2], dtype=torch.bool, device=search.device)
        if mask.shape != search.shape[:2]:
            raise ValueError(
                f"Search validity mask must have shape {search.shape[:2]}, got {mask.shape}."
            )
        return mask.to(device=search.device, dtype=torch.bool)

    @staticmethod
    def _validate_inputs(template_rgb, template_tir, search_rgb, search_tir):
        if template_rgb.ndim != 3 or template_tir.ndim != 3:
            raise ValueError("Template tensors must have shape (B, Nt, C).")
        if search_rgb.ndim != 3 or search_tir.ndim != 3:
            raise ValueError("Search tensors must have shape (B, Ns, C).")
        if template_rgb.shape != template_tir.shape:
            raise ValueError(
                f"RGB/TIR template shapes must match, got {template_rgb.shape} and {template_tir.shape}."
            )
        if search_rgb.shape != search_tir.shape:
            raise ValueError(
                f"RGB/TIR search shapes must match, got {search_rgb.shape} and {search_tir.shape}."
            )
        if template_rgb.shape[0] != search_rgb.shape[0] or template_rgb.shape[-1] != search_rgb.shape[-1]:
            raise ValueError("Template and search batch/channel dimensions must match.")

    def _full_template_match(self, template, search, valid_mask):
        """Return the complete Nt-by-Ns template-search matching matrix."""
        query = self.template_proj(self.template_norm(template))
        key = self.search_proj(self.search_norm(search))
        query = F.normalize(query, p=2, dim=-1, eps=self.eps)
        key = F.normalize(key, p=2, dim=-1, eps=self.eps)
        match = torch.bmm(query, key.transpose(1, 2))
        return match.masked_fill(~valid_mask.unsqueeze(1), 0.0)

    @staticmethod
    def _center_match_over_template(match, valid_mask):
        """Remove the per-search-token offset along the template-token axis."""
        centered = match - match.mean(dim=1, keepdim=True)
        return centered * valid_mask.to(match.dtype).unsqueeze(1)

    def forward(
        self,
        template_rgb,
        template_tir,
        search_rgb,
        search_tir,
        valid_rgb=None,
        valid_tir=None,
    ):
        self._validate_inputs(template_rgb, template_tir, search_rgb, search_tir)
        if template_rgb.shape[-1] != self.dim:
            raise ValueError(
                f"Fusion was built for dim={self.dim}, got {template_rgb.shape[-1]}."
            )
        valid_rgb = self._prepare_mask(valid_rgb, search_rgb)
        valid_tir = self._prepare_mask(valid_tir, search_tir)

        norm_template_rgb = self.template_norm(template_rgb)
        norm_template_tir = self.template_norm(template_tir)


        template_common = self.common_fuse(
            torch.cat([norm_template_rgb, norm_template_tir], dim=-1)
        )


        match_rgb_private = self._full_template_match(
            template_rgb, search_rgb, valid_rgb
        )
        match_tir_private = self._full_template_match(
            template_tir, search_tir, valid_tir
        )
        match_rgb_common = self._full_template_match(
            template_common, search_rgb, valid_rgb
        )
        match_tir_common = self._full_template_match(
            template_common, search_tir, valid_tir
        )


        centered_match_rgb_private = self._center_match_over_template(
            match_rgb_private, valid_rgb
        )
        centered_match_rgb_common = self._center_match_over_template(
            match_rgb_common, valid_rgb
        )
        centered_match_tir_private = self._center_match_over_template(
            match_tir_private, valid_tir
        )
        centered_match_tir_common = self._center_match_over_template(
            match_tir_common, valid_tir
        )


        similarity_rgb = F.cosine_similarity(
            centered_match_rgb_private,
            centered_match_rgb_common,
            dim=1,
            eps=self.eps,
        )
        similarity_tir = F.cosine_similarity(
            centered_match_tir_private,
            centered_match_tir_common,
            dim=1,
            eps=self.eps,
        )
        similarities = torch.stack([similarity_rgb, similarity_tir], dim=-1)


        modality_valid = torch.stack([valid_rgb, valid_tir], dim=-1)
        min_value = torch.finfo(similarities.dtype).min
        modality_logits = similarities.masked_fill(~modality_valid, min_value)
        modality_weights = F.softmax(modality_logits, dim=-1)
        modality_weights = modality_weights * modality_valid.to(
            modality_weights.dtype
        )

        weight_rgb = modality_weights[..., 0].unsqueeze(-1)
        weight_tir = modality_weights[..., 1].unsqueeze(-1)
        valid_rgb_value = valid_rgb.to(search_rgb.dtype).unsqueeze(-1)
        valid_tir_value = valid_tir.to(search_tir.dtype).unsqueeze(-1)
        weighted_sum = (
            weight_rgb * valid_rgb_value * search_rgb
            + weight_tir * valid_tir_value * search_tir
        )
        available_weight = (
            weight_rgb * valid_rgb_value + weight_tir * valid_tir_value
        ).clamp_min(self.eps)
        return weighted_sum / available_weight
