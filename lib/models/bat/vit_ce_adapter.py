from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple


from lib.models.layers.patch_embed import PatchEmbed


from .utils import combine_tokens, recover_tokens, resolve_encoder_layer_indices


from .vit import VisionTransformer


from ..layers.attn_adapt_blocks import CEABlock
from ..layers.moe_bridge import MoEBridge
from ..layers.fusion import TemplateResponseFusion


class VisionTransformerCE(VisionTransformer):

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            representation_size=None,
            distilled=False,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            weight_init='',
            ce_loc=None,
            ce_keep_ratio=None,
            search_size=None,
            template_size=None,
            new_patch_size=None,
            adapter_type=None,
            moe_enable=False,
            moe_loc=None,
            moe_mamba_d_state=16,
            moe_mamba_d_conv=4,
            moe_mamba_expand=2,
            moe_history_alpha=0.8,
            moe_router_hidden_ratio=0.25,
            moe_past_cache_gap=1,
    ):

        super().__init__()


        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)


        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W


        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W


        self.pos_embed_z = nn.Parameter(
            torch.zeros(1, self.num_patches_template, embed_dim)
        )
        self.pos_embed_x = nn.Parameter(
            torch.zeros(1, self.num_patches_search, embed_dim)
        )


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        self.moe_enable = bool(moe_enable)
        if self.moe_enable:

            raw_moe_loc = list(moe_loc) if moe_loc is not None else [-1]
            resolved_moe_loc = resolve_encoder_layer_indices(raw_moe_loc, depth)
            self.moe_layer = resolved_moe_loc[-1] if len(resolved_moe_loc) > 0 else depth - 1
            self.moe_loc = [self.moe_layer]
        else:
            self.moe_layer = None
            self.moe_loc = []
        self.moe_mamba_d_state = moe_mamba_d_state
        self.moe_mamba_d_conv = moe_mamba_d_conv
        self.moe_mamba_expand = moe_mamba_expand

        self.moe_bridge = None
        if self.moe_enable:
            self.moe_bridge = MoEBridge(
                dim=embed_dim,
                mamba_d_state=self.moe_mamba_d_state,
                mamba_d_conv=self.moe_mamba_d_conv,
                mamba_expand=self.moe_mamba_expand,
                history_alpha=moe_history_alpha,
                router_hidden_ratio=moe_router_hidden_ratio,
                past_cache_gap=moe_past_cache_gap,
            )

        for i in range(depth):
            ce_keep_ratio_i = 1.0

            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            blocks.append(
                CEABlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i,
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.moe_inner_dim = int(self.moe_bridge.inner_dim) if self.moe_bridge is not None else int(embed_dim)

        self.norm = norm_layer(embed_dim)
        self.template_response_fusion = TemplateResponseFusion(dim=embed_dim)

        self.init_weights(weight_init)
        self.template_response_fusion.reset_learnable_identity()


        if self.moe_bridge is not None:
            self.moe_bridge.zero_init_delta_projections()
            self.moe_bridge.reset_router_heads()

    def reset_moe_history(self):
        if self.moe_bridge is not None and hasattr(self.moe_bridge, "reset_history"):
            self.moe_bridge.reset_history()

    def set_moe_dynamic_history(self, enabled: bool = True):
        if self.moe_bridge is not None and hasattr(self.moe_bridge, "set_dynamic_history"):
            self.moe_bridge.set_dynamic_history(enabled)

    @staticmethod
    def _empty_moe_meta():
        return {
            "aux_rgb": None,
            "aux_tir": None,
            "router_quality_logits": None,
            "ran_moe": False,
        }

    def _apply_moe_bridge(
        self,
        x,
        xi,
        global_index_template,
        x_base_out,
        xi_base_out,
        return_expert_hist_ctx: bool,
        past_search_rgb=None,
        past_search_tir=None,
    ):
        moe_meta = self._empty_moe_meta()
        if self.moe_bridge is None:
            return x, xi, moe_meta

        template_len = global_index_template.shape[1]
        if template_len <= 0 or template_len >= x.shape[1]:
            raise ValueError(f"Invalid template length {template_len} for token length {x.shape[1]}")

        rgb_bundle = self.moe_bridge.build_stream_candidates(
            x,
            template_len=template_len,
            base_out=x_base_out,
            return_expert_hist_ctx=return_expert_hist_ctx,
            past_search_tokens=past_search_rgb,
            stream_id="rgb",
        )
        tir_bundle = self.moe_bridge.build_stream_candidates(
            xi,
            template_len=template_len,
            base_out=xi_base_out,
            return_expert_hist_ctx=return_expert_hist_ctx,
            past_search_tokens=past_search_tir,
            stream_id="tir",
        )
        routes = self.moe_bridge.route_candidate_pair(rgb_bundle, tir_bundle)
        x_out, aux_rgb = self.moe_bridge.apply_candidate_route(
            rgb_bundle,
            routes["rgb"],
            return_expert_hist_ctx=return_expert_hist_ctx,
        )
        xi_out, aux_tir = self.moe_bridge.apply_candidate_route(
            tir_bundle,
            routes["tir"],
            return_expert_hist_ctx=return_expert_hist_ctx,
        )

        moe_meta.update({
            "aux_rgb": aux_rgb,
            "aux_tir": aux_tir,
            "router_quality_logits": torch.stack(
                [routes["rgb"]["logits"], routes["tir"]["logits"]], dim=1
            ),
            "ran_moe": True,
        })
        return x_out, xi_out, moe_meta

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
        if self.moe_bridge is not None:
            new_prefix = prefix + "moe_bridge."
            for old_layer in self.moe_loc:
                old_prefix = prefix + f"blocks.{old_layer}.moe_bridge."
                for key in list(state_dict.keys()):
                    if key.startswith(old_prefix):
                        new_key = new_prefix + key[len(old_prefix):]
                        state_dict.setdefault(new_key, state_dict[key])
                        state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


    def forward_features(
        self,
        z,
        x,
        mask_z=None,
        mask_x=None,
        ce_template_mask=None,
        ce_keep_rate=None,
        return_last_attn=False,
        collect_layer_search_tokens=False,
        collect_layers=None,
        disable_moe=False,
        moe_past_search_tokens_rgb=None,
        moe_past_search_tokens_tir=None,
    ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        if self.training and self.moe_enable:
            self._moe_aux_hist_ctx_layers = []
        moe_router_quality_logits_layers = {}
        collect_layers_set = None
        search_tokens_rgb = {}
        search_tokens_tir = {}
        if collect_layer_search_tokens:
            if collect_layers is None:
                collect_layers = self.moe_loc
            collect_layers_set = set(
                resolve_encoder_layer_indices(collect_layers, len(self.blocks))
            )


        x_rgb = x[:, :3, :, :]
        z_rgb_0 = z[0][:, :3, :, :]
        z_rgb_1 = z[1][:, :3, :, :]

        x_dte = x[:, 3:, :, :]
        z_dte_0 = z[0][:, 3:, :, :]
        z_dte_1 = z[1][:, 3:, :, :]


        x, z0, z1 = x_rgb, z_rgb_0, z_rgb_1

        xi, zi0, zi1 = x_dte, z_dte_0, z_dte_1


        z0 = self.patch_embed(z0)
        z1 = self.patch_embed(z1)
        x = self.patch_embed(x)

        zi0 = self.patch_embed(zi0)
        zi1 = self.patch_embed(zi1)
        xi = self.patch_embed(xi)


        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(
                mask_z[None].float(),
                scale_factor=1. / self.patch_size
            ).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)
            mask_x = F.interpolate(
                mask_x[None].float(),
                scale_factor=1. / self.patch_size
            ).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)
            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)


        z0 += self.pos_embed_z
        z1 += self.pos_embed_z
        x += self.pos_embed_x

        zi0 += self.pos_embed_z
        zi1 += self.pos_embed_z
        xi += self.pos_embed_x


        z_list =  torch.cat([z0, z1], dim=1)
        x = combine_tokens(z_list, x, mode=self.cat_mode)

        zi_list =  torch.cat([zi0, zi1], dim=1)
        xi = combine_tokens(zi_list, xi, mode=self.cat_mode)


        x = self.pos_drop(x)
        xi = self.pos_drop(xi)


        lens_z = self.pos_embed_z.shape[1] * 2
        lens_x = self.pos_embed_x.shape[1]


        global_index_t = torch.linspace(
            0, lens_z - 1, lens_z,
            dtype=torch.int64
        ).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(
            0, lens_x - 1, lens_x,
            dtype=torch.int64
        ).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        global_index_ti = global_index_t.clone()
        global_index_si = global_index_s.clone()

        removed_indexes_s = []
        removed_indexes_si = []


        for i, blk in enumerate(self.blocks):
            past_search_rgb_i = None
            past_search_tir_i = None
            if isinstance(moe_past_search_tokens_rgb, dict):
                past_search_rgb_i = moe_past_search_tokens_rgb.get(i, None)
            if isinstance(moe_past_search_tokens_tir, dict):
                past_search_tir_i = moe_past_search_tokens_tir.get(i, None)
            x, global_index_t, global_index_s, removed_index_s, attn, \
            xi, global_index_ti, global_index_si, removed_index_si, attn_i = \
                blk(
                    x, xi,
                    global_index_t,
                    global_index_ti,
                    global_index_s,
                    global_index_si,
                    mask_x,
                    ce_template_mask,
                    ce_keep_rate,
                )
            x_base_out = x
            xi_base_out = xi
            moe_meta = self._empty_moe_meta()

            if collect_layers_set is not None and i in collect_layers_set:

                lens_z_cur = global_index_t.shape[1]
                lens_zi_cur = global_index_ti.shape[1]
                search_tokens_rgb[i] = x_base_out[:, lens_z_cur:, :]
                search_tokens_tir[i] = xi_base_out[:, lens_zi_cur:, :]

            if (not disable_moe) and self.moe_enable and i == self.moe_layer:
                x, xi, moe_meta = self._apply_moe_bridge(
                    x=x,
                    xi=xi,
                    global_index_template=global_index_t,
                    x_base_out=x_base_out,
                    xi_base_out=xi_base_out,
                    return_expert_hist_ctx=self.training,
                    past_search_rgb=past_search_rgb_i,
                    past_search_tir=past_search_tir_i,
                )

            if moe_meta["ran_moe"]:
                quality_logits = moe_meta.get("router_quality_logits", None)
                if torch.is_tensor(quality_logits):
                    moe_router_quality_logits_layers[i] = quality_logits
                if self.training:
                    self._moe_aux_hist_ctx_layers.append((i, moe_meta["aux_rgb"], moe_meta["aux_tir"]))


            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
                removed_indexes_si.append(removed_index_si)


        x = self.norm(x)
        xi = self.norm(xi)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        lens_xi_new = global_index_si.shape[1]
        lens_zi_new = global_index_ti.shape[1]


        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]
        zi = xi[:, :lens_zi_new]
        xi = xi[:, lens_zi_new:]


        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros(
                [B, pruned_lens_x, x.shape[2]],
                device=x.device
            )
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat(
                [global_index_s, removed_indexes_cat],
                dim=1
            )
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(
                dim=1,
                index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                src=x
            )

        if removed_indexes_si and removed_indexes_si[0] is not None:
            removed_indexes_cat_i = torch.cat(removed_indexes_si, dim=1)
            pruned_lens_xi = lens_x - lens_xi_new
            pad_xi = torch.zeros(
                [B, pruned_lens_xi, xi.shape[2]],
                device=xi.device
            )
            xi = torch.cat([xi, pad_xi], dim=1)
            index_all = torch.cat(
                [global_index_si, removed_indexes_cat_i],
                dim=1
            )
            C = xi.shape[-1]
            xi = torch.zeros_like(xi).scatter_(
                dim=1,
                index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                src=xi
            )


        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        xi = recover_tokens(xi, lens_zi_new, lens_x, mode=self.cat_mode)

        valid_search_rgb = torch.zeros(
            (B, lens_x), dtype=torch.bool, device=x.device
        ).scatter_(1, global_index_s.to(torch.int64), True)
        valid_search_tir = torch.zeros(
            (B, lens_x), dtype=torch.bool, device=xi.device
        ).scatter_(1, global_index_si.to(torch.int64), True)


        fused_search = self.template_response_fusion(
            template_rgb=z,
            template_tir=zi,
            search_rgb=x,
            search_tir=xi,
            valid_rgb=valid_search_rgb,
            valid_tir=valid_search_tir,
        )
        x = torch.cat([z + zi, fused_search], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
        }
        if len(moe_router_quality_logits_layers) > 0:
            aux_dict["moe_router_quality_logits"] = moe_router_quality_logits_layers
        if (
            self.training
            and self.moe_enable
            and len(getattr(self, "_moe_aux_hist_ctx_layers", [])) > 0
        ):
            aux_dict["moe_aux_hist_ctx"] = self._moe_aux_hist_ctx_layers
        if collect_layer_search_tokens:
            aux_dict["moe_layer_search_tokens_rgb"] = search_tokens_rgb
            aux_dict["moe_layer_search_tokens_tir"] = search_tokens_tir
        return x, aux_dict


    def forward(
        self,
        z,
        x,
        ce_template_mask=None,
        ce_keep_rate=None,
        tnc_keep_rate=None,
        return_last_attn=False,
        collect_layer_search_tokens=False,
        collect_layers=None,
        disable_moe=False,
        moe_past_search_tokens_rgb=None,
        moe_past_search_tokens_tir=None,
    ):
        x, aux_dict = self.forward_features(
            z,
            x,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            collect_layer_search_tokens=collect_layer_search_tokens,
            collect_layers=collect_layers,
            disable_moe=disable_moe,
            moe_past_search_tokens_rgb=moe_past_search_tokens_rgb,
            moe_past_search_tokens_tir=moe_past_search_tokens_tir,
        )

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    return model


def vit_base_patch16_224_ce_adapter(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
    model = _create_vision_transformer(
        pretrained=pretrained,
        **model_kwargs
    )

    return model


def vit_large_patch16_224_ce_adapter(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs
    )
    model = _create_vision_transformer(
        pretrained=pretrained,
        **model_kwargs
    )

    return model
