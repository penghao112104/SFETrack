import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.bat.utils import resolve_moe_layer_indices


from lib.models.bat.vit_ce_adapter import vit_base_patch16_224_ce_adapter


def _resolve_moe_past_cache_gap(cfg):
    train_cfg = getattr(cfg, "TRAIN", None)
    if train_cfg is None:
        return 1
    gap = getattr(train_cfg, "MOE_AUX_PAST_GAP", 1)
    return max(1, int(gap))


class BATrack(nn.Module):
    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def reset_moe_history(self):
        if hasattr(self.backbone, "reset_moe_history"):
            self.backbone.reset_moe_history()

    def set_moe_dynamic_history(self, enabled: bool = True):
        if hasattr(self.backbone, "set_moe_dynamic_history"):
            self.backbone.set_moe_dynamic_history(enabled)

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        ce_template_mask=None,
        ce_keep_rate=None,
        return_last_attn=False,
        moe_past_search_tokens=None,
        disable_moe=False,
    ):

        x, aux_dict = self.backbone(
            z=template,
            x=search,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            disable_moe=disable_moe,
            moe_past_search_tokens_rgb=(
                moe_past_search_tokens.get("rgb", None)
                if isinstance(moe_past_search_tokens, dict)
                else None
            ),
            moe_past_search_tokens_tir=(
                moe_past_search_tokens.get("tir", None)
                if isinstance(moe_past_search_tokens, dict)
                else None
            ),
        )
        feat_last = x


        if isinstance(x, list):
            feat_last = x[-1]

        out = self.forward_head(feat_last, None)
        out.update(aux_dict)
        out['backbone_feat'] = x

        return out


    def forward_head(self, cat_feature, gt_score_map=None):

        enc_opt = cat_feature[:, -self.feat_len_s:]

        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()

        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)

            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            return {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map
            }

        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(
                opt_feat,
                gt_score_map
            )
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            return {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map
            }

        else:
            raise NotImplementedError


def build_batrack(cfg, training=True, load_pretrained=True):
    pretrained_init = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_adapter':
        moe_loc = resolve_moe_layer_indices(cfg.MODEL.MOE, depth=12)
        moe_past_cache_gap = _resolve_moe_past_cache_gap(cfg)
        backbone = vit_base_patch16_224_ce_adapter(
            pretrained_init,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
            template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
            new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
            adapter_type=cfg.TRAIN.PROMPT.TYPE,
            moe_enable=getattr(cfg.MODEL.MOE, "ENABLE", False),
            moe_loc=moe_loc,
            moe_mamba_d_state=getattr(cfg.MODEL.MOE, "MAMBA_D_STATE", 16),
            moe_mamba_d_conv=getattr(cfg.MODEL.MOE, "MAMBA_D_CONV", 4),
            moe_mamba_expand=getattr(cfg.MODEL.MOE, "MAMBA_EXPAND", 2),
            moe_history_alpha=getattr(cfg.MODEL.MOE, "ALPHA", 0.8),
            moe_router_hidden_ratio=getattr(cfg.MODEL.MOE, "ROUTER_HIDDEN_RATIO", 0.25),
            moe_past_cache_gap=moe_past_cache_gap,
        )
        hidden_dim = backbone.embed_dim

    else:
        raise NotImplementedError


    box_head = build_box_head(cfg, hidden_dim)


    model = BATrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE
    )

    if not training:
        return model


    if not load_pretrained:
        return model


    pretrained_path = cfg.MODEL.PRETRAIN_FILE
    if not pretrained_path or not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained backbone checkpoint not found: {pretrained_path}. "
            "Download the DropTrack/DropMAE pretrained weights and place them "
            "at the path configured by MODEL.PRETRAIN_FILE."
        )

    try:
        checkpoint = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            pretrained_path,
            map_location="cpu"
        )

    state_dict = checkpoint["net"] if "net" in checkpoint else checkpoint
    print(f"Loading pretrained weights: {pretrained_path}")

    new_dict = {}


    temp_pos_x = state_dict.get('backbone.temporal_pos_embed_x', None)
    temp_pos_z = state_dict.get('backbone.temporal_pos_embed_z', None)


    for k, v in state_dict.items():

        if k.startswith('module.'):
            k = k.replace('module.', '')


        if k == 'backbone.pos_embed':
            k = 'backbone.pos_embed_x'


        if 'pos_embed_x' in k:
            if temp_pos_x is not None:
                v_resized = resize_pos_embed(v, 16, 16)
                t_resized = resize_pos_embed(temp_pos_x, 16, 16)
                v = v_resized + t_resized
        elif 'pos_embed_z' in k:
            if temp_pos_z is not None:
                v_resized = resize_pos_embed(v, 8, 8)
                t_resized = resize_pos_embed(temp_pos_z, 8, 8)
                v = v_resized + t_resized


        if not k.startswith('backbone.') and 'box_head' not in k:
            k = 'backbone.' + k

        new_dict[k] = v


    msg = model.load_state_dict(new_dict, strict=False)
    if 'backbone.pos_embed_x' in msg.missing_keys:
        print("Warning: pos_embed_x was not loaded.")
    else:
        print("Pretrained weights loaded successfully.")

    return model


def resize_pos_embed(posemb, hight, width):
    posemb = posemb.float()
    posemb_grid = posemb[0]

    if posemb_grid.shape[0] in [197, 577, 65]:
        posemb_grid = posemb_grid[1:]

    gs_old = int(math.sqrt(len(posemb_grid)))
    if gs_old * gs_old != len(posemb_grid):
        return posemb

    posemb_grid = posemb_grid.reshape(
        1, gs_old, gs_old, -1
    ).permute(0, 3, 1, 2)

    posemb_grid = F.interpolate(
        posemb_grid,
        size=(hight, width),
        mode='bilinear',
        align_corners=False
    )

    posemb_grid = posemb_grid.permute(
        0, 2, 3, 1
    ).reshape(1, hight * width, -1)

    return posemb_grid
