
from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu
import torch.nn.functional as F
from lib.models.bat.utils import resolve_moe_layer_indices
from lib.utils.moe_loss import modality_router_balance_loss


class BATActor(BaseActor):

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)

        self.loss_weight = loss_weight

        self.settings = settings

        self.cfg = cfg

        self._moe_schedule_tag = None
        self._moe_expert_training_active = False
        self._moe_router_training_active = False

    def _is_moe_forward_active(self, epoch: int) -> bool:
        """Return whether the main branch should execute MoE at this epoch."""
        if not getattr(self.cfg.MODEL, "MOE", None) or not getattr(self.cfg.MODEL.MOE, "ENABLE", False):
            return False
        start_epoch = int(getattr(self.cfg.TRAIN, "MOE_STAGE_START_EPOCH", 0))
        if start_epoch <= 0:
            return True
        return int(epoch) >= start_epoch

    def _prepare_search_image(self, img6ch):
        if img6ch.dim() == 5:
            if img6ch.shape[0] != 1:
                raise ValueError(f"Expected one aux frame, got shape {tuple(img6ch.shape)}")
            img6ch = img6ch[0]
        elif img6ch.dim() != 4:
            raise ValueError(f"Unexpected aux image shape: {tuple(img6ch.shape)}")
        return img6ch

    def _build_template_tensor_from_batch(self, data):
        template_imgs = data['template_images']
        if template_imgs.dim() == 5:
            template_list = []
            for i in range(self.settings.num_template):
                template_img_i = template_imgs[i].view(-1, *template_imgs.shape[2:])
                template_list.append(template_img_i)
            return torch.stack(template_list, dim=0)
        if template_imgs.dim() == 4:
            return template_imgs.unsqueeze(0)
        raise ValueError(f"Unexpected template_images shape: {template_imgs.shape}")

    def _build_ce_controls(self, template_tensor, data):
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(
                self.cfg,
                template_tensor.shape[1],
                template_tensor.device,
                data['template_anno'][0],
            )
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(
                data['epoch'],
                warmup_epochs=ce_start_epoch,
                total_epochs=ce_start_epoch + ce_warm_epoch,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0],
            )
        return box_mask_z, ce_keep_rate

    def _encode_search_region_layer_tokens(self, backbone, template_tensor, img6ch, box_mask_z, ce_keep_rate, layers):
        img6ch = self._prepare_search_image(img6ch)
        with torch.no_grad():
            _, aux_dict = backbone.forward_features(
                z=template_tensor,
                x=img6ch,
                ce_template_mask=box_mask_z,
                ce_keep_rate=ce_keep_rate,
                collect_layer_search_tokens=True,
                collect_layers=layers,
                disable_moe=True,
            )

        rgb_map = {int(k): v.detach() for k, v in aux_dict.get("moe_layer_search_tokens_rgb", {}).items()}
        tir_map = {int(k): v.detach() for k, v in aux_dict.get("moe_layer_search_tokens_tir", {}).items()}
        return rgb_map, tir_map

    @staticmethod
    def _moe_temporal_alignment_loss(
        pred,
        tgt,
    ):
        """MSE over the batch, search-token, and channel dimensions."""
        return F.mse_loss(pred, tgt, reduction="mean")

    @staticmethod
    def _collect_temporal_aux_predictions(pred_entry):
        if pred_entry is None:
            return []
        if torch.is_tensor(pred_entry):
            return [pred_entry]
        if isinstance(pred_entry, dict):
            preds = []
            tensor = pred_entry.get("decoded_search", None)
            if torch.is_tensor(tensor):
                preds.append(tensor)
            return preds
        return []

    @staticmethod
    def _validate_temporal_aux_shape(pred: torch.Tensor, target: torch.Tensor, layer_idx: int, tag: str):
        if pred.shape != target.shape:
            raise ValueError(
                f"MOE aux shape mismatch at layer {layer_idx} ({tag}): "
                f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )

    def _compute_moe_aux_temporal_loss(
        self,
        pred_dict,
        gt_dict,
    ):
        cfg = self.cfg
        self._last_moe_aux_status = {}
        if not getattr(cfg.MODEL.MOE, "ENABLE", False):
            return None
        if "moe_aux_hist_ctx" not in pred_dict or not pred_dict["moe_aux_hist_ctx"]:
            return None

        fut_rgb_l, fut_tir_l = {}, {}
        future_feature_available = (
            gt_dict.get("aux_future_images", None) is not None
            and gt_dict.get("aux_future_anno", None) is not None
        )
        if future_feature_available:
            net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
            bb = net.backbone
            aux_layers = resolve_moe_layer_indices(cfg.MODEL.MOE, len(bb.blocks))
            template_tensor = self._build_template_tensor_from_batch(gt_dict)
            box_mask_z, ce_keep_rate = self._build_ce_controls(template_tensor, gt_dict)
            fut_rgb_l, fut_tir_l = self._encode_search_region_layer_tokens(
                bb,
                template_tensor,
                gt_dict["aux_future_images"],
                box_mask_z,
                ce_keep_rate,
                aux_layers,
            )


        past_feature_losses = []
        future_feature_losses = []

        for layer_idx, aux_rgb, aux_tir in pred_dict["moe_aux_hist_ctx"]:
            if aux_rgb is None or aux_tir is None:
                continue
            layer_idx = int(layer_idx)
            cur_rgb_tgt = aux_rgb.get("target_search", None)
            cur_tir_tgt = aux_tir.get("target_search", None)

            if torch.is_tensor(cur_rgb_tgt) and torch.is_tensor(cur_tir_tgt):
                past_aux_rgb = aux_rgb.get("past", None)
                for pred_past_rgb in self._collect_temporal_aux_predictions(past_aux_rgb):
                    self._validate_temporal_aux_shape(
                        pred_past_rgb, cur_rgb_tgt, layer_idx, "past-rgb"
                    )
                    feature_loss = self._moe_temporal_alignment_loss(
                        pred_past_rgb,
                        cur_rgb_tgt,
                    )
                    past_feature_losses.append(feature_loss)
                past_aux_tir = aux_tir.get("past", None)
                for pred_past_tir in self._collect_temporal_aux_predictions(past_aux_tir):
                    self._validate_temporal_aux_shape(
                        pred_past_tir, cur_tir_tgt, layer_idx, "past-tir"
                    )
                    feature_loss = self._moe_temporal_alignment_loss(
                        pred_past_tir,
                        cur_tir_tgt,
                    )
                    past_feature_losses.append(feature_loss)

            if future_feature_available:
                fut_rgb_tgt = fut_rgb_l.get(layer_idx)
                fut_tir_tgt = fut_tir_l.get(layer_idx)
                if not torch.is_tensor(fut_rgb_tgt) or not torch.is_tensor(fut_tir_tgt):
                    continue
                for pred_future_rgb in self._collect_temporal_aux_predictions(aux_rgb.get("future", None)):
                    self._validate_temporal_aux_shape(
                        pred_future_rgb, fut_rgb_tgt, layer_idx, "future-rgb"
                    )
                    feature_loss = self._moe_temporal_alignment_loss(
                        pred_future_rgb,
                        fut_rgb_tgt,
                    )
                    future_feature_losses.append(feature_loss)
                for pred_future_tir in self._collect_temporal_aux_predictions(aux_tir.get("future", None)):
                    self._validate_temporal_aux_shape(
                        pred_future_tir, fut_tir_tgt, layer_idx, "future-tir"
                    )
                    feature_loss = self._moe_temporal_alignment_loss(
                        pred_future_tir,
                        fut_tir_tgt,
                    )
                    future_feature_losses.append(feature_loss)

        terms = []
        details = {}
        if len(past_feature_losses) > 0:
            feature_past = torch.stack(past_feature_losses).mean()
            terms.append(feature_past)
            details["Loss/moe_feature_past"] = feature_past.detach()
        if len(future_feature_losses) > 0:
            feature_future = torch.stack(future_feature_losses).mean()
            terms.append(feature_future)
            details["Loss/moe_feature_future"] = feature_future.detach()

        if len(terms) == 0:
            return None
        total = torch.stack(terms).sum()
        self._last_moe_aux_status = details
        return total

    @staticmethod
    def _iter_moe_bridges(backbone):
        out = []
        bridge = getattr(backbone, "moe_bridge", None)
        if bridge is not None:
            out.append(bridge)
        blocks = getattr(backbone, "blocks", None)
        if blocks is None:
            return out
        for blk in blocks:
            bridge = getattr(blk, "moe_bridge", None)
            if bridge is not None:
                out.append(bridge)
        return out

    def _set_moe_all_requires_grad(self, bridges, value: bool):
        for bridge in bridges:
            for p in bridge.parameters():
                p.requires_grad = value

    def _set_moe_frozen_train_router_only(self, bridges):
        for bridge in bridges:
            for p in bridge.parameters():
                p.requires_grad = False
            for p in bridge.router.parameters():
                p.requires_grad = True

    def _set_backbone_non_moe_requires_grad(self, net, value: bool):
        for n, p in net.named_parameters():
            if "backbone" not in n:
                continue
            if "moe_bridge" in n:
                continue
            p.requires_grad = value

    def _set_box_head_requires_grad(self, net, value: bool):
        for n, p in net.named_parameters():
            if "box_head" in n:
                p.requires_grad = value

    def _set_fusion_requires_grad(self, net, value: bool):
        """Control the learnable full-template RGB/TIR fusion separately."""
        for n, p in net.named_parameters():
            if "template_response_fusion" in n:
                p.requires_grad = value

    def _maybe_update_moe_freeze(self, epoch: int):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        if not getattr(self.cfg.MODEL, "MOE", None) or not getattr(self.cfg.MODEL.MOE, "ENABLE", False):
            return

        mode = str(getattr(self.cfg.TRAIN, "MOE_SCHEDULE_MODE", "staged")).lower()
        if mode == "none":
            return

        bridges = self._iter_moe_bridges(net.backbone)
        if len(bridges) == 0:
            return


        if mode == "staged":
            freeze_all = int(getattr(self.cfg.TRAIN, "MOE_STAGE_FREEZE_ALL_EPOCHS", 0))
            all0, all1 = int(getattr(self.cfg.TRAIN, "MOE_STAGE_START_EPOCH", 0)), int(
                getattr(self.cfg.TRAIN, "MOE_STAGE_ALL_MOE_END", 0)
            )
            joint_start = int(getattr(self.cfg.TRAIN, "MOE_STAGE_JOINT_START", 0))
            freeze_moe_joint = bool(getattr(self.cfg.TRAIN, "MOE_STAGE_FREEZE_MOE_IN_JOINT", False))
            joint_train_router = bool(getattr(self.cfg.TRAIN, "MOE_STAGE_JOINT_TRAIN_ROUTER", True))
            freeze_bb = bool(getattr(self.cfg.TRAIN, "MOE_STAGE_FREEZE_BACKBONE", True))
            freeze_head = bool(getattr(self.cfg.TRAIN, "MOE_STAGE_FREEZE_BOX_HEAD", True))
            freeze_router = bool(getattr(self.cfg.TRAIN, "MOE_STAGE_FREEZE_ROUTER", True))

            tag = None
            train_bb, train_head = True, True
            train_router = not freeze_router
            moe_frozen_now = False
            router_train_now = False
            expert_train_now = False


            if freeze_all > 0 and epoch <= freeze_all:
                tag = f"staged:freeze_all_moe(epoch<={freeze_all})"
                moe_frozen_now = True
                self._set_moe_all_requires_grad(bridges, False)
                train_bb, train_head = True, True

            elif all0 > 0 and all0 <= epoch <= all1:
                tag = f"staged:joint_moe[{all0},{all1}]"
                self._set_moe_all_requires_grad(bridges, True)
                if freeze_router:
                    for bridge in bridges:
                        for p in bridge.router.parameters():
                            p.requires_grad = False
                router_train_now = not freeze_router
                expert_train_now = True
                train_bb, train_head = not freeze_bb, not freeze_head

            elif joint_start > 0 and epoch >= joint_start:
                tag = f"staged:joint(epoch>={joint_start})"
                if freeze_moe_joint:
                    if joint_train_router:
                        tag = f"staged:router_only(epoch>={joint_start}),experts_frozen"
                        router_train_now = True
                        moe_frozen_now = False
                        self._set_moe_frozen_train_router_only(bridges)


                        train_head = False
                    else:
                        tag += ",moe_frozen"
                        moe_frozen_now = True
                        self._set_moe_all_requires_grad(bridges, False)
                else:
                    self._set_moe_all_requires_grad(bridges, True)
                train_bb = not freeze_bb
                if not (freeze_moe_joint and joint_train_router):
                    train_head = not freeze_head
            else:

                last_stage_end = all1 if all0 > 0 and all1 >= all0 else 0

                if joint_start == 0 and last_stage_end > 0 and epoch > last_stage_end:
                    tag = "staged:joint(implicit_after_last_expert_stage)"
                    if freeze_moe_joint:
                        if joint_train_router:
                            tag += ",experts_frozen_router_train"
                            router_train_now = True
                            moe_frozen_now = False
                            self._set_moe_frozen_train_router_only(bridges)
                        else:
                            tag += ",moe_frozen"
                            moe_frozen_now = True
                            self._set_moe_all_requires_grad(bridges, False)
                    else:
                        self._set_moe_all_requires_grad(bridges, True)
                    train_bb, train_head = True, True
                elif freeze_all > 0 and epoch > freeze_all and all0 <= 0:
                    tag = "staged:joint(after_freeze_all_no_all_moe)"
                    if freeze_moe_joint:
                        if joint_train_router:
                            tag += ",experts_frozen_router_train"
                            router_train_now = True
                            moe_frozen_now = False
                            self._set_moe_frozen_train_router_only(bridges)
                        else:
                            tag += ",moe_frozen"
                            moe_frozen_now = True
                            self._set_moe_all_requires_grad(bridges, False)
                    else:
                        self._set_moe_all_requires_grad(bridges, True)
                    train_bb, train_head = True, True
                else:
                    tag = "staged:freeze_moe_backbone_only"
                    moe_frozen_now = True
                    self._set_moe_all_requires_grad(bridges, False)
                    train_bb, train_head = True, True

            self._set_backbone_non_moe_requires_grad(net, train_bb)
            self._set_box_head_requires_grad(net, train_head)

            train_fusion = bool(train_bb or expert_train_now)
            self._set_fusion_requires_grad(net, train_fusion)
            self._moe_expert_training_active = expert_train_now
            self._moe_router_training_active = router_train_now

            if tag != self._moe_schedule_tag:
                self._moe_schedule_tag = tag
                print(
                    f"[MOE] epoch={epoch} schedule={tag} "
                    f"backbone_train={train_bb} box_head_train={train_head} "
                    f"fusion_train={train_fusion} experts_train={expert_train_now} "
                    f"router_train={router_train_now}"
                )
            return

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()


    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        epoch = int(data.get('epoch', 0))
        self._maybe_update_moe_freeze(epoch)
        moe_forward_active = self._is_moe_forward_active(epoch)
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net

        template_tensor = self._build_template_tensor_from_batch(data)


        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])


        nt = self.settings.num_template
        assert template_tensor.shape[0] == nt, f"Expected {nt} templates, got {template_tensor.shape[0]}"

        box_mask_z, ce_keep_rate = self._build_ce_controls(template_tensor, data)
        moe_past_search_tokens = None
        if (
            moe_forward_active
            and getattr(self.cfg.MODEL.MOE, "ENABLE", False)
            and data.get("aux_past_images", None) is not None
        ):
            aux_layers = resolve_moe_layer_indices(self.cfg.MODEL.MOE, len(net.backbone.blocks))
            if len(aux_layers) > 0:
                past_rgb_l, past_tir_l = self._encode_search_region_layer_tokens(
                    net.backbone,
                    template_tensor,
                    data["aux_past_images"],
                    box_mask_z,
                    ce_keep_rate,
                    aux_layers,
                )
                if past_rgb_l and past_tir_l:


                    moe_past_search_tokens = {"rgb": past_rgb_l, "tir": past_tir_l}


        out_dict = self.net(template=template_tensor,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            moe_past_search_tokens=moe_past_search_tokens,
                            disable_moe=not moe_forward_active)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):

        gt_bbox = gt_dict['search_anno'][-1]

        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)


        pred_boxes = pred_dict['pred_boxes']

        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        num_queries = pred_boxes.size(1)

        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)

        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)


        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except Exception:

            _dev, _dt = pred_boxes_vec.device, pred_boxes_vec.dtype
            giou_loss = torch.tensor(0.0, device=_dev, dtype=_dt)
            iou = torch.zeros(pred_boxes_vec.shape[0], device=_dev, dtype=_dt)

        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)


        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        router_loss = None
        router_status = {}
        if self._moe_router_training_active:
            quality_logit_layers = pred_dict.get("moe_router_quality_logits", None)
            balance_losses = []
            if isinstance(quality_logit_layers, dict):
                for layer_idx, quality_logits in quality_logit_layers.items():
                    if not torch.is_tensor(quality_logits):
                        continue
                    if quality_logits.ndim != 3 or quality_logits.shape[1:] != (2, 2):
                        raise ValueError(
                            f"Router logit shape mismatch at layer {layer_idx}: "
                            f"logits={tuple(quality_logits.shape)}"
                        )
                    balance_losses.append(
                        modality_router_balance_loss(quality_logits)
                    )
            if len(balance_losses) == 0:
                raise RuntimeError(
                    "Joint MoE stage did not receive trainable routing logits."
                )
            router_balance_loss = torch.stack(balance_losses).mean()
            router_balance_weight = float(self.cfg.TRAIN.MOE_ROUTER_BALANCE_WEIGHT)
            router_loss = router_balance_weight * router_balance_loss
            loss = loss + router_loss
            router_status = {
                "Loss/moe_balance": router_loss.detach(),
            }


        self._last_moe_aux_status = {}
        aux_moe = None
        if self._moe_expert_training_active:
            aux_moe = self._compute_moe_aux_temporal_loss(
                pred_dict,
                gt_dict,
            )
        aux_moe_weight = float(self.cfg.TRAIN.MOE_AUX_FEATURE_WEIGHT)
        if aux_moe is not None:
            loss = loss + aux_moe_weight * aux_moe

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            if aux_moe is not None:
                status["Loss/moe_feature"] = aux_moe_weight * aux_moe.item()
                moe_aux_status = getattr(self, "_last_moe_aux_status", {})
                for key in ("Loss/moe_feature_past", "Loss/moe_feature_future"):
                    value = moe_aux_status.get(key, None)
                    if torch.is_tensor(value):
                        status[key] = aux_moe_weight * value.detach().item()
            for key, value in router_status.items():
                if torch.is_tensor(value):
                    status[key] = value.detach().item()
                else:
                    status[key] = float(value)
            return loss, status
        else:
            return loss
