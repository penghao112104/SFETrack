from lib.models.bat import build_batrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target

import cv2
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class BATTrack(BaseTracker):
    def __init__(self, params):
        super(BATTrack, self).__init__(params)
        network = build_batrack(params.cfg, training=False)

        ckpt_obj = torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)
        state = ckpt_obj['net'] if isinstance(ckpt_obj, dict) and 'net' in ckpt_obj else ckpt_obj
        network.load_state_dict(state, strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        if hasattr(self.network, "set_moe_dynamic_history"):
            self.network.set_moe_dynamic_history(True)
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()


        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.debug = params.debug
        self.frame_id = 0

        self.save_all_boxes = params.save_all_boxes


        self.num_template = getattr(params, "num_template", 2)
        self.update_intervals = getattr(params, "update_intervals", 50)
        self.update_threshold = getattr(params, "update_threshold", 0.65)
        self.template_update_mode = str(getattr(params, "template_update_mode", "tptu")).lower()
        self.target_preservation_threshold = getattr(params, "target_preservation_threshold", 0.4)
        self.update_fallback_threshold = getattr(params, "update_fallback_threshold", 0.5)
        self._window_best_score = float("-inf")
        self._window_best_template = None
        print(
            "[template-update] "
            f"interval={self.update_intervals}, "
            f"update_threshold={self.update_threshold}, "
            f"target_preservation_threshold={self.target_preservation_threshold}"
        )

    def initialize(self, image, info: dict):
        if hasattr(self.network, "set_moe_dynamic_history"):
            self.network.set_moe_dynamic_history(True)
        if hasattr(self.network, "reset_moe_history"):
            self.network.reset_moe_history()


        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():

            self.z_dict = [template for _ in range(self.num_template)]
        self.initial_image = image.copy()
        self.initial_bbox = list(info["init_bbox"])

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)


        self.state = info['init_bbox']
        self.frame_id = 0
        self._window_best_score = float("-inf")
        self._window_best_template = None
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    @staticmethod
    def _xywh_iou(box_a, box_b):
        ax, ay, aw, ah = [float(v) for v in box_a]
        bx, by, bw, bh = [float(v) for v in box_b]
        ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
        bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _map_box_back_from_state(prev_state, pred_box: list, search_size: int, resize_factor: float):
        cx_prev = prev_state[0] + 0.5 * prev_state[2]
        cy_prev = prev_state[1] + 0.5 * prev_state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def _set_moe_dynamic_history(self, enabled: bool):
        if hasattr(self.network, "set_moe_dynamic_history"):
            self.network.set_moe_dynamic_history(enabled)

    def _get_moe_dynamic_history(self):
        backbone = getattr(self.network, "backbone", None)
        moe_bridge = getattr(backbone, "moe_bridge", None)
        if moe_bridge is None or not hasattr(moe_bridge, "dynamic_history"):
            return None
        return bool(moe_bridge.dynamic_history)

    def _forward_with_templates(self, image, search_state, templates, preserve_moe_history=False):
        H, W, _ = image.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, search_state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)

        prev_moe_dynamic_history = self._get_moe_dynamic_history()
        if preserve_moe_history:
            self._set_moe_dynamic_history(False)
        with torch.no_grad():
            try:

                template_tensor = torch.stack(templates, dim=0)
                out_dict = self.network.forward(
                    template=template_tensor,
                    search=search,
                    ce_template_mask=self.box_mask_z,
                )
            finally:
                if preserve_moe_history and prev_moe_dynamic_history is not None:
                    self._set_moe_dynamic_history(prev_moe_dynamic_history)

        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)

        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()

        mapped_box = self._map_box_back_from_state(search_state, pred_box, self.params.search_size, resize_factor)
        clipped_box = clip_box(mapped_box, H, W, margin=10)
        return clipped_box, max_score, pred_boxes, response, resize_factor

    def _make_template(self, image, state):
        z_patch_arr, _, _ = sample_target(
            image,
            state,
            self.params.template_factor,
            output_sz=self.params.template_size,
        )
        return self.preprocessor.process(z_patch_arr)

    def _remember_window_best_template(self, image, score_val):
        if score_val > self._window_best_score:
            self._window_best_score = float(score_val)
            self._window_best_template = self._make_template(image, self.state)

    def _reset_template_update_window(self):
        self._window_best_score = float("-inf")
        self._window_best_template = None

    def _target_preservation_score(self, candidate_template):
        if not hasattr(self, "initial_image") or not hasattr(self, "initial_bbox"):
            return 0.0


        templates = [candidate_template for _ in range(self.num_template)]
        pred_bbox, _, _, _, _ = self._forward_with_templates(
            self.initial_image,
            self.initial_bbox,
            templates,
            preserve_moe_history=True,
        )
        return self._xywh_iou(pred_bbox, self.initial_bbox)

    def _maybe_update_template_tptu(self, image, score_val):
        if self.num_template < 2:
            return
        self._remember_window_best_template(image, score_val)
        if self.update_intervals <= 0 or self.frame_id % self.update_intervals != 0:
            return

        window_best_score = self._window_best_score
        window_best_template = self._window_best_template
        if window_best_template is None or window_best_score <= self.update_threshold:
            self._reset_template_update_window()
            return

        preservation_score = self._target_preservation_score(window_best_template)
        preservation_pass = preservation_score > self.target_preservation_threshold
        if preservation_pass:
            self.z_dict[1] = window_best_template
        self._reset_template_update_window()

    def _maybe_update_template_score(self, image, score_val):
        if self.num_template < 2:
            return
        if score_val > self._window_best_score:
            self._window_best_score = score_val
            self._window_best_template = self._make_template(image, self.state)

        if self.update_intervals <= 0 or self.frame_id % self.update_intervals != 0:
            return
        if score_val > self.update_threshold:
            self.z_dict[1] = self._make_template(image, self.state)
        elif (
            self._window_best_template is not None
            and self._window_best_score >= self.update_fallback_threshold
        ):
            self.z_dict[1] = self._window_best_template
        self._reset_template_update_window()

    def track(self, image, info: dict = None):
        self.frame_id += 1
        prev_state = list(self.state)
        self.state, max_score, pred_boxes, response, resize_factor = self._forward_with_templates(
            image,
            self.state,
            self.z_dict,
            preserve_moe_history=False,
        )

        if self.num_template > 1:

            conf_score, _ = torch.max(response.flatten(1), dim=1)
            score_val = conf_score.mean().item()
            if self.template_update_mode == "tptu":
                self._maybe_update_template_tptu(image, score_val)
            elif self.template_update_mode == "score":
                self._maybe_update_template_score(image, score_val)


        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)


        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.params.search_size / resize_factor,
                resize_factor,
                prev_state,
            )
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        return self._map_box_back_from_state(self.state, pred_box, self.params.search_size, resize_factor)

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float, prev_state=None):
        prev_state = self.state if prev_state is None else prev_state
        cx_prev, cy_prev = prev_state[0] + 0.5 * prev_state[2], prev_state[1] + 0.5 * prev_state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return BATTrack
