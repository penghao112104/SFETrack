import math
import torch
import torch.nn.functional as F


def generate_bbox_mask(bbox_mask, bbox):
    b, h, w = bbox_mask.shape


    for i in range(b):

        bbox_i = bbox[i].cpu().tolist()


        x_start = int(bbox_i[0])
        y_start = int(bbox_i[1])
        width = int(bbox_i[2])
        height = int(bbox_i[3])


        bbox_mask[i,
                  y_start : y_start + height - 1,
                  x_start : x_start + width - 1] = 1

    return bbox_mask


def generate_mask_cond(cfg, bs, device, gt_bbox):

    template_size = cfg.DATA.TEMPLATE.SIZE

    stride = cfg.MODEL.BACKBONE.STRIDE

    template_feat_size = template_size // stride


    mode = cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE

    if mode == 'ALL':

        box_mask_z = None

    elif mode == 'CTR_POINT':


        if template_feat_size == 8:
            index = slice(3, 4)
        elif template_feat_size == 12:
            index = slice(5, 6)
        elif template_feat_size == 7:
            index = slice(3, 4)
        elif template_feat_size == 14:
            index = slice(6, 7)
        else:
            raise NotImplementedError(f"Unsupported feature size {template_feat_size} for CTR_POINT")


        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)

        box_mask_z[:, index, index] = 1

        box_mask_z = box_mask_z.flatten(1).to(torch.bool)

    elif mode == 'CTR_REC':

        if template_feat_size == 8:
            index = slice(3, 5)


        elif template_feat_size == 12:
            index = slice(5, 7)
        elif template_feat_size == 7:
            index = slice(3, 4)
        else:
            raise NotImplementedError(f"Unsupported feature size {template_feat_size} for CTR_REC")

        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)

    elif mode == 'GT_BOX':


        box_mask_z = torch.zeros([bs, template_size, template_size], device=device)


        box_mask_z = generate_bbox_mask(box_mask_z, gt_bbox * template_size)


        box_mask_z = box_mask_z.unsqueeze(1).to(torch.float)


        box_mask_z = F.interpolate(box_mask_z,
                                   scale_factor=1. / cfg.MODEL.BACKBONE.STRIDE,
                                   mode='bilinear',
                                   align_corners=False)


        box_mask_z = box_mask_z.flatten(1).to(torch.bool)

    else:
        raise NotImplementedError(f"Unknown CE_TEMPLATE_RANGE: {mode}")

    return box_mask_z


def adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH,
                     base_keep_rate=0.5, max_keep_rate=1, iters=-1):

    if epoch < warmup_epochs:
        return 1.0


    if epoch >= total_epochs:
        return base_keep_rate


    if iters == -1:
        iters = epoch * ITERS_PER_EPOCH


    total_iters = ITERS_PER_EPOCH * (total_epochs - warmup_epochs)


    current_iters = iters - ITERS_PER_EPOCH * warmup_epochs


    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) \
        * (math.cos(current_iters / total_iters * math.pi) + 1) * 0.5

    return keep_rate
