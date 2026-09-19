import torch
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np


from lib.train.dataset import LasHeR


from lib.train.data import sampler, opencv_loader, processing, LTRLoader

import lib.train.data.transforms as tfm

from lib.utils.misc import is_main_process


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed):
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def update_settings(settings, cfg):

    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL


    settings.search_area_factor = {
        'template': cfg.DATA.TEMPLATE.FACTOR,
        'search': cfg.DATA.SEARCH.FACTOR
    }


    settings.output_sz = {
        'template': cfg.DATA.TEMPLATE.SIZE,
        'search': cfg.DATA.SEARCH.SIZE
    }


    settings.center_jitter_factor = {
        'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
        'search': cfg.DATA.SEARCH.CENTER_JITTER
    }


    settings.scale_jitter_factor = {
        'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
        'search': cfg.DATA.SEARCH.SCALE_JITTER
    }


    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None


    settings.batchsize = cfg.TRAIN.BATCH_SIZE


    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


    settings.fix_bn = getattr(cfg.TRAIN, "FIX_BN", False)


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    split_map = {
        "LasHeR_all": "all",
        "LasHeR_train": "train",
        "LasHeR_val": "val",
    }

    datasets = []
    for name in name_list:
        if name not in split_map:
            raise ValueError(
                f"Unsupported training dataset '{name}'. This project keeps only LasHeR datasets."
            )
        datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split=split_map[name]))

    return datasets


def build_dataloaders(cfg, settings):


    transform_joint = tfm.Transform(
        tfm.ToGrayscale(probability=0.05),
        tfm.RandomHorizontalFlip(probability=0.5)
    )

    transform_train = tfm.Transform(
        tfm.ToTensorAndJitter(0.2),
        tfm.RandomHorizontalFlip_Norm(probability=0.5),
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    )

    transform_val = tfm.Transform(
        tfm.ToTensor(),
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    )


    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor


    data_processing_train = processing.BATProcessing(
        search_area_factor=search_area_factor,
        output_sz=output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode='sequence',
        transform=transform_train,
        joint_transform=transform_joint,
        settings=settings
    )


    data_processing_val = processing.BATProcessing(
        search_area_factor=search_area_factor,
        output_sz=output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode='sequence',
        transform=transform_val,
        joint_transform=transform_joint,
        settings=settings
    )


    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)


    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")


    load_aux = bool(getattr(cfg.MODEL.MOE, "ENABLE", False))
    aux_past_gap = getattr(cfg.TRAIN, "MOE_AUX_PAST_GAP", 1)
    aux_future_gap = getattr(cfg.TRAIN, "MOE_AUX_FUTURE_GAP", 1)
    aux_start_epoch = getattr(cfg.TRAIN, "MOE_STAGE_START_EPOCH", 0)
    dataset_train = sampler.TrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_train,
        frame_sample_mode=sampler_mode,
        load_aux_temporal_frames=load_aux,
        aux_past_gap=aux_past_gap,
        aux_future_gap=aux_future_gap,
        aux_temporal_start_epoch=aux_start_epoch,
    )


    sampler_seed = int(getattr(settings, "seed", 0) or 0)
    rank_seed = int(getattr(settings, "rank_seed", sampler_seed) or sampler_seed)

    train_sampler = DistributedSampler(dataset_train, seed=sampler_seed) \
        if settings.local_rank != -1 else None

    shuffle = False if settings.local_rank != -1 else True


    loader_train = LTRLoader(
        'train',
        dataset_train,
        training=True,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        sampler=train_sampler,
        worker_init_fn=seed_worker,
        generator=make_generator(rank_seed)
    )

    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(
            datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
            p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
            num_search_frames=settings.num_search,
            num_template_frames=settings.num_template,
            processing=data_processing_val,
            frame_sample_mode=sampler_mode,
            load_aux_temporal_frames=False,
        )

        val_sampler = DistributedSampler(dataset_val, seed=sampler_seed + 1) \
            if settings.local_rank != -1 else None

        loader_val = LTRLoader(
            'val',
            dataset_val,
            training=False,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKER,
            drop_last=True,
            stack_dim=1,
            sampler=val_sampler,
            worker_init_fn=seed_worker,
            generator=make_generator(rank_seed + 1),
            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL
        )

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg, initial_epoch=1):


    mode = str(getattr(cfg.TRAIN, "MOE_SCHEDULE_MODE", "legacy")).lower()
    initial_epoch = int(initial_epoch)
    in_initial_all_moe_stage = False
    if mode == "staged":
        all0 = int(getattr(cfg.TRAIN, "MOE_STAGE_START_EPOCH", 0))
        all1 = int(getattr(cfg.TRAIN, "MOE_STAGE_ALL_MOE_END", 0))
        in_initial_all_moe_stage = all0 > 0 and all0 <= initial_epoch <= all1
        freeze_bb = bool(getattr(cfg.TRAIN, "MOE_STAGE_FREEZE_BACKBONE", False))
        freeze_head = bool(getattr(cfg.TRAIN, "MOE_STAGE_FREEZE_BOX_HEAD", False))
        freeze_router = bool(getattr(cfg.TRAIN, "MOE_STAGE_FREEZE_ROUTER", True))

        def _is_moe_param(name: str) -> bool:

            return ("moe_bridge" in name) or ("moe_bridges" in name)

        def _is_fusion_param(name: str) -> bool:
            return "template_response_fusion" in name

        for n, p in net.named_parameters():
            is_moe = _is_moe_param(n)
            if in_initial_all_moe_stage:
                if is_moe:
                    is_router_param = ".router." in n
                    p.requires_grad = not (freeze_router and is_router_param)
                elif _is_fusion_param(n):
                    p.requires_grad = True
                elif "backbone" in n:
                    p.requires_grad = not freeze_bb
                elif "box_head" in n:
                    p.requires_grad = not freeze_head
                else:
                    p.requires_grad = True
            else:


                p.requires_grad = not is_moe

    train_type = getattr(cfg.TRAIN.PROMPT, "TYPE", "")


    if 'bat' in train_type:
        moe_mul = float(getattr(cfg.TRAIN, "MOE_LR_MULTIPLIER", 1.0))
        head_mul = float(getattr(cfg.TRAIN, "HEAD_LR_MULTIPLIER", 1.0))
        stage_head_mul = float(getattr(cfg.TRAIN, "MOE_STAGE_HEAD_LR_MULTIPLIER", 0.0))
        fusion_mul = 1.0
        stage_fusion_mul = float(
            getattr(cfg.TRAIN, "MOE_STAGE_FUSION_LR_MULTIPLIER", 0.0)
        )
        if mode == "staged" and in_initial_all_moe_stage and stage_head_mul > 0:
            head_mul = stage_head_mul
        if mode == "staged" and in_initial_all_moe_stage and stage_fusion_mul > 0:
            fusion_mul = stage_fusion_mul
        lr_bb = cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER
        lr_moe = cfg.TRAIN.LR * moe_mul
        lr_fusion = cfg.TRAIN.LR * fusion_mul
        lr_head = cfg.TRAIN.LR * head_mul

        def _is_moe_param(name: str) -> bool:
            return ("moe_bridge" in name) or ("moe_bridges" in name)

        def _is_fusion_param(name: str) -> bool:
            return "template_response_fusion" in name

        p_bb = [
            p for n, p in net.named_parameters()
            if "backbone" in n and not _is_moe_param(n) and not _is_fusion_param(n)
        ]
        p_moe = [
            p for n, p in net.named_parameters()
            if _is_moe_param(n)
        ]
        p_fusion = [
            p for n, p in net.named_parameters()
            if _is_fusion_param(n)
        ]
        p_head = [
            p for n, p in net.named_parameters()
            if "box_head" in n
        ]
        p_other = [
            p for n, p in net.named_parameters()
            if (
                "backbone" not in n
                and "box_head" not in n
                and not _is_moe_param(n)
                and not _is_fusion_param(n)
            )
        ]
        param_dicts = []
        if len(p_bb) > 0:
            param_dicts.append({"params": p_bb, "lr": lr_bb})
        if len(p_moe) > 0:
            param_dicts.append({"params": p_moe, "lr": lr_moe})
        if len(p_fusion) > 0:
            param_dicts.append({"params": p_fusion, "lr": lr_fusion})
        if len(p_head) > 0:
            param_dicts.append({"params": p_head, "lr": lr_head})
        if len(p_other) > 0:
            param_dicts.append({"params": p_other, "lr": cfg.TRAIN.LR})
    else:

        p_non_backbone = [
            p for n, p in net.named_parameters()
            if "backbone" not in n
        ]
        p_backbone = [
            p for n, p in net.named_parameters()
            if "backbone" in n
        ]
        param_dicts = []
        if len(p_non_backbone) > 0:
            param_dicts.append({"params": p_non_backbone})
        if len(p_backbone) > 0:
            param_dicts.append({"params": p_backbone, "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER})


    if len(param_dicts) == 0:
        raise ValueError("No trainable parameters found. Please check requires_grad settings.")


    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    else:
        raise ValueError("Unsupported Optimizer")


    if cfg.TRAIN.SCHEDULER.TYPE == 'step':

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.TRAIN.LR_DROP_EPOCH
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
            gamma=cfg.TRAIN.SCHEDULER.GAMMA
        )
    else:
        raise ValueError("Unsupported scheduler")

    return optimizer, lr_scheduler
