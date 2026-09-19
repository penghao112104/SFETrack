import os
import glob
import re
import copy
import torch
import yaml
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from lib.train.trainers import LTRTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from .base_functions import *
from lib.models.bat import build_batrack
from lib.train.actors import BATActor
import importlib
from ..utils.focal_loss import FocalLoss


def _latest_checkpoint_path(*directories):
    checkpoint_paths = []
    for directory in directories:
        if directory:
            checkpoint_paths.extend(glob.glob(os.path.join(directory, "*_ep*.pth.tar")))
    return sorted(checkpoint_paths)[-1] if checkpoint_paths else None


def _checkpoint_epoch(checkpoint_path):
    if not checkpoint_path:
        return 0
    match = re.search(r"_ep(\d+)\.pth\.tar$", os.path.basename(checkpoint_path))
    return int(match.group(1)) if match else 0


def _apply_cfg_overrides(cfg, overrides):
    if not overrides:
        return
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid cfg override '{override}', expected KEY.SUBKEY=VALUE")
        key_path, raw_value = override.split("=", 1)
        keys = [k for k in key_path.split(".") if k]
        if len(keys) == 0:
            raise ValueError(f"Invalid cfg override key in '{override}'")

        node = cfg
        for key in keys[:-1]:
            if key not in node:
                raise ValueError(f"Config override path does not exist: {key_path}")
            node = node[key]
        final_key = keys[-1]
        if final_key not in node:
            raise ValueError(f"Config override key does not exist: {key_path}")
        node[final_key] = yaml.safe_load(raw_value)


def run(settings):
    settings.description = 'Training script for bat'


    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)

    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    _apply_cfg_overrides(cfg, getattr(settings, "cfg_overrides", []))

    update_settings(settings, cfg)


    loader_train, loader_val = build_dataloaders(cfg, settings)


    ckpt_dir = os.path.join(settings.save_dir, "checkpoints")
    legacy_ckpt_dir = os.path.join(ckpt_dir, settings.project_path)
    explicit_resume_checkpoint = getattr(settings, "resume_checkpoint", None)
    no_resume = bool(getattr(settings, "no_resume", False))
    if explicit_resume_checkpoint:
        latest_ckpt_path = explicit_resume_checkpoint
    elif no_resume:
        latest_ckpt_path = None
    else:
        latest_ckpt_path = _latest_checkpoint_path(ckpt_dir, legacy_ckpt_dir)
    resume_epoch = _checkpoint_epoch(latest_ckpt_path)
    initial_train_epoch = resume_epoch + 1
    has_latest_ckpt = latest_ckpt_path is not None
    load_pretrained = not has_latest_ckpt
    if settings.local_rank in [-1, 0]:
        if explicit_resume_checkpoint:
            print(f"[init] Using the specified checkpoint and skipping PRETRAIN_FILE: {explicit_resume_checkpoint}")
        elif no_resume:
            print("[init] no_resume=True; ignoring existing checkpoints and using PRETRAIN_FILE.")
        elif has_latest_ckpt:
            print(f"[init] Checkpoint found; skipping PRETRAIN_FILE and resuming from: {ckpt_dir}")
        else:
            print("[init] No checkpoint found; initializing from PRETRAIN_FILE.")

    build_cfg = cfg
    stage_start_epoch = int(getattr(cfg.TRAIN, "MOE_STAGE_START_EPOCH", 0))
    moe_enabled = bool(getattr(getattr(cfg.MODEL, "MOE", None), "ENABLE", False))
    if moe_enabled and stage_start_epoch > 0 and initial_train_epoch < stage_start_epoch:
        build_cfg = copy.deepcopy(cfg)
        build_cfg.MODEL.MOE.ENABLE = False
        if settings.local_rank in [-1, 0]:
            print(
                f"[init] building no-MoE stage-1 model for epoch {initial_train_epoch} "
                f"(< {stage_start_epoch})"
            )
    elif settings.local_rank in [-1, 0]:
        print(f"[init] building model with MoE enabled={moe_enabled}")

    if settings.script_name == "bat":
        net = build_batrack(build_cfg, load_pretrained=load_pretrained)
    else:
        raise ValueError("illegal script name")

    if settings.local_rank != -1:
        torch.cuda.set_device(settings.local_rank)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        torch.cuda.set_device(0)
        settings.device = torch.device("cuda:0")
    net = net.to(settings.device)


    if settings.local_rank != -1:


        net = DDP(net,
            device_ids=[settings.local_rank],
            output_device=settings.local_rank,
            find_unused_parameters=True
        )
    else:

        settings.device = torch.device("cuda:0")


    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)


    if settings.script_name == "bat":

        focal_loss = FocalLoss()

        objective = {
            'giou': giou_loss,
            'l1': l1_loss,
            'focal': focal_loss,
            'cls': BCEWithLogitsLoss()
        }

        loss_weight = {
            'giou': cfg.TRAIN.GIOU_WEIGHT,
            'l1': cfg.TRAIN.L1_WEIGHT,
            'focal': 1.,
            'cls': 1.0
        }

        actor = BATActor(
            net=net,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )
    else:
        raise ValueError("illegal script name")


    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg, initial_epoch=initial_train_epoch)

    use_amp = getattr(cfg.TRAIN, "AMP", False)

    settings.save_epoch_interval = getattr(cfg.TRAIN, "SAVE_EPOCH_INTERVAL", 1)

    settings.save_last_n_epoch = getattr(cfg.TRAIN, "SAVE_LAST_N_EPOCH", 1)


    if loader_val is None:
        trainer = LTRTrainer(
            actor,
            [loader_train],
            optimizer,
            settings,
            lr_scheduler,
            use_amp=use_amp
        )
    else:
        trainer = LTRTrainer(
            actor,
            [loader_train, loader_val],
            optimizer,
            settings,
            lr_scheduler,
            use_amp=use_amp
        )


    load_latest = not no_resume
    if explicit_resume_checkpoint:
        trainer.load_checkpoint(explicit_resume_checkpoint)
        load_latest = False

    trainer.train(
        cfg.TRAIN.EPOCH,
        load_latest=load_latest,
        fail_safe=True
    )
