from lib.test.utils import TrackerParams
import glob
import os
from lib.test.evaluation.environment import env_settings
from lib.config.bat.config import cfg, update_config_from_file


def _resolve_test_checkpoint(prj_dir, save_dir, yaml_name, epoch, network_path=None):
    """
    Pick a weight file for evaluation. Order:
    1) SFETRACK_CHECKPOINT or a legacy checkpoint environment variable
    2) models/SFETrack_{yaml}.pth(.tar) or legacy model filenames
    3) the same model filenames under the optional network_path
    4) {save_dir}/checkpoints/BATrack_ep{epoch:04d}.pth.tar
    5) latest BATrack_ep*.pth.tar in {save_dir}/checkpoints
    6) legacy {save_dir}/checkpoints/train/bat/{yaml}/...
    """
    env_ckpt = (
        os.environ.get("SFETRACK_CHECKPOINT")
        or os.environ.get("CTETRACK_CHECKPOINT")
        or os.environ.get("BAT_CHECKPOINT")
        or os.environ.get("TEST_CHECKPOINT")
    )
    if env_ckpt:
        env_ckpt = os.path.expanduser(env_ckpt)
        if os.path.isfile(env_ckpt):
            return env_ckpt
        raise FileNotFoundError(
            f"Configured test checkpoint is not a file: {env_ckpt}"
        )

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    legacy_train_subdir = os.path.join(ckpt_dir, "train", "bat", yaml_name)
    ep = int(epoch) if epoch is not None else 0

    candidates = [
        os.path.join(prj_dir, "models", f"SFETrack_{yaml_name}.pth"),
        os.path.join(prj_dir, "models", f"SFETrack_{yaml_name}.pth.tar"),
        os.path.join(prj_dir, "models", f"CTETrack_{yaml_name}.pth"),
        os.path.join(prj_dir, "models", f"CTETrack_{yaml_name}.pth.tar"),
        os.path.join(prj_dir, "models", f"BAT_{yaml_name}.pth"),
        os.path.join(prj_dir, "models", f"BAT_{yaml_name}.pth.tar"),
    ]
    if network_path:
        candidates.extend(
            [
                os.path.join(network_path, f"SFETrack_{yaml_name}.pth"),
                os.path.join(network_path, f"SFETrack_{yaml_name}.pth.tar"),
                os.path.join(network_path, f"CTETrack_{yaml_name}.pth"),
                os.path.join(network_path, f"CTETrack_{yaml_name}.pth.tar"),
                os.path.join(network_path, f"BAT_{yaml_name}.pth"),
                os.path.join(network_path, f"BAT_{yaml_name}.pth.tar"),
            ]
        )
    candidates.extend(
        [
            os.path.join(ckpt_dir, f"BATrack_ep{ep:04d}.pth.tar"),
            os.path.join(legacy_train_subdir, f"BATrack_ep{ep:04d}.pth.tar"),
        ]
    )
    for path in candidates:
        if os.path.isfile(path):
            return path

    for directory in [ckpt_dir, legacy_train_subdir]:
        if os.path.isdir(directory):
            ckpt_list = sorted(glob.glob(os.path.join(directory, "BATrack_ep*.pth.tar")))
            if ckpt_list:
                return ckpt_list[-1]

    tried = "\n  ".join(candidates)
    for directory in [ckpt_dir, legacy_train_subdir]:
        if os.path.isdir(directory):
            tried += f"\n  (glob) {os.path.join(directory, 'BATrack_ep*.pth.tar')}"
    raise FileNotFoundError(
        "No checkpoint found for testing. Tried:\n  "
        + tried
        + "\nSet SFETRACK_CHECKPOINT=/path/to/checkpoint.pth.tar or place weights under models/."
    )


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()
    settings = env_settings()
    prj_dir = settings.prj_dir
    save_dir = settings.save_dir
    network_path = getattr(settings, "network_path", None) or None

    yaml_file = os.path.join(prj_dir, 'experiments/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)


    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE


    params.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 2)
    params.update_intervals = getattr(cfg.TEST, "UPDATE_INTERVAL", 50)
    params.update_threshold = getattr(cfg.TEST, "UPDATE_THRESHOLD", 0.65)

    params.template_update_mode = "tptu"
    params.target_preservation_threshold = getattr(cfg.TEST, "TARGET_PRESERVATION_THRESHOLD", 0.4)


    test_epoch = epoch if epoch is not None else getattr(cfg.TEST, "EPOCH", 0)
    params.checkpoint = _resolve_test_checkpoint(
        prj_dir, save_dir, yaml_name, test_epoch, network_path=network_path
    )
    print("test checkpoint:", params.checkpoint)

    params.save_all_boxes = False

    return params
