import os
import argparse
import importlib
import cv2 as cv
import torch.backends.cudnn
import torch.distributed as dist
import random
import numpy as np

torch.backends.cudnn.benchmark = False
import _init_paths
import lib.train.admin.settings as ws_settings
from lib.utils.misc import setup_for_distributed


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1"):
        return True
    if value in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None, base_seed=None,
                 cfg_overrides=None, resume_checkpoint=None, no_resume=False,
                 ):

    if save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")


    cv.setNumThreads(0)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('script_name: {}.py  config_name: {}.yaml'.format(script_name, config_name))


    rank_seed = None
    if base_seed is not None:
        if local_rank != -1:
            rank_seed = base_seed + local_rank
            init_seeds(rank_seed)
        else:
            rank_seed = base_seed
            init_seeds(base_seed)
        print("seed: {}  rank_seed: {}".format(base_seed, rank_seed))


    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = 'train/{}/{}'.format(script_name, config_name)

    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)


    settings.env.workspace_dir = settings.save_dir
    settings.env.tensorboard_dir = os.path.join(settings.save_dir, 'tensorboard')
    settings.seed = base_seed
    settings.rank_seed = rank_seed
    settings.cfg_overrides = cfg_overrides or []
    settings.resume_checkpoint = (
        os.path.abspath(os.path.expanduser(resume_checkpoint))
        if resume_checkpoint else None
    )
    settings.no_resume = bool(no_resume)

    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s.yaml' % config_name)

    expr_module = importlib.import_module('lib.train.train_script')


    expr_func = getattr(expr_module, 'run')


    expr_func(settings)


def main():

    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, required=True,help='Name of the train script.')
    parser.add_argument('--config', type=str, required=True,help="Name of the config file.")
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True,help='Set cudnn benchmark on (1) or off (0)')
    parser.add_argument('--local_rank', '--local-rank', default=-1,type=int, help='node rank for distributed training')
    parser.add_argument('--save_dir', type=str,help='the directory to save checkpoints and logs')
    parser.add_argument('--seed', type=int, default=0,help='seed for random numbers')
    parser.add_argument('--cfg_override', action='append', default=[],
                        help='Runtime config override in KEY.SUBKEY=VALUE form. Can be used multiple times.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Explicit checkpoint path to resume from instead of the latest checkpoint.')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not load existing checkpoints; start from PRETRAIN_FILE.')

    args = parser.parse_args()


    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])


    if args.local_rank != -1:

        dist.init_process_group(backend='nccl')

        torch.cuda.set_device(args.local_rank)
        setup_for_distributed(dist.get_rank() == 0)
    else:

        torch.cuda.set_device(0)


    run_training(
        args.script,
        args.config,
        cudnn_benchmark=args.cudnn_benchmark,
        local_rank=args.local_rank,
        save_dir=args.save_dir,
        base_seed=args.seed,
        cfg_overrides=args.cfg_override,
        resume_checkpoint=args.resume_checkpoint,
        no_resume=args.no_resume
    )

if __name__ == '__main__':
    main()
