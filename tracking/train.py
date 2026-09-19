import argparse
import random
import subprocess
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')


    parser.add_argument('--script', type=str, default='bat', help='Training script name.')
    parser.add_argument('--config', type=str, help='Experiment YAML name.')
    parser.add_argument('--save_dir', type=str, default='./output', help='Output directory.')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"],
                        default="multiple", help='Single-GPU, multi-GPU, or multi-node mode.')

    parser.add_argument('--nproc_per_node', type=int, default=torch.cuda.device_count(),
                        help='Number of GPU processes per node.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--cfg_override', action='append', default=[],
                        help='Runtime config override in KEY.SUBKEY=VALUE form. Can be used multiple times.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Explicit checkpoint path to resume from instead of the latest checkpoint.')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not load existing checkpoints; start from PRETRAIN_FILE.')


    parser.add_argument('--rank', type=int, help='Current node rank.')

    parser.add_argument('--world-size', type=int, help='Number of participating nodes.')

    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Address of the rank-0 node.')

    parser.add_argument('--port', type=int, default=20000, help='Port of the rank-0 node.')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()



    base_train_args = [
        "--script", args.script,
        "--config", args.config,
        "--save_dir", args.save_dir,
        "--seed", str(args.seed),
    ]
    for override in args.cfg_override:
        base_train_args.extend(["--cfg_override", override])
    if args.resume_checkpoint is not None:
        base_train_args.extend(["--resume_checkpoint", args.resume_checkpoint])
    if args.no_resume:
        base_train_args.append("--no_resume")
    if args.mode == "single":
        train_cmd = ["python", "lib/train/run_training.py"] + base_train_args


    elif args.mode == "multiple":
        train_cmd = [
            "python", "-m", "torch.distributed.run",
            "--nproc_per_node", str(args.nproc_per_node),
            "--master_port", str(random.randint(10000, 50000)),
            "lib/train/run_training.py",
        ] + base_train_args


    elif args.mode == "multi_node":
        train_cmd = [
            "python", "-m", "torch.distributed.run",
            "--nproc_per_node", str(args.nproc_per_node),
            "--master_addr", args.ip,
            "--master_port", str(args.port),
            "--nnodes", str(args.world_size),
            "--node_rank", str(args.rank),
            "lib/train/run_training.py",
        ] + base_train_args
    else:
        raise ValueError("mode should be 'single' or 'multiple' or 'multi_node'.")


    subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
