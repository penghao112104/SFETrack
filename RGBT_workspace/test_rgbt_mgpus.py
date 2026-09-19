import os
import cv2
import re
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.bat import BATTrack
import lib.test.parameter.bat as rgbt_adapter_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time


torch.set_num_threads(1)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def load_bbox_anno(path):
    last_error = None
    for delimiter in [',', None, '\t', ' ']:
        try:
            anno = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
            if anno.size == 0:
                last_error = ValueError(f"empty annotation file: {path}")
                continue
            if anno.ndim == 1:
                anno = anno.reshape(1, -1)
            return anno
        except Exception as exc:
            last_error = exc
    raise last_error


def load_gtot_anno(seq_path):
    try:
        rgb_gt = load_bbox_anno(join(seq_path, 'groundTruth_v.txt'))
        t_gt = load_bbox_anno(join(seq_path, 'groundTruth_i.txt'))

        x_min = np.min(rgb_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(rgb_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(rgb_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(rgb_gt[:, [1, 3]], axis=1)[:, None]
        rgb_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

        x_min = np.min(t_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(t_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(t_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(t_gt[:, [1, 3]], axis=1)[:, None]
        t_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
        return rgb_gt, t_gt
    except Exception as exc:
        init_path = join(seq_path, 'init.txt')
        if not os.path.isfile(init_path):
            raise exc
        init_gt = load_bbox_anno(init_path)
        return init_gt, init_gt.copy()


def natural_path_key(path):
    name = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', name)
    ]


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        [
            join(folder, p) for p in os.listdir(folder)
            if os.path.splitext(p)[1].lower() in IMG_EXTENSIONS
        ],
        key=natural_path_key,
    )


def first_existing_dir(seq_path, names):
    for name in names:
        path = join(seq_path, name)
        if os.path.isdir(path):
            return path
    return join(seq_path, names[0])


def genConfig(seq_path, set_type):
    if set_type == 'RGBT210':
        RGB_img_list = list_images(join(seq_path, 'visible'))
        T_img_list = list_images(join(seq_path, 'infrared'))

        RGB_gt = load_bbox_anno(seq_path + '/init.txt')
        T_gt = RGB_gt.copy()

    elif set_type == 'RGBT234':
        RGB_img_list = list_images(join(seq_path, 'visible'))
        T_img_list = list_images(join(seq_path, 'infrared'))

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'GTOT':
        rgb_dir = first_existing_dir(seq_path, ['v', 'visible'])
        t_dir = first_existing_dir(seq_path, ['i', 'infrared'])
        RGB_img_list = list_images(rgb_dir)
        T_img_list = list_images(t_dir)

        RGB_gt, T_gt = load_gtot_anno(seq_path)

    elif set_type == 'LasHeR':
        RGB_img_list = list_images(join(seq_path, 'visible'))
        T_img_list = list_images(join(seq_path, 'infrared'))

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(
        seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0,
        script_name='bat'):
    seq_txt = seq_name
    result_root = os.environ.get("RGBT_RESULT_ROOT") or './RGBT_workspace/results'
    save_name = os.environ.get("RGBT_SAVE_NAME") or 'SFETrack'
    save_folder = os.path.join(result_root, dataset_name, save_name)
    save_path = os.path.join(save_folder, seq_txt + '.txt')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if num_gpu < 1:
        raise ValueError(f"num_gpu must be positive, got {num_gpu}")
    worker_identity = multiprocessing.current_process()._identity
    worker_id = worker_identity[0] - 1 if worker_identity else 0
    torch.cuda.set_device(worker_id % num_gpu)

    if script_name != 'bat':
        raise ValueError(f"Unsupported tracker: {script_name}")
    params = rgbt_adapter_params.parameters(yaml_name, epoch)
    mmtrack = BATTrack(params)
    mmtrack.dataset_name = dataset_name
    mmtrack.sequence_name = seq_name
    tracker = BAT_RGBT(tracker=mmtrack)

    seq_path = seq_home + '/' + seq_name
    print(f'Processing sequence: {seq_name}')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == 0 or len(T_img_list) == 0:
        raise ValueError(
            f"No images found for {dataset_name}/{seq_name}. "
            f"visible={len(RGB_img_list)}, infrared={len(T_img_list)}, seq_path={seq_path}"
        )
    if len(RGB_img_list) != len(T_img_list):
        raise ValueError(
            f"RGB/TIR frame-count mismatch for {dataset_name}/{seq_name}: "
            f"visible={len(RGB_img_list)}, infrared={len(T_img_list)}"
        )
    if len(RGB_gt) == 0:
        raise ValueError(f"No annotations loaded for {dataset_name}/{seq_name}, seq_path={seq_path}")
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])

    toc = 0
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        image = get_x_frame(
            rgb_path,
            T_path,
            dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'),
        )
        if frame_idx == 0:
            tracker.initialize(image, RGB_gt[0].tolist())
        else:
            region, _ = tracker.track(image)
            result[frame_idx] = np.asarray(region, dtype=result.dtype)
        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result)
    tracked_frames = max(0, len(RGB_img_list) - 1)
    fps = tracked_frames / toc if toc > 0 else 0.0
    print('{} , fps:{}'.format(seq_name, fps))


class BAT_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)

        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='bat', choices=['bat'])
    parser.add_argument('--yaml_name', type=str, default='rgbt', help='Experiment YAML name.')
    parser.add_argument('--dataset_name', type=str, default='LasHeR', choices=['GTOT', 'RGBT210', 'RGBT234', 'LasHeR'])
    parser.add_argument('--seq_home', type=str, required=True, help='Dataset root containing sequence directories.')
    parser.add_argument('--threads', default=1, type=int, help='Number of worker processes.')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=None, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    seq_home = abspath(os.path.expanduser(args.seq_home))
    if not os.path.isdir(seq_home):
        raise FileNotFoundError(f"Dataset root does not exist: {seq_home}")
    seq_list = sorted(
        (f for f in os.listdir(seq_home) if isdir(join(seq_home, f))),
        key=natural_path_key,
    )
    if not seq_list:
        raise ValueError(f"No sequence directories found under: {seq_home}")

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [
            (
                s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch,
                args.debug, args.script_name,
            )
            for s in seq_list
        ]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [
            (
                s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch,
                args.debug, args.script_name,
            )
            for s in seq_list
        ]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")
