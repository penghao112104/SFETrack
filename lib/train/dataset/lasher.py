import os
import os.path
import numpy as np
import torch
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class LasHeR(BaseVideoDataset):

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None):

        root = env_settings().lasher_dir if root is None else root
        assert split in ['train', 'val','all'], 'Only support all, train or val split in LasHeR, got {}'.format(split)
        super().__init__('LasHeR', root)
        self.dtype = dtype


        self.sequence_list = self._get_sequence_list(split)


        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))


        self.sequence_list = [self.sequence_list[i] for i in seq_ids]


        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        self._frame_path_list_cache = {}

    def get_name(self):
        return 'lasher'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _get_sequence_list(self, split):

        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

        file_path = os.path.join(ltr_path, 'data_specs', 'lasher_{}.txt'.format(split))
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):


        anno_candidates = [
            os.path.join(seq_path, "init.txt"),
            os.path.join(seq_path, "visible.txt"),
            os.path.join(seq_path, "infrared.txt"),
            os.path.join(seq_path, "groundTruth_v.txt"),
            os.path.join(seq_path, "groundTruth_i.txt"),
        ]
        anno_file = next((path for path in anno_candidates if os.path.isfile(path)), None)
        if anno_file is None:
            raise FileNotFoundError(
                "No LasHeR annotation file found for {}. Tried: {}".format(
                    seq_path,
                    ", ".join(anno_candidates),
                )
            )

        last_error = None
        for delimiter in [",", r"\s+", "\t"]:
            try:
                read_kwargs = dict(
                    header=None,
                    dtype=np.float32,
                    na_filter=False,
                )
                if delimiter == r"\s+":
                    read_kwargs.update(sep=delimiter, engine="python")
                else:
                    read_kwargs.update(delimiter=delimiter, low_memory=False)
                rgb_gt = pandas.read_csv(anno_file, **read_kwargs).values
                break
            except Exception as exc:
                last_error = exc
        else:
            raise last_error

        if rgb_gt.ndim == 1:
            rgb_gt = rgb_gt.reshape(1, -1)
        if rgb_gt.shape[1] >= 8:
            x_min = np.min(rgb_gt[:, [0, 2, 4, 6]], axis=1, keepdims=True)
            y_min = np.min(rgb_gt[:, [1, 3, 5, 7]], axis=1, keepdims=True)
            x_max = np.max(rgb_gt[:, [0, 2, 4, 6]], axis=1, keepdims=True)
            y_max = np.max(rgb_gt[:, [1, 3, 5, 7]], axis=1, keepdims=True)
            rgb_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
        elif rgb_gt.shape[1] >= 4:
            rgb_gt = rgb_gt[:, :4]
        else:
            raise ValueError(
                f"LasHeR annotation file {anno_file} must contain at least 4 columns, "
                f"got shape {rgb_gt.shape}."
            )

        return torch.tensor(rgb_gt, dtype=torch.float32)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)


        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        cached_paths = self._frame_path_list_cache.get(seq_path)
        if cached_paths is None:
            visible_dir = os.path.join(seq_path, 'visible')
            infrared_dir = os.path.join(seq_path, 'infrared')

            def _frame_sort_key(filename):
                stem = os.path.splitext(filename)[0]
                digits = ''.join(ch for ch in stem if ch.isdigit())
                if digits:
                    return int(digits), stem
                return -1, stem

            def _list_image_paths(folder):
                if not os.path.isdir(folder):
                    raise FileNotFoundError(f"LasHeR image folder not found: {folder}")
                image_names = [
                    name for name in os.listdir(folder)
                    if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ]
                image_names.sort(key=_frame_sort_key)
                return [os.path.join(folder, name) for name in image_names]

            rgb_paths = _list_image_paths(visible_dir)
            ir_paths = _list_image_paths(infrared_dir)
            if len(rgb_paths) == 0 or len(ir_paths) == 0:
                raise FileNotFoundError(
                    f"LasHeR sequence has no images: visible={visible_dir}, infrared={infrared_dir}"
                )
            cached_paths = (rgb_paths, ir_paths)
            self._frame_path_list_cache[seq_path] = cached_paths

        rgb_paths, ir_paths = cached_paths
        if frame_id < 0 or frame_id >= len(rgb_paths) or frame_id >= len(ir_paths):
            raise IndexError(
                f"LasHeR frame_id={frame_id} out of range for {seq_path}: "
                f"visible={len(rgb_paths)}, infrared={len(ir_paths)}"
            )
        return rgb_paths[frame_id], ir_paths[frame_id]

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)


        img = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)


        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]


        if anno is None:
            anno = self.get_sequence_info(seq_id)


        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]


        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})


        return frame_list, anno_frames, object_meta
