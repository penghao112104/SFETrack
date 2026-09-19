import random
import torch
import torch.utils.data
from lib.utils import TensorDict


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 load_aux_temporal_frames=False, aux_past_gap=1,
                 aux_future_gap=1, aux_temporal_start_epoch=0):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets


        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]


        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        self.load_aux_temporal_frames = load_aux_temporal_frames
        self.aux_past_gap = max(1, int(aux_past_gap))
        self.aux_future_gap = max(1, int(aux_future_gap))
        self.aux_temporal_start_epoch = int(aux_temporal_start_epoch)
        self.current_epoch = 0
        self.max_sample_attempts = 200
        self.max_seq_sample_attempts = 200

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _use_aux_temporal_frames(self):
        if not self.load_aux_temporal_frames:
            return False
        return self.aux_temporal_start_epoch <= 0 or self.current_epoch >= self.aux_temporal_start_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]


        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        return self.getitem()

    def _neighbor_search_candidates(self, visible, valid=None):
        if self.num_search_frames != 1:
            raise ValueError("Neighbor-frame auxiliary sampling currently expects num_search_frames=1.")

        if not torch.is_tensor(visible):
            visible = torch.as_tensor(visible)
        usable = visible.to(torch.bool)

        if valid is not None:
            if not torch.is_tensor(valid):
                valid = torch.as_tensor(valid)
            if len(valid) == len(usable):
                usable = usable & valid.to(torch.bool)

        past_gap = self.aux_past_gap
        future_gap = self.aux_future_gap
        min_search_id = max(past_gap, int(self.num_template_frames))
        max_search_id = len(usable) - future_gap
        if max_search_id <= min_search_id:
            return []

        return [
            i for i in range(min_search_id, max_search_id)
            if bool(usable[i - past_gap]) and bool(usable[i]) and bool(usable[i + future_gap])
        ]

    def _sample_neighbor_search_frame_ids(self, visible, valid=None):
        valid_ids = self._neighbor_search_candidates(visible, valid)
        if len(valid_ids) == 0:
            return None, None, None

        search_id = random.choice(valid_ids)
        return [search_id], [search_id - self.aux_past_gap], [search_id + self.aux_future_gap]

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        attempts = 0
        last_error = None

        while not valid:
            attempts += 1
            if attempts > self.max_sample_attempts:
                msg = "TrackingSampler failed to sample a valid item after {} attempts".format(
                    self.max_sample_attempts
                )
                if last_error is not None:
                    msg += "; last error: {}".format(repr(last_error))
                raise RuntimeError(msg)


            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()


            load_aux_temporal_frames = self._use_aux_temporal_frames()
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)


            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                aux_past_frame_ids = None
                aux_future_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode != 'causal':
                    raise ValueError("Illegal frame sample mode")

                while search_frame_ids is None:
                    if load_aux_temporal_frames:
                        search_frame_ids, aux_past_frame_ids, aux_future_frame_ids = \
                            self._sample_neighbor_search_frame_ids(visible, seq_info_dict.get("valid", None))
                        if search_frame_ids is None:
                            last_error = RuntimeError(
                                "{} sequence {} has no valid t-{}/t/t+{} triplet".format(
                                    dataset.get_name(), seq_id, self.aux_past_gap, self.aux_future_gap
                                )
                            )
                            break
                        base_frame_id = self._sample_visible_ids(
                            visible,
                            num_ids=1,
                            min_id=self.num_template_frames - 1,
                            max_id=search_frame_ids[0],
                        )
                    else:
                        base_frame_id = self._sample_visible_ids(
                            visible,
                            num_ids=1,
                            min_id=self.num_template_frames - 1,
                            max_id=len(visible) - self.num_search_frames,
                        )
                    if base_frame_id is None:
                        gap_increase += 5
                        search_frame_ids = None
                        continue
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        search_frame_ids = None
                        continue
                    template_frame_ids = base_frame_id + prev_frame_ids
                    if not load_aux_temporal_frames:
                        search_frame_ids = self._sample_visible_ids(
                            visible,
                            min_id=template_frame_ids[0] + 1,
                            max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                            num_ids=self.num_search_frames,
                        )

                    gap_increase += 5

                if load_aux_temporal_frames and search_frame_ids is None:
                    continue
            else:

                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
                aux_past_frame_ids = [1] if load_aux_temporal_frames else None
                aux_future_frame_ids = [1] if load_aux_temporal_frames else None
            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})
                if load_aux_temporal_frames:
                    aux_past_frames, aux_past_anno, _ = dataset.get_frames(
                        seq_id, aux_past_frame_ids, seq_info_dict
                    )
                    aux_future_frames, aux_future_anno, _ = dataset.get_frames(
                        seq_id, aux_future_frame_ids, seq_info_dict
                    )
                    data['aux_past_images'] = aux_past_frames
                    data['aux_past_anno'] = aux_past_anno['bbox']
                    data['aux_future_images'] = aux_future_frames
                    data['aux_future_anno'] = aux_future_anno['bbox']
                data = self.processing(data)


                valid = data['valid']
            except Exception as e:
                last_error = e
                valid = False
        return data

    def sample_seq_from_dataset(self, dataset, is_video_dataset):


        enough_visible_frames = False
        attempts = 0
        while not enough_visible_frames:
            attempts += 1
            if attempts > self.max_seq_sample_attempts:
                raise RuntimeError(
                    "Failed to sample a valid sequence from {} after {} attempts. "
                    "If temporal auxiliary training is enabled, check whether the dataset has enough valid "
                    "temporal frame triplets (t-past_gap, t, t+future_gap), "
                    "past_gap={}, future_gap={}.".format(
                        dataset.get_name(), self.max_seq_sample_attempts,
                        self.aux_past_gap, self.aux_future_gap
                    )
                )


            seq_id = random.randint(0, dataset.get_num_sequences() - 1)


            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            if enough_visible_frames and is_video_dataset and self._use_aux_temporal_frames() \
                    and self.frame_sample_mode == 'causal':
                enough_visible_frames = len(
                    self._neighbor_search_candidates(visible, seq_info_dict.get("valid", None))
                ) > 0

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict
