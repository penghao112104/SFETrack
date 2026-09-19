import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class ViPTProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(
            torch.randn(2, device=box.device, dtype=box.dtype)
            * self.scale_jitter_factor[mode]
        )
        center_factor = torch.as_tensor(
            self.center_jitter_factor[mode], device=box.device, dtype=box.dtype
        )
        max_offset = jittered_size.prod().sqrt() * center_factor
        center_noise = torch.rand(2, device=box.device, dtype=box.dtype) - 0.5
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * center_noise

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
                images: list of np.ndarray [(H,W,6)]
                anno: list of torch.Tensor [(4,)]
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno'
        """

        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'])
            data['search_images'], data['search_anno'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], new_roll=False)
            if 'aux_past_images' in data and 'aux_past_anno' in data:
                data['aux_past_images'], data['aux_past_anno'] = self.transform['joint'](
                    image=data['aux_past_images'], bbox=data['aux_past_anno'], new_roll=False)
            if 'aux_future_images' in data and 'aux_future_anno' in data:
                data['aux_future_images'], data['aux_future_anno'] = self.transform['joint'](
                    image=data['aux_future_images'], bbox=data['aux_future_anno'], new_roll=False)

        branches = [('template', 'template'), ('search', 'search')]
        if 'aux_past_images' in data and 'aux_past_anno' in data:
            branches.append(('aux_past', 'search'))

        for s, crop_mode in branches:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"


            jittered_anno = [
                self._get_jittered_box(a, crop_mode)
                for a in data[s + '_anno']
            ]


            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[crop_mode])
            if (crop_sz < 1).any():
                data['valid'] = False

                return data


            crops, boxes, _, _ = prutils.jittered_center_crop(
                data[s + '_images'],
                jittered_anno,
                data[s + '_anno'],
                self.search_area_factor[crop_mode],
                self.output_sz[crop_mode],
            )


            transform_kwargs = {
                'image': crops,
                'bbox': boxes,
                'joint': False,
            }
            if s == 'aux_past':


                transform_kwargs['new_roll'] = False
            data[s + '_images'], data[s + '_anno'] = self.transform[crop_mode](**transform_kwargs)

            if s == 'search' and 'aux_future_images' in data and 'aux_future_anno' in data:

                future_crop_anno = [
                    self._get_jittered_box(a, crop_mode)
                    for a in data['aux_future_anno']
                ]
                future_w = torch.stack(future_crop_anno, dim=0)[:, 2]
                future_h = torch.stack(future_crop_anno, dim=0)[:, 3]
                future_crop_sz = torch.ceil(
                    torch.sqrt(future_w * future_h) * self.search_area_factor[crop_mode]
                )
                if (future_crop_sz < 1).any():
                    data['valid'] = False
                    return data
                future_crops, future_boxes, _, _ = prutils.jittered_center_crop(
                    data['aux_future_images'], future_crop_anno,
                    data['aux_future_anno'], self.search_area_factor[crop_mode],
                    self.output_sz[crop_mode])
                data['aux_future_images'], data['aux_future_anno'] = self.transform[crop_mode](
                    image=future_crops, bbox=future_boxes, joint=False, new_roll=False
                )

        data['valid'] = True

        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class BATProcessing(ViPTProcessing):
    """Compatibility alias used by BAT training configs."""
    pass
