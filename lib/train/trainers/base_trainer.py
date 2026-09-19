import os
import glob
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler


def _latest_checkpoint(directory, net_type):
    checkpoint_list = sorted(glob.glob(os.path.join(directory, f"{net_type}_ep*.pth.tar")))
    return checkpoint_list[-1] if checkpoint_list else None


class BaseTrainer:

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}


        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")


        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:

            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)


            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)


            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None


    def train(self, max_epochs, load_latest=False, fail_safe=True):
        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:

                if load_latest:

                    self.load_checkpoint()
                if self.epoch >= max_epochs:
                    rk = getattr(self.settings, "local_rank", -1)
                    if rk in (-1, 0):
                        print(
                            f"[train] loaded epoch {self.epoch} >= target max_epochs {max_epochs}; "
                            "no epochs left to run. Increase TRAIN.EPOCH or move old checkpoints."
                        )


                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch


                    self.train_epoch()


                    if self.lr_scheduler is not None:

                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)


                    save_epoch_interval = getattr(self.settings, "save_epoch_interval", 1)
                    save_last_n_epoch = getattr(self.settings, "save_last_n_epoch", 1)

                    save_periodically = (
                        save_epoch_interval > 0 and epoch % save_epoch_interval == 0
                    )
                    save_in_final_epochs = (
                        save_last_n_epoch > 0 and epoch > (max_epochs - save_last_n_epoch)
                    )

                    if save_periodically or save_in_final_epochs:
                        if self._checkpoint_dir:

                            if self.settings.local_rank in [-1, 0]:
                                self.save_checkpoint()
            except:

                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe and i < num_tries - 1:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')


    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self):

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__


        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'settings': self.settings
        }


        directory = self._checkpoint_dir
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)


        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)


        os.rename(tmp_file_path, file_path)

    def _drop_incompatible_optimizer_states(self):
        """Clear optimizer state tensors whose shape no longer matches the parameter."""
        dropped = 0
        for group in self.optimizer.param_groups:
            for param in group.get("params", []):
                state = self.optimizer.state.get(param, None)
                if not isinstance(state, dict) or len(state) == 0:
                    continue
                incompatible = False
                for value in state.values():
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape != param.shape:
                        incompatible = True
                        break
                if incompatible:
                    self.optimizer.state[param] = {}
                    dropped += 1
        return dropped

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__


        if checkpoint is None:

            checkpoint_path = _latest_checkpoint(self._checkpoint_dir, net_type)
            if checkpoint_path is None:
                legacy_dir = os.path.join(self._checkpoint_dir, self.settings.project_path)
                checkpoint_path = _latest_checkpoint(legacy_dir, net_type)
            if checkpoint_path is None:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):

            checkpoint_path = os.path.join(self._checkpoint_dir, f"{net_type}_ep{checkpoint:04d}.pth.tar")
            if not os.path.isfile(checkpoint_path):
                legacy_path = os.path.join(
                    self._checkpoint_dir,
                    self.settings.project_path,
                    f"{net_type}_ep{checkpoint:04d}.pth.tar",
                )
                if os.path.isfile(legacy_path):
                    checkpoint_path = legacy_path
        elif isinstance(checkpoint, str):

            if os.path.isdir(checkpoint):

                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:

                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        rk = getattr(self.settings, "local_rank", -1)
        if rk in (-1, 0):
            print(f"[checkpoint] loading: {checkpoint_path}")


        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']


        ignore_fields.extend([
            'lr_scheduler',
            'constructor',
            'net_type',
            'actor_type',
            'net_info',
            'stats',
        ])


        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':

                missing_k, unexpected_k = net.load_state_dict(checkpoint_dict[key], strict=False)
                rk = getattr(self.settings, "local_rank", -1)
                if rk in (-1, 0):
                    if len(missing_k) > 0:
                        print(f"[checkpoint] missing keys ({len(missing_k)}): {missing_k}")
                    if len(unexpected_k) > 0:
                        print(f"[checkpoint] unexpected keys ({len(unexpected_k)}): {unexpected_k}")
            elif key == 'optimizer':
                try:
                    self.optimizer.load_state_dict(checkpoint_dict[key])
                    dropped_states = self._drop_incompatible_optimizer_states()
                    rk = getattr(self.settings, "local_rank", -1)
                    if dropped_states > 0 and rk in (-1, 0):
                        print(
                            "[checkpoint] optimizer states cleared for "
                            f"{dropped_states} parameter(s) with incompatible shapes"
                        )
                except (ValueError, RuntimeError) as e:

                    rk = getattr(self.settings, "local_rank", -1)
                    if rk in (-1, 0):
                        print(
                            "[checkpoint] optimizer state skipped (param_groups mismatch or incompatible); "
                            f"reason: {e}"
                        )
            else:
                setattr(self, key, checkpoint_dict[key])
        if rk in (-1, 0):
            print(f"[checkpoint] loaded epoch: {self.epoch}")


        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']


        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch

            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
        return True

    def load_state_dict(self, checkpoint=None):

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__


        if isinstance(checkpoint, str):
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError


        print("Loading pretrained model from ", checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'


        missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)

        return True
