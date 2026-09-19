import importlib
import os
from collections import OrderedDict


def create_default_local_file():

    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''


    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'tensorboard_dir': 'self.workspace_dir + \'/tensorboard/\'',
        'pretrained_networks': 'self.workspace_dir + \'/pretrained/\'',
        'lasher_dir': empty_str,
    })


    comment = {
        'workspace_dir': 'Base directory for saving network checkpoints.',
        'tensorboard_dir': 'Directory for tensorboard files.'
    }


    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]


            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def _resolve_lasher_dir(data_dir):
    data_dir = os.path.normpath(data_dir)
    base = os.path.basename(data_dir).lower()
    if base == 'trainingset':
        return data_dir


    if base == 'lasher':
        training_dir = os.path.join(data_dir, 'trainingset')
        if os.path.isdir(training_dir):
            return training_dir
        return data_dir

    candidates = [
        os.path.join(data_dir, 'LasHeR/trainingset'),
        os.path.join(data_dir, 'LasHeR'),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def create_default_local_file_train(workspace_dir, data_dir):
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''


    default_settings = OrderedDict({
        'workspace_dir': workspace_dir,
        'tensorboard_dir': os.path.join(workspace_dir, 'tensorboard'),
        'pretrained_networks': os.path.join(workspace_dir, 'pretrained'),

        'lasher_dir': _resolve_lasher_dir(data_dir),
    })

    comment = {
        'workspace_dir': 'Base directory for saving network checkpoints.',
        'tensorboard_dir': 'Directory for tensorboard files.'
    }

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]

            if comment_str is None:

                if attr_val == empty_str:
                    f.write('        self.{} = {}\n'.format(attr, attr_val))
                else:

                    f.write('        self.{} = {}\n'.format(attr, repr(attr_val)))
            else:

                f.write('        self.{} = {}    # {}\n'.format(attr, repr(attr_val), comment_str))


def env_settings():
    env_module_name = 'lib.train.admin.local'

    try:

        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()

    except ModuleNotFoundError as exc:
        if exc.name != env_module_name:
            raise

        env_file = os.path.join(os.path.dirname(__file__), 'local.py')


        create_default_local_file()


        raise RuntimeError(
            'YOU HAVE NOT SETUP YOUR local.py!!!\n '
            'Go to "{}" and set all the paths you need. Then try to run again.'.format(env_file)
        )
