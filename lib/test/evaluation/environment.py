import importlib
import os


class EnvSettings:
    def __init__(self):
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        project_root = os.path.abspath(os.path.join(test_path, '..', '..'))
        self.prj_dir = project_root
        self.save_dir = os.path.join(project_root, 'output')
        self.network_path = os.path.join(self.save_dir, 'test', 'networks')


def create_default_local_file_test(workspace_dir, data_dir, save_dir):
    path = os.path.join(os.path.dirname(__file__), 'local.py')
    network_path = os.path.join(save_dir, 'test', 'networks')

    with open(path, 'w') as f:
        f.write('from lib.test.evaluation.environment import EnvSettings\n\n')
        f.write('def local_env_settings():\n')
        f.write('    settings = EnvSettings()\n')
        f.write(f"    settings.prj_dir = {workspace_dir!r}\n")
        f.write(f"    settings.save_dir = {save_dir!r}\n")
        f.write(f"    settings.network_path = {network_path!r}\n")
        f.write('    return settings\n')


def env_settings():
    env_module_name = 'lib.test.evaluation.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.local_env_settings()
    except ModuleNotFoundError as exc:
        if exc.name != env_module_name:
            raise
        create_default_local_file_test(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
            '',
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output')),
        )
        raise RuntimeError(
            'YOU HAVE NOT SETUP YOUR local.py. A default file has been created in lib/test/evaluation/local.py.'
        )
