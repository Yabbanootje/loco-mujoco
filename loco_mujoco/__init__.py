from pathlib import Path
import os
import yaml

__version__ = '1.0.1'


try:

    PATH_TO_MODELS = Path(__file__).resolve().parent / "models"
    PATH_TO_VARIABLES = Path(__file__).resolve().parent / "LOCOMUJOCO_VARIABLES.yaml"
    PATH_TO_SMPL_ROBOT_CONF = Path(__file__).resolve().parent / "smpl" / "robot_confs"

    from .core import Mujoco, Mjx
    from .environments import LocoEnv
    from .task_factories import (TaskFactory, RLFactory, ImitationFactory)

    def get_registered_envs():
        return LocoEnv.registered_envs

    def get_variable(key):
        try:
            return yaml.load(open(PATH_TO_VARIABLES), Loader=yaml.FullLoader)[key]
        except KeyError:
            return None

except ImportError as e:
    print(e)
