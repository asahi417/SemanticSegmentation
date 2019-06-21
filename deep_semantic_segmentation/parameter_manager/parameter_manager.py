import os
import json
import string
import random
from glob import glob
from . import deeplab
from ..util import CHECKPOINTS


def get_dict_from_instance(__instance):
    return {key: value for key, value in __instance.__dict__.items() if not key.startswith('__') and not callable(key)}


DEFAULT_PARAMETERS = dict(
    DeepLab=dict(
        ade20k=get_dict_from_instance(deeplab.ade20k.Parameter),
        pascal=get_dict_from_instance(deeplab.pascal.Parameter)
    )
)


class ParameterManager:

    def __init__(self,
                 model_name: str = None,
                 data_name: str = None,
                 checkpoint_version: str=None,
                 checkpoint_dir: str=None,
                 debug: bool = False,
                 **kwargs):

        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(CHECKPOINTS, 'model', model_name)

        if checkpoint_version is None:
            if model_name is None or data_name is None:
                raise ValueError('model_name and data_name should not be None.')
            default_dict = DEFAULT_PARAMETERS[model_name][data_name]
            parameter = dict()
            for k, v in default_dict.items():
                if k in kwargs.keys():
                    parameter[k] = kwargs[k]
                else:
                    parameter[k] = v
        else:
            parameter = None

        if debug:
            self.checkpoint_dir = None
            self.parameter = parameter
        else:
            self.checkpoint_dir, self.parameter = self.checkpoint_version(checkpoint_dir, parameter, checkpoint_version)

    def __call__(self, key):
        if key not in self.parameter.keys():
            raise ValueError('unknown parameter %s' % key)
        return self.parameter[key]

    @staticmethod
    def random_string(string_length=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(string_length))

    def checkpoint_version(self,
                           checkpoint_dir: str,
                           config: dict = None,
                           version: str = None):
        """ Checkpoint versioner: Either of `config` or `version` need to be specified (`config` has priority)

         Parameter
        ---------------------
        checkpoint_dir: directory where specific model's checkpoints are (will be) saved, eg) `checkpoint/cnn`
        config: parameter configuration to find same setting checkpoint
        version: checkpoint id

         Return
        --------------------
        path_to_checkpoint, config
        - if there are no checkpoints, having same config as provided one, return new version
            eg) in case there are 'checkpoint/cnn/{v0,v1,v2}`, path_to_checkpoint = 'checkpoint/cnn/v3'
        - if there is a checkpoint, which has same config as provided one, return that version
            eg) in case there are 'checkpoint/cnn/{v0,v1,v2}`, and `v2` has same config, path_to_checkpoint = 'checkpoint/cnn/v2'
        - if `config` is None, `version` is required.
            eg) in case there are 'checkpoint/cnn/{v0,v1,v2}`, path_to_checkpoint = 'checkpoint/cnn/v0' if `version`=0
        """

        if version is None and config is None:
            raise ValueError('either of `version` or `config` is needed.')

        if version is not None:
            checkpoints = glob(os.path.join(checkpoint_dir, version, 'hyperparameters.json'))
            if len(checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated')
            elif len(checkpoints) == 0:
                raise ValueError('No checkpoint: %s' % os.path.join(checkpoint_dir, version))
            else:
                parameter = json.load(open(checkpoints[0]))
                target_checkpoints_dir = checkpoints[0].replace('/hyperparameters.json', '')
                return target_checkpoints_dir, parameter

        elif config is not None:
            # check if there are any checkpoints with same hyperparameters
            target_checkpoints = []
            version_name = []
            for parameter_path in glob(os.path.join(checkpoint_dir, '*/hyperparameters.json')):
                i = parameter_path.replace('/hyperparameters.json', '')
                json_dict = json.load(open(parameter_path))
                version_name.append(i.split('/')[-1])
                if config == json_dict:
                    target_checkpoints.append(i)
            print('Existing checkpoints:', version_name)
            if len(target_checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated')
            elif len(target_checkpoints) == 1:
                if os.path.exists(os.path.join(target_checkpoints[0], 'model.ckpt.index')):
                    inp = input(
                        'found a checkpoint with same configuration\n'
                        'do you want to delete existing checkpoint? (`y` to delete it)')
                    if inp == 'y':
                        os.remove(os.path.join(target_checkpoints[0], 'model.ckpt.index'))
                    return target_checkpoints[0], config

            while True:
                new_version = self.random_string()
                if new_version not in version_name:
                    break
            new_checkpoint_path = os.path.join(checkpoint_dir, new_version)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open(os.path.join(new_checkpoint_path, 'hyperparameters.json'), 'w') as outfile:
                json.dump(config, outfile)
            return new_checkpoint_path, config
