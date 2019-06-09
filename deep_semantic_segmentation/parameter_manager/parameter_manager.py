import os
import json
from glob import glob
from . import default_parameters
from ..util import CHECKPOINTS
from ..data import VALID_DATA_NAME


def get_dict_from_instance(__instance):
    return {key: value for key, value in __instance.__dict__.items() if not key.startswith('__') and not callable(key)}


DEFAULT_PARAMETERS = dict(
    DeepLab=get_dict_from_instance(default_parameters.DeepLab)
)


class ParameterManager:

    def __init__(self,
                 model_name,
                 checkpoint_version: int=None,
                 checkpoint_dir: str=None,
                 **kwargs):

        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(CHECKPOINTS, 'model', model_name)
        self.default_dict = DEFAULT_PARAMETERS[model_name]

        if checkpoint_version is None:
            parameter = dict()
            for k, v in self.default_dict.items():
                if k in kwargs.keys():
                    parameter[k] = kwargs[k]
                else:
                    parameter[k] = v
            height, width = VALID_DATA_NAME[self.default_dict['data_name']]['shape']
            parameter['crop_size_height'] = height
            parameter['crop_size_width'] = width
        else:
            parameter = None
        self.checkpoint_dir, self.parameter = self.checkpoint_version(checkpoint_dir, parameter, checkpoint_version)

    def __call__(self, key):
        if key not in self.parameter.keys():
            raise ValueError('unknown parameter %s' % key)
        return self.parameter[key]

    @staticmethod
    def checkpoint_version(checkpoint_dir: str,
                           config: dict = None,
                           version: int = None):
        """ Checkpoint versioner: Either of `config` or `version` need to be specified (`config` has priority)

         Parameter
        ---------------------
        checkpoint_dir: directory where specific model's checkpoints are (will be) saved, eg) `checkpoint/cnn`
        config: parameter configuration to find same setting checkpoint
        version: number of checkpoint to warmstart from

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
            checkpoints = glob(os.path.join(checkpoint_dir, 'v%i' % version, 'hyperparameters.json'))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint: %s, %s' % (checkpoint_dir, version))
            elif len(checkpoints) > 1:
                raise ValueError('Multiple checkpoint found: %s, %s' % (checkpoint_dir, version))
            else:
                parameter = json.load(open(checkpoints[0]))
                target_checkpoints_dir = checkpoints[0].replace('/hyperparameters.json', '')
                return target_checkpoints_dir, parameter

        elif config is not None:
            # check if there are any checkpoints with same hyperparameters
            target_checkpoints = []
            for parameter_path in glob(os.path.join(checkpoint_dir, '*/hyperparameters.json')):
                i = parameter_path.replace('/hyperparameters.json', '')
                json_dict = json.load(open(parameter_path))
                if config == json_dict:
                    target_checkpoints.append(i)
            if len(target_checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated')
            elif len(target_checkpoints) == 1:
                if os.path.exists(os.path.join(target_checkpoints[0], 'model.ckpt.index')):
                    inp = input(
                        'found a checkpoint with same configuration\n'
                        'do you want to delet existing checkpoint? (`y` to delete it)')
                    if inp == 'y':
                        os.remove(os.path.join(target_checkpoints[0], 'model.ckpt.index'))
                    return target_checkpoints[0], config

            new_checkpoint_id = len(glob(os.path.join(checkpoint_dir, '*/hyperparameters.json')))
            new_checkpoint_path = os.path.join(checkpoint_dir, 'v%i' % new_checkpoint_id)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open(os.path.join(new_checkpoint_path, 'hyperparameters.json'), 'w') as outfile:
                json.dump(config, outfile)
            return new_checkpoint_path, config
