""" test model """

import deep_semantic_segmentation
import argparse
import os

MODELS = dict(
    deeplab=deep_semantic_segmentation.model.DeepLab
)


def get_options():
    parser = argparse.ArgumentParser(description='Decode tfrecord.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='Model', default='deeplab', type=str, **share_param)
    parser.add_argument('--checkpoint', help='Checkpoint', required=True, type=str, **share_param)
    parser.add_argument('--training_data', help='training_data', action='store_true')
    parser.add_argument('--training_setting', help='training_setting', action='store_true')
    parser.add_argument('--training_process', help='training_process', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = get_options()

    model_constructor = MODELS[args.model]
    model = model_constructor(checkpoint_version=args.checkpoint)
    model.test(is_training_data=args.training_data,
               is_training=args.training_setting,
               is_training_process=args.training_process)

