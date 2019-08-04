""" train model """

import deep_semantic_segmentation
import argparse
import os

MODELS = dict(
    deeplab=deep_semantic_segmentation.model.DeepLab
)

def get_options():
    parser = argparse.ArgumentParser(description='Visualize predictions.', formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='Model', default='deeplab', type=str, **share_param)
    parser.add_argument('-d', '--data', help='Dataset', default='pascal', type=str, **share_param)
    parser.add_argument('-c', '--checkpoint', help='Model checkpoint', required=True, type=str, **share_param)
    parser.add_argument('--iter', help='iteration number', default=None, type=int, **share_param)
    parser.add_argument('--data_type', help='Data type (`train`)', default=None, type=float, **share_param)
    parser.add_argument('--seed', help='random seed', default=123, type=int, **share_param)
    parser.add_argument('--training_data', help='use training data', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = get_options()

    model_constructor = MODELS[args.model]
    model = model_constructor(checkpoint_version=args.checkpoint, random_seed=args.seed)
    images, segmentations, predictions = model.predict_dataset(args.iter, is_training=args.training_data)
    print(len(images), images[0].shape)


