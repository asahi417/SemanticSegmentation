""" train model """

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
    parser.add_argument('-d', '--data', help='Dataset', default='ade20k', type=str, **share_param)
    parser.add_argument('-m', '--model', help='Model', default='deeplab', type=str, **share_param)
    parser.add_argument('-b', '--batch_size', help='Batch size', default=None, type=int, **share_param)
    parser.add_argument('-l', '--learning_rate', help='learning rate', default=None, type=float, **share_param)
    parser.add_argument('--off_weight_decay', help='weight decay', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = get_options()


    parameters = dict()
    if args.batch_size:
        parameters['batch_size'] = args.batch_size
    if args.off_weight_decay:
        parameters['weight_decay'] = 0.0
    if args.learning_rate:
        parameters['base_learning_rate'] = args.learning_rate

    model_constructor = MODELS[args.model]
    model = model_constructor(data_name=args.data, **parameters)
    model.train(debug=True)
