""" Tensorflow graph debug """

import deep_semantic_segmentation
import argparse
import os
import shutil


MODELS = dict(
    deeplab=deep_semantic_segmentation.model.DeepLab
)


def get_options():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-d', '--data', help='Dataset', default='ade20k', type=str, **share_param)
    parser.add_argument('-m', '--model', help='Model', default='deeplab', type=str, **share_param)
    parser.add_argument('-n', '--network', help='Network to test.', default='xception_65', type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options()
    model_constructor = MODELS[args.model]
    graph = model_constructor(checkpoint='./tmp.model', data_name=args.data, model_variant=args.network)
    shutil.rmtree('./tmp.model', ignore_errors=True)
