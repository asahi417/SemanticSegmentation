""" Tensorflow graph debug """

import deep_semantic_segmentation
import argparse


def get_options():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-d', '--data', help='Data to test.', default='ade20k', type=str, **share_param)
    parser.add_argument('-m', '--model', help='Backbone model type.', default='xception_65', type=str, **share_param)
    parser.add_argument('-n', '--network', help='Network to test.', default='deeplab_v3', type=str, **share_param)
    return parser.parse_args()


def map_to_graph(name):
    if name == 'deeplab_v3':
        return deep_semantic_segmentation.model.DeepLab3
    elif name == 'test':
        return deep_semantic_segmentation.model.BaseModel
    else:
        raise ValueError('unknown name %s' % name)


if __name__ == '__main__':
    args = get_options()
    instance = map_to_graph(args.network)
    graph = instance(checkpoint='tmp.model', data_name=args.data, model_variant=args.model)
    print(graph.test(True).shape)
    print(graph.test(False).shape)

