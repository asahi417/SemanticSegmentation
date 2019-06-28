""" script to build tfrecord """

import deep_semantic_segmentation
import argparse
import os


def get_options():
    parser = argparse.ArgumentParser(description='Encoding tfrecord.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-d', '--data', help='Dataset name (`pascal` or `ade20k`)', default='ade20k', type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = get_options()
    recorder = deep_semantic_segmentation.data.TFRecord(data_name=args.data)
    recorder.convert_to_tfrecord()
