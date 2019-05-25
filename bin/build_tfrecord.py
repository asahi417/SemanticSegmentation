""" script to build tfrecord """

import deep_semantic_segmentation
import argparse


def get_options():
    parser = argparse.ArgumentParser(description='Decode tfrecord.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-d', '--data', help='Dataset.', default='ade20k', type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options()
    recorder = deep_semantic_segmentation.data.TFRecord(data_name=args.data)
    recorder.convert_to_tfrecord()
