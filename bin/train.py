""" train model """

import deep_semantic_segmentation
import argparse
import os

MODELS = dict(
    deeplab=deep_semantic_segmentation.model.DeepLab
)


def get_options():
    parser = argparse.ArgumentParser(description='Training',
                                     formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-d', '--data', help='Dataset', default='pascal', type=str, **share_param)
    parser.add_argument('-m', '--model', help='Model', default='deeplab', type=str, **share_param)
    parser.add_argument('-b', '--batch_size', help='Batch size', default=None, type=int, **share_param)
    parser.add_argument('-l', '--learning_rate', help='learning rate', default=None, type=float, **share_param)
    parser.add_argument('-w', '--weight_decay', help='weight decay', default=None, type=float, **share_param)
    parser.add_argument('--aspp_depth', help='aspp depth', default=None, type=int, **share_param)
    parser.add_argument('--crop_size', help='crop size', default=None, type=int, **share_param)
    parser.add_argument('--output_stride', help='output_stride', default=None, type=int, **share_param)
    parser.add_argument('--off_decoder', help='unuse decoder', action='store_true')
    parser.add_argument('--off_fine_tune_batch_norm', help='off_fine_tune_batch_norm', action='store_true')
    parser.add_argument('--backbone', help='backbone network', default=None, type=str, **share_param)
    parser.add_argument('--checkpoint', help='checkpoint', default=None, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = get_options()

    model_constructor = MODELS[args.model]

    if args.checkpoint:
        model = model_constructor(checkpoint_version=args.checkpoint)
    else:
        parameters = dict(data_name=args.data)
        if args.crop_size:
            parameters['crop_height'] = args.crop_size
            parameters['crop_width'] = args.crop_size
        if args.batch_size:
            parameters['batch_size'] = args.batch_size
        if args.weight_decay:
            parameters['weight_decay'] = args.weight_decay
        if args.learning_rate:
            parameters['base_learning_rate'] = args.learning_rate
        if args.off_decoder:
            parameters['decoder_output_stride'] = None
        if args.output_stride:
            if args.output_stride == 8:
                parameters['output_stride'] = 8
                parameters['atrous_rate'] = [12, 24, 36]
            elif args.output_stride == 16:
                parameters['output_stride'] = 16
                parameters['atrous_rate'] = [6, 12, 18]
        if args.backbone:
            parameters['model_variant'] = args.backbone
        if args.aspp_depth:
            parameters['depth'] = args.aspp_depth
        if args.off_fine_tune_batch_norm:
            parameters['fine_tune_batch_norm'] = False

        model = model_constructor(**parameters)

    model.train()
