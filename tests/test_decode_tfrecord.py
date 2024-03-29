
import deep_semantic_segmentation
import argparse
import os


def get_options():
    parser = argparse.ArgumentParser(description='Decode tfrecord.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-d', '--data', help='Dataset', default='ade20k', type=str, **share_param)
    parser.add_argument('-m', '--model', help='Model', default='deeplab', type=str, **share_param)
    parser.add_argument('-b', '--batch_size', help='Batch size', default=None, type=int, **share_param)
    parser.add_argument('-l', '--learning_rate', help='learning rate', default=None, type=float, **share_param)
    parser.add_argument('-w', '--weight_decay', help='weight decay', default=None, type=float, **share_param)
    parser.add_argument('--crop_size', help='crop size', default=None, type=int, **share_param)
    parser.add_argument('--off_decoder', help='unuse decoder', action='store_true')
    parser.add_argument('--output_stride', help='output_stride', default=None, type=int, **share_param)
    parser.add_argument('--decoder_batch_norm', help='decoder_batch_norm', action='store_true')
    parser.add_argument('--aspp_batch_norm', help='aspp_batch_norm', action='store_true')
    parser.add_argument('--backbone', help='Backbone', default=None, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = get_options()


    parameters = dict()
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
    if args.decoder_batch_norm:
        parameters['decoder_batch_norm'] = True
    if args.aspp_batch_norm:
            parameters['aspp_batch_norm'] = True
    if args.backbone:
        parameters['model_variant'] = args.backbone

    model = deep_semantic_segmentation.model.VisImage(data_name=args.data, **parameters)

    while True:
        inp = input('0: Train, 1: Valid >>>')
        if inp == 'q':
            break
        if inp == '0':
            model.setup(is_training=True)
        elif inp == '1':
            model.setup(is_training=False)
        else:
            model.array_to_image()
