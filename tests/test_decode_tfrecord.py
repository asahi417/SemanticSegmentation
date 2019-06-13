
import deep_semantic_segmentation
import argparse



def get_options():
    parser = argparse.ArgumentParser(description='Building TFRecord.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-d', '--data', help='Dataset.', default='ade20k', type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options()
    model = deep_semantic_segmentation.model.VisImage(data_name=args.data, batch_size=5)

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
