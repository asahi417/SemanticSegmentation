""" train model """

import deep_semantic_segmentation
import argparse
import os
import numpy as np
import scipy.misc as misc


MODELS = dict(
    deeplab=deep_semantic_segmentation.model.DeepLab
)

def get_options():
    parser = argparse.ArgumentParser(description='Visualize predictions.', formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='Model', default='deeplab', type=str, **share_param)
    parser.add_argument('-c', '--checkpoint', help='Model checkpoint', required=True, type=str, **share_param)
    parser.add_argument('--path', help='path to save output', default='./sample.jpg', type=str, **share_param)
    parser.add_argument('-n', '--number', help='image size', default=5, type=int, **share_param)
    parser.add_argument('--seed', help='random seed', default=123, type=int, **share_param)
    parser.add_argument('--training_data', help='use training data', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = get_options()

    model_constructor = MODELS[args.model]
    model = model_constructor(checkpoint_version=args.checkpoint, random_seed=args.seed)
    images, segmentations, predictions = model.predict_from_data(args.number, is_training=args.training_data)

    row_n = args.number  # rows
    col_n = 3  # cols
    margin = 10  # margin for output
    canvas_shape = (
        row_n * model.option('crop_height') + (margin * row_n) + margin,
        col_n * model.option('crop_width') + (margin * col_n) + margin,
        3)
    canvas = 255 * np.ones(canvas_shape, dtype=np.uint8)

    start_y = margin

    for i in range(args.number):

        segmentation = deep_semantic_segmentation.util.coloring_segmentation(segmentations[i])
        prediction = deep_semantic_segmentation.util.coloring_segmentation(predictions[i])

        start_x = margin
        print(start_y, start_y + model.option('crop_height'), canvas.shape)

        canvas[start_y:start_y + model.option('crop_height'), start_x:start_x+model.option('crop_width'), :] = images[i].astype(np.uint8)
        start_x += model.option('crop_width') + margin
        canvas[start_y:start_y + model.option('crop_height'), start_x:start_x+model.option('crop_width'), :] = segmentation
        start_x += model.option('crop_width') + margin
        canvas[start_y:start_y + model.option('crop_height'), start_x:start_x+model.option('crop_width'), :] = prediction

        start_y +=  model.option('crop_height') + margin

    misc.imsave(args.path, canvas)

# misc.imsave('./bin/img/generated_img/%s-%s-v%s.jpg' % (args.model, args.data, version), canvas)
